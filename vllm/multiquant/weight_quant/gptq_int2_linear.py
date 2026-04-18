# SPDX-License-Identifier: Apache-2.0
"""GPTQ INT2/INT3 Linear — direct dequant without Marlin repack.

AutoRound INT2/INT3 models use standard GPTQ format (qweight/scales/qzeros).
Marlin doesn't support sub-4-bit natively, so we dequant on-the-fly.
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter

logger = init_logger(__name__)


def dequant_gptq_sub4(
    qweight: torch.Tensor,    # [K_packed, N] int32
    scales: torch.Tensor,      # [n_groups, N] float16
    qzeros: torch.Tensor,      # [n_groups, N_zp_packed] int32
    group_size: int,
    bits: int,
) -> torch.Tensor:
    """Dequantize GPTQ sub-4-bit packed weights to float16.

    Supports:
    - bits=2: pack_factor=16, simple shift+mask (16 values per int32)
    - bits=3: bit-stream packing (ceil(K*3/32) int32 per column)

    Returns: [N, K] float16 weight matrix (ready for F.linear).
    """
    K_packed, N = qweight.shape
    mask = (1 << bits) - 1

    if bits == 2:
        # Simple: 16 values per int32
        pack_factor = 16
        K = K_packed * pack_factor
        shifts = torch.arange(0, 32, 2, device=qweight.device,
                              dtype=torch.int32)
        expanded = qweight.unsqueeze(1).expand(-1, pack_factor, -1)
        unpacked = ((expanded >> shifts.view(1, -1, 1)) & mask).reshape(K, N)
    elif bits == 3:
        # Bit-stream: K = K_packed * 32 / 3 (rounded)
        K = (K_packed * 32) // 3
        # Unpack via bit manipulation
        # Flatten to 1D per column, extract 3-bit values
        unpacked_list = []
        for col in range(N):
            col_data = qweight[:, col].cpu().tolist()
            vals = []
            bit_buf = 0
            bits_in_buf = 0
            for word in col_data:
                bit_buf |= (word & 0xFFFFFFFF) << bits_in_buf
                bits_in_buf += 32
                while bits_in_buf >= 3 and len(vals) < K:
                    vals.append(bit_buf & 0x7)
                    bit_buf >>= 3
                    bits_in_buf -= 3
            while len(vals) < K:
                vals.append(0)
            unpacked_list.append(vals[:K])
        unpacked = torch.tensor(unpacked_list, device=qweight.device,
                                dtype=torch.int32).T  # [K, N]
    else:
        raise ValueError(f"Unsupported bits={bits}")

    n_groups = K // group_size

    # Unpack zero points (same bit-packing as weights)
    if bits == 2:
        zp_shifts = torch.arange(0, 32, 2, device=qzeros.device,
                                 dtype=torch.int32)
        zp_expanded = qzeros.unsqueeze(1).expand(-1, 16, -1)
        zp_all = ((zp_expanded >> zp_shifts.view(1, -1, 1)) & mask)
        zp_all = zp_all.reshape(n_groups, -1)[:, :N]
    elif bits == 3:
        # Zero points also bit-packed
        zp_packed_n = qzeros.shape[1]
        zp_K = (zp_packed_n * 32) // 3
        zp_list = []
        for g in range(n_groups):
            row = qzeros[g].cpu().tolist()
            vals = []
            bit_buf = 0
            bits_in_buf = 0
            for word in row:
                bit_buf |= (word & 0xFFFFFFFF) << bits_in_buf
                bits_in_buf += 32
                while bits_in_buf >= 3 and len(vals) < N:
                    vals.append(bit_buf & 0x7)
                    bit_buf >>= 3
                    bits_in_buf -= 3
            while len(vals) < N:
                vals.append(0)
            zp_list.append(vals[:N])
        zp_all = torch.tensor(zp_list, device=qzeros.device,
                              dtype=torch.int32)  # [n_groups, N]

    # Dequant: weight = scale * (qval - zero_point)
    group_idx = torch.arange(K, device=qweight.device) // group_size
    w = scales[group_idx] * (unpacked.float() - zp_all[group_idx].float())

    return w.T.contiguous().to(torch.float16)


class GPTQInt2LinearMethod(QuantizeMethodBase):
    """Linear method for GPTQ INT2/INT3 without Marlin.

    Dequantizes on-the-fly per forward pass.
    Slower than Marlin but correct for sub-4-bit.
    """

    def __init__(self, group_size: int = 128, bits: int = 2):
        self.group_size = group_size
        self.bits = bits

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        pack_factor = 32 // self.bits

        n_groups = input_size_per_partition // self.group_size
        zp_packed_n = (output_size_per_partition + pack_factor - 1) // pack_factor

        def _default_loader(param, loaded_weight, *args):
            param.data.copy_(loaded_weight)

        for name, shape, dtype in [
            ("qweight", (input_size_per_partition // pack_factor,
                         output_size_per_partition), torch.int32),
            ("scales", (n_groups, output_size_per_partition), torch.float16),
            ("qzeros", (n_groups, zp_packed_n), torch.int32),
        ]:
            p = torch.nn.Parameter(
                torch.empty(*shape, dtype=dtype), requires_grad=False)
            p.weight_loader = _default_loader
            layer.register_parameter(name, p)

        # Store config on layer for apply()
        layer._gptq_bits = self.bits
        layer._gptq_group_size = self.group_size

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        W = dequant_gptq_sub4(
            layer.qweight.data,
            layer.scales.data,
            layer.qzeros.data,
            layer._gptq_group_size,
            layer._gptq_bits,
        )
        out = F.linear(x, W, bias)
        return out
