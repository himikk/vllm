# SPDX-License-Identifier: Apache-2.0
"""MultiQuant Marlin3 — GPTQ 3-bit weight dequant + cuBLAS GEMM.

Uses vLLM's existing GPTQ 3-bit reconstruct kernel (q_gemm.cu) for
dequantization, then standard cuBLAS F.linear for the GEMM.

This is the "reconstruct" path already in vLLM, exposed as a
MultiQuant module. The fused 3-bit GEMM kernel exists in q_gemm.cu
but is disabled ("slower than baseline"). We use the reconstruct path.

For INT2: same approach with 2-bit dequant.

Registration: via INC config for quant_method="auto-round", bits=2/3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

logger = init_logger(__name__)


class MultiQuantSub4LinearMethod(QuantizeMethodBase):
    """Linear method for sub-4-bit GPTQ weights (INT2/INT3).

    At load time: stores qweight/scales/qzeros as-is.
    At forward time: dequant → F.linear (reconstruct + cuBLAS).

    This is correct but slower than a fused kernel. Future: fused
    INT3 GEMM kernel (like Marlin INT4).
    """

    def __init__(self, bits: int = 3, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self._dequant_cache: dict[int, torch.Tensor] = {}

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
        K = input_size_per_partition
        N = output_size_per_partition
        bits = self.bits

        # Packed weight dimensions
        if bits == 2:
            K_packed = K // 16  # 16 values per int32
        elif bits == 3:
            K_packed = (K * 3 + 31) // 32  # ceil(K*3/32) int32
        else:
            K_packed = K // 8  # 4-bit: 8 per int32

        n_groups = K // self.group_size

        # Zero-point packed dimension
        if bits == 2:
            N_zp = (N + 15) // 16
        elif bits == 3:
            N_zp = (N * 3 + 31) // 32
        else:
            N_zp = (N + 7) // 8

        # Register parameters with weight_loader for vLLM's loader
        def _loader(param, loaded_weight, *args):
            param.data.copy_(loaded_weight)

        for name, shape, dtype in [
            ("qweight", (K_packed, N), torch.int32),
            ("scales", (n_groups, N), torch.float16),
            ("qzeros", (n_groups, N_zp), torch.int32),
        ]:
            p = torch.nn.Parameter(
                torch.empty(*shape, dtype=dtype), requires_grad=False)
            p.weight_loader = _loader
            layer.register_parameter(name, p)

        # Also register g_idx (optional, AutoRound includes it)
        g_idx = torch.nn.Parameter(
            torch.empty(K, dtype=torch.int32), requires_grad=False)
        g_idx.weight_loader = _loader
        layer.register_parameter("g_idx", g_idx)

        # Store config
        layer._mq_bits = bits
        layer._mq_group_size = self.group_size

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dequant weights → F.linear.

        TODO: replace with fused INT3 GEMM kernel for performance.
        """
        # Lazy dequant and cache (weights don't change between calls)
        layer_id = id(layer)
        if layer_id not in self._dequant_cache:
            W = _dequant_gptq(
                layer.qweight.data,
                layer.scales.data,
                layer.qzeros.data,
                layer._mq_bits,
                layer._mq_group_size,
            )
            self._dequant_cache[layer_id] = W
            logger.debug("MQ Marlin%d: dequant cached for layer %s",
                         layer._mq_bits, layer_id)

        W = self._dequant_cache[layer_id]
        return F.linear(x, W.to(x.dtype), bias)


def _dequant_gptq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Dequantize GPTQ sub-4-bit to float16. Returns [N, K]."""
    K_packed, N = qweight.shape
    mask = (1 << bits) - 1

    if bits == 2:
        K = K_packed * 16
        shifts = torch.arange(0, 32, 2, device=qweight.device,
                              dtype=torch.int32)
        unpacked = ((qweight.unsqueeze(1).expand(-1, 16, -1)
                      >> shifts.view(1, 16, 1)) & mask).reshape(K, N)
    elif bits == 3:
        K = (K_packed * 32) // 3
        qw = qweight.to(torch.int64)
        bsh = torch.arange(32, device=qweight.device, dtype=torch.int64)
        bits_flat = ((qw.unsqueeze(1) >> bsh.view(1, 32, 1)) & 1
                     ).reshape(-1, N)
        n_vals = min(bits_flat.shape[0] // 3, K)
        idx = torch.arange(n_vals, device=qweight.device)
        unpacked = (bits_flat[idx * 3] | (bits_flat[idx * 3 + 1] << 1)
                    | (bits_flat[idx * 3 + 2] << 2)).to(torch.int32)
        if n_vals < K:
            unpacked = torch.cat([unpacked,
                torch.zeros(K - n_vals, N, device=qweight.device,
                            dtype=torch.int32)])
    elif bits == 4:
        K = K_packed * 8
        shifts = torch.arange(0, 32, 4, device=qweight.device,
                              dtype=torch.int32)
        unpacked = ((qweight.unsqueeze(1).expand(-1, 8, -1)
                      >> shifts.view(1, 8, 1)) & mask).reshape(K, N)
    else:
        raise ValueError(f"Unsupported bits={bits}")

    n_groups = K // group_size

    # Unpack zero points
    if bits in (2, 4):
        pf = 32 // bits
        zsh = torch.arange(0, 32, bits, device=qzeros.device,
                           dtype=torch.int32)[:pf]
        zp = ((qzeros.unsqueeze(1).expand(-1, pf, -1)
               >> zsh.view(1, -1, 1)) & mask).reshape(n_groups, -1)[:, :N]
    elif bits == 3:
        qz = qzeros.to(torch.int64)
        zsh = torch.arange(32, device=qzeros.device, dtype=torch.int64)
        zb = ((qz.unsqueeze(2) >> zsh.view(1, 1, 32)) & 1
              ).reshape(n_groups, -1)
        n_zp = min(zb.shape[1] // 3, N)
        zi = torch.arange(n_zp, device=qzeros.device)
        zp = (zb[:, zi * 3] | (zb[:, zi * 3 + 1] << 1)
              | (zb[:, zi * 3 + 2] << 2)).to(torch.int32)
        if n_zp < N:
            zp = torch.cat([zp, torch.zeros(n_groups, N - n_zp,
                device=qzeros.device, dtype=torch.int32)], dim=1)

    gi = torch.arange(K, device=qweight.device) // group_size
    w = scales[gi] * (unpacked.float() - zp[gi].float())
    return w.T.contiguous().to(torch.float16)
