# SPDX-License-Identifier: Apache-2.0
"""AutoRound RTN online linear method — BF16 → GPTQ format at load time.

Loads BF16 normally, packs to GPTQ int32 in process_weights_after_loading,
then delegates inference to MQSub4LinearMethod (INT2/INT3) or Marlin (INT4).

Same storage format as pre-quantized AutoRound/GPTQ models → same kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

if TYPE_CHECKING:
    from vllm.multiquant.autoround.config import AutoRoundRTNConfig

logger = init_logger(__name__)


def _noop_weight_loader(param, loaded_weight, weight_name="",
                        shard_id=None, expert_id=None):
    """Dummy weight_loader for parameters created after loading."""
    pass


class AutoRoundRTNLinearMethod(QuantizeMethodBase):
    """BF16 → GPTQ-format INT2/INT3/INT4 at load time.

    After packing, delegates apply() to MQSub4LinearMethod (INT2/3)
    or Marlin kernel (INT4).
    """

    uses_meta_device: bool = False

    def __init__(self, quant_config: "AutoRoundRTNConfig",
                 dtype: str = "int4", group_size: int = 128):
        from vllm.multiquant.policy import DTYPE_BITS
        self.quant_config = quant_config
        self.dtype = dtype
        self.bits = DTYPE_BITS.get(dtype, 4)
        self.group_size = group_size

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from vllm.model_executor.parameter import ModelWeightParameter
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)
        # Store dimensions and dtype for Marlin setup
        layer._rtn_input_size = input_size
        layer._rtn_output_size = output_size
        layer._rtn_params_dtype = params_dtype

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """BF16 → GPTQ int32 packed format.

        For INT4: PackedvLLMParameter → MarlinLinearKernel (fused GEMM).
        For INT2/INT3: raw GPTQ format for mq_gemm kernels.
        """
        if getattr(layer, "_rtn_packed", False):
            return

        from vllm.multiquant.autoround.rtn_pack import rtn_pack_gptq

        W = layer.weight.data  # [N, K] (out_features, in_features)
        N, K = W.shape
        device = W.device

        qweight, scales, qzeros = rtn_pack_gptq(
            W.float(), self.bits, self.group_size)

        if self.dtype == "int4":
            self._setup_marlin(layer, qweight, scales, N, K, device)
        else:
            # INT2/INT3: store raw GPTQ tensors for mq_gemm kernels
            layer.qweight = nn.Parameter(
                qweight.to(device), requires_grad=False)
            layer.scales = nn.Parameter(
                scales.to(device), requires_grad=False)
            layer.qzeros = nn.Parameter(
                qzeros.to(device), requires_grad=False)

        # Free BF16 weight
        del layer.weight

        layer._rtn_packed = True
        layer._rtn_dtype = self.dtype
        layer._rtn_K = K
        layer._rtn_N = N

        logger.info(
            "RTN: %s (%dx%d) → %s %s, %.1f%% of BF16",
            getattr(layer, "layer_name", "?"), N, K, self.dtype,
            "Marlin" if self.dtype == "int4" else "mq_gemm",
            100.0 * qweight.numel() * 4 / (N * K * 2),
        )

    def _setup_marlin(self, layer: nn.Module,
                      qweight: torch.Tensor, scales: torch.Tensor,
                      N: int, K: int, device: torch.device) -> None:
        """Register PackedvLLMParameter + MarlinLinearKernel for INT4."""
        from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
            MarlinLinearKernel,
        )
        from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
            MPLinearLayerConfig,
        )
        from vllm.model_executor.parameter import (
            GroupQuantScaleParameter,
            PackedvLLMParameter,
        )
        from vllm.scalar_type import scalar_types

        pack_factor = 8  # INT4: 8 values per int32
        input_size = getattr(layer, "_rtn_input_size", K)
        output_size = getattr(layer, "_rtn_output_size", N)
        params_dtype = getattr(layer, "_rtn_params_dtype", torch.bfloat16)

        # Register qweight as PackedvLLMParameter (Marlin needs metadata)
        qw_param = PackedvLLMParameter(
            data=qweight.to(device),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=pack_factor,
            weight_loader=_noop_weight_loader,
        )
        layer.register_parameter("qweight", qw_param)

        # Register scales as GroupQuantScaleParameter
        sc_param = GroupQuantScaleParameter(
            data=scales.to(device),
            output_dim=1,
            input_dim=0,
            weight_loader=_noop_weight_loader,
        )
        layer.register_parameter("scales", sc_param)

        # g_idx and qzeros: empty (symmetric, no desc_act)
        # MarlinLinearKernel.process_weights_after_loading will set these

        # Build config for MarlinLinearKernel
        mp_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(K, N),
            weight_type=scalar_types.uint4b8,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=False,
            has_g_idx=False,
        )

        kernel = MarlinLinearKernel(
            mp_config,
            w_q_param_name="qweight",
            w_s_param_name="scales",
        )
        # Marlin repack (permute_param_layout_ + gptq_marlin_repack)
        kernel.process_weights_after_loading(layer)
        layer._marlin_kernel = kernel

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not getattr(layer, "_rtn_packed", False):
            return torch.nn.functional.linear(x, layer.weight, bias)

        dtype = layer._rtn_dtype
        if dtype == "int4":
            return layer._marlin_kernel.apply_weights(layer, x, bias)

        if dtype not in ("int2", "int3"):
            raise RuntimeError(f"RTN: unsupported dtype {dtype!r}")

        # INT2/INT3: fused mq_gemm kernels
        from vllm.multiquant.weight_quant.mq_sub4_linear import (
            _load_mq_gemm,
        )
        bits = self.bits
        kernel = _load_mq_gemm(bits)
        if kernel is None:
            raise RuntimeError(
                f"mq_gemm_{dtype} kernel not available")

        out_shape = x.shape[:-1] + (layer.qweight.shape[-1],)
        reshaped_x = x.reshape(-1, x.shape[-1]).to(torch.float16)
        M = reshaped_x.shape[0]
        N = layer.qweight.shape[1]

        C = torch.zeros(M, N, dtype=torch.float16, device=x.device)
        if dtype == "int2":
            kernel.mq_gemm_int2(
                reshaped_x, layer.qweight, layer.scales,
                layer.qzeros, C, self.group_size)
        else:
            K = reshaped_x.shape[1]
            kernel.mq_gemm_int3(
                reshaped_x, layer.qweight, layer.scales,
                layer.qzeros, C, K, self.group_size)

        if bias is not None:
            C.add_(bias)
        return C.reshape(out_shape)
