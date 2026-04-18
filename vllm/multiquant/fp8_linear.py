# SPDX-License-Identifier: Apache-2.0
"""FP8 linear method for MultiQuant — quant-on-load bf16 → fp8_e4m3fn.

Halves weight memory. Uses torch._scaled_mm for FP8 Tensor Core GEMM.
No calibration data needed — simple per-tensor absmax scaling.

Usage: --weight-dtype fp8  (sets all weight classes)
       --weight-dtype-attn fp8  (per-class)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

logger = init_logger(__name__)


class FP8LinearMethod(QuantizeMethodBase):
    """BF16 → FP8 E4M3 quant-on-load for Linear layers.

    - create_weights: allocate bf16 (same as unquantized)
    - process_weights_after_loading: cast to float8_e4m3fn + scale
    - apply: torch._scaled_mm (FP8 × FP8 → BF16, Tensor Core native)
    """

    def __init__(self, quant_config=None):
        self.quant_config = quant_config

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

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Cast bf16 → float8_e4m3fn + per-tensor scale."""
        w = layer.weight.data
        device = w.device
        N, K = w.shape

        max_fp8 = torch.finfo(torch.float8_e4m3fn).max
        amax = w.float().abs().max().clamp(min=1e-12)
        scale = (amax / max_fp8).to(torch.float32)

        w_fp8 = (w.float() / scale).clamp(-max_fp8, max_fp8).to(
            torch.float8_e4m3fn)

        layer.weight = nn.Parameter(w_fp8.to(device), requires_grad=False)
        layer.weight_scale = nn.Parameter(
            scale.to(device), requires_grad=False)

        # Pre-cache transposed weight for _scaled_mm
        layer._fp8_weight_t = w_fp8.t().contiguous().to(device)
        layer._fp8_x_scale = torch.tensor(
            1.0, dtype=torch.float32, device=device)

        saved_mb = N * K / 1e6  # bf16(2B) → fp8(1B) = 1B saved per element
        logger.info(
            "FP8 Linear: [%dx%d] bf16 -> fp8_e4m3 (scale=%.4g, saved %.0f MB)",
            N, K, scale.item(), saved_mb,
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FP8 × FP8 → BF16 via torch._scaled_mm."""
        x_fp8 = x.to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_fp8,
            layer._fp8_weight_t,
            scale_a=layer._fp8_x_scale,
            scale_b=layer.weight_scale,
            out_dtype=torch.bfloat16,
        )
        if bias is not None:
            out = out + bias
        return out
