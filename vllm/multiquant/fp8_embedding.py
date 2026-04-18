# SPDX-License-Identifier: Apache-2.0
"""FP8 embedding/LM-head method for MultiQuant.

Stores embedding weights in float8_e4m3fn (half the memory of bf16).
Dequantizes on-the-fly during apply (LM Head GEMM) and embedding lookup.

Usage: --weight-dtype-lm-head fp8
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

logger = init_logger(__name__)


class FP8EmbeddingMethod(QuantizeMethodBase):
    """Store embedding/LM-head weights in FP8 E4M3.

    - create_weights: allocate bf16 weight (same as unquantized)
    - process_weights_after_loading: cast to float8_e4m3fn + compute scale
    - apply: LM Head path — dequant to bf16 then GEMM
    - embedding: lookup path — dequant selected rows to bf16
    """

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
        from vllm.model_executor.utils import set_weight_attrs
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Cast weight bf16 → float8_e4m3fn + store per-tensor scale."""
        w = layer.weight.data
        device = w.device

        # Compute scale: max(abs(w)) / max_fp8
        max_fp8 = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        amax = w.float().abs().max().clamp(min=1e-12)
        scale = (amax / max_fp8).to(torch.float32)

        # Cast
        w_fp8 = (w.float() / scale).clamp(-max_fp8, max_fp8).to(
            torch.float8_e4m3fn)

        layer.weight = nn.Parameter(w_fp8.to(device), requires_grad=False)
        layer.weight_scale = nn.Parameter(
            scale.to(device), requires_grad=False)

        N, K = w.shape
        saved_mb = N * K * (2 - 1) / 1e6  # bf16(2B) - fp8(1B)
        logger.info(
            "FP8 LM Head: [%dx%d] bf16 -> fp8_e4m3 (scale=%.4g, saved %.0f MB)",
            N, K, scale.item(), saved_mb,
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """LM Head GEMM via torch._scaled_mm (FP8 native, no dequant).

        Caches transposed weight on first call. Keeps weight in FP8 —
        real memory saving (no bf16 copy).
        """
        if not hasattr(layer, '_fp8_weight_t'):
            # Cache transposed FP8 weight + scale tensors
            layer._fp8_weight_t = layer.weight.t().contiguous()
            layer._fp8_x_scale = torch.tensor(
                1.0, dtype=torch.float32, device=layer.weight.device)

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

    def embedding(
        self,
        layer: nn.Module,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        """Embedding lookup: dequant selected rows."""
        # Lookup in FP8, then dequant
        out_fp8 = F.embedding(input_, layer.weight)
        return out_fp8.to(torch.bfloat16) * layer.weight_scale
