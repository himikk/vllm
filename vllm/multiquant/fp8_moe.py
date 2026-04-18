# SPDX-License-Identifier: Apache-2.0
"""FP8 MoE method for MultiQuant — quant-on-load bf16 → fp8 for FusedMoE.

Stores MoE expert weights in FP8 E4M3 (half memory). Dequants to bf16
on first forward call (cached). Uses the standard Triton MoE kernel.

This is a memory optimization, not a compute optimization — the weights
are stored in FP8 to save memory during loading of large BF16 models
(e.g. Qwen3.5-122B at 250 GB → ~125 GB).

Usage: --weight-dtype fp8  (or --weight-dtype-routed fp8)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

logger = init_logger(__name__)


class FP8MoEMethod(FusedMoEMethodBase):
    """BF16 → FP8 quant-on-load for FusedMoE layers.

    Delegates create_weights and apply to UnquantizedFusedMoEMethod.
    Only overrides process_weights_after_loading to cast bf16→fp8+scale,
    then dequants back to bf16 (cached) for the standard Triton kernel.
    """

    def __init__(self, quant_config=None, moe_config: "FusedMoEConfig | None" = None):
        if moe_config is not None:
            super().__init__(moe_config)
        self.quant_config = quant_config
        self._unquant = None

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        self._unquant = UnquantizedFusedMoEMethod(self.moe)
        from vllm.model_executor.model_loader.utils import _moe_meta_active
        if _moe_meta_active():
            with torch.device("meta"):
                self._unquant.create_weights(
                    layer, num_experts, hidden_size,
                    intermediate_size_per_partition, params_dtype,
                    **extra_weight_attrs,
                )
        else:
            self._unquant.create_weights(
                layer, num_experts, hidden_size,
                intermediate_size_per_partition, params_dtype,
                **extra_weight_attrs,
            )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Cast w13/w2 bf16 → fp8_e4m3fn + per-tensor scale, then dequant cached."""
        for wname in ["w13_weight", "w2_weight"]:
            w = getattr(layer, wname, None)
            if w is None:
                continue
            data = w.data  # [E, N, K]
            device = data.device

            max_fp8 = torch.finfo(torch.float8_e4m3fn).max
            amax = data.float().abs().max().clamp(min=1e-12)
            scale = (amax / max_fp8).to(torch.float32)

            # Cast to FP8
            w_fp8 = (data.float() / scale).clamp(-max_fp8, max_fp8).to(
                torch.float8_e4m3fn)

            # Immediately dequant back to bf16 (cached) — saves memory during
            # loading (fp8 is half of bf16) but the Triton kernel needs bf16.
            w_bf16 = (w_fp8.to(torch.float32) * scale).to(torch.bfloat16)

            setattr(layer, wname, nn.Parameter(
                w_bf16.to(device), requires_grad=False))

            E = data.shape[0]
            N, K = data.shape[1], data.shape[2]
            saved_mb = E * N * K / 1e6  # temp saving during load
            logger.info(
                "FP8 MoE: %s [%dx%dx%d] bf16->fp8->bf16 (scale=%.4g, "
                "peak saved %.0f MB during load)",
                wname, E, N, K, scale.item(), saved_mb,
            )

    def apply(
        self,
        layer: "FusedMoE",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ):
        """Delegate to standard unquantized MoE (weights are bf16 after dequant)."""
        if self._unquant is None:
            from vllm.model_executor.layers.fused_moe import fused_experts
            return fused_experts(
                x, layer.w13_weight, layer.w2_weight,
                topk_weights=topk_weights, topk_ids=topk_ids,
            )
        return self._unquant.apply(
            layer, x, topk_weights, topk_ids, shared_experts_input)
