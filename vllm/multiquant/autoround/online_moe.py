# SPDX-License-Identifier: Apache-2.0
"""AutoRound RTN online MoE method — BF16 → GPTQ format at load time.

Loads BF16 MoE weights normally, packs per-expert to GPTQ int32 in
process_weights_after_loading, then delegates inference to MQSub4MoEMethod.

Same storage format as pre-quantized MoE models → same per-expert GEMM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.multiquant.autoround.config import AutoRoundRTNConfig

logger = init_logger(__name__)


class AutoRoundRTNMoEMethod(FusedMoEMethodBase):
    """BF16 MoE → GPTQ-format INT2/INT3/INT4 at load time.

    Inherits weight allocation from UnquantizedFusedMoEMethod.
    process_weights_after_loading packs each expert via rtn_pack_gptq,
    then transforms to kernel format via MQSub4MoEMethod logic.
    apply() delegates to per-expert fused GEMM.
    """

    def __init__(self, quant_config: "AutoRoundRTNConfig",
                 bits: int = 4, group_size: int = 128,
                 moe_config: "FusedMoEConfig | None" = None):
        if moe_config is not None:
            super().__init__(moe_config)
        self.quant_config = quant_config
        self.bits = bits
        self.group_size = group_size

    def get_fused_moe_quant_config(self, layer):
        """No special quant config — we handle packing ourselves."""
        return None

    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Allocate BF16 weights — standard FusedMoE allocation."""
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
        # Store dimensions for later packing
        layer._rtn_moe_hidden = hidden_size
        layer._rtn_moe_intermediate = intermediate_size_per_partition

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """BF16 MoE weights → GPTQ packed format, per expert."""
        if getattr(layer, "_rtn_moe_packed", False):
            return

        from vllm.multiquant.autoround.rtn_pack import rtn_pack_gptq

        bits = self.bits
        group_size = self.group_size
        device = layer.w13_weight.device

        # w13_weight: [E, 2*intermediate, hidden] BF16 (gate+up fused)
        # w2_weight:  [E, hidden, intermediate] BF16 (down)
        w13 = layer.w13_weight.data  # [E, N_gate_up, K]
        w2 = layer.w2_weight.data    # [E, N_down, K_down]
        E = w13.shape[0]

        # Pack each expert's gate+up and down projections
        w13_qw_list, w13_sc_list, w13_qz_list = [], [], []
        w2_qw_list, w2_sc_list, w2_qz_list = [], [], []

        for e in range(E):
            # Gate+Up: [N_gate_up, K] → GPTQ [K_packed, N_gate_up]
            qw, sc, qz = rtn_pack_gptq(w13[e].float(), bits, group_size)
            w13_qw_list.append(qw)
            w13_sc_list.append(sc)
            w13_qz_list.append(qz)

            # Down: [N_down, K_down] → GPTQ [K_down_packed, N_down]
            qw2, sc2, qz2 = rtn_pack_gptq(w2[e].float(), bits, group_size)
            w2_qw_list.append(qw2)
            w2_sc_list.append(sc2)
            w2_qz_list.append(qz2)

        # Stack into [E, ...] tensors (kernel format after MQSub4MoE transform)
        layer.w13_qweight = nn.Parameter(
            torch.stack(w13_qw_list).to(device), requires_grad=False)
        layer.w13_scales = nn.Parameter(
            torch.stack(w13_sc_list).to(device), requires_grad=False)
        layer.w13_qzeros = nn.Parameter(
            torch.stack(w13_qz_list).to(device), requires_grad=False)

        layer.w2_qweight = nn.Parameter(
            torch.stack(w2_qw_list).to(device), requires_grad=False)
        layer.w2_scales = nn.Parameter(
            torch.stack(w2_sc_list).to(device), requires_grad=False)
        layer.w2_qzeros = nn.Parameter(
            torch.stack(w2_qz_list).to(device), requires_grad=False)

        # Free BF16 weights
        del layer.w13_weight, layer.w2_weight

        layer._rtn_moe_packed = True
        layer._rtn_moe_bits = bits
        layer._rtn_moe_dtype = f"int{bits}"

        total_params = (w13.numel() + w2.numel())
        packed_bytes = sum(
            t.numel() * 4 for t in [
                layer.w13_qweight, layer.w2_qweight,
                layer.w13_qzeros, layer.w2_qzeros,
            ]
        ) + sum(
            t.numel() * 2 for t in [layer.w13_scales, layer.w2_scales]
        )
        logger.info(
            "RTN MoE: %d experts (%dx%d + %dx%d) → INT%d GPTQ, "
            "%.1f%% of BF16",
            E, w13.shape[1], w13.shape[2], w2.shape[1], w2.shape[2],
            bits, 100.0 * packed_bytes / (total_params * 2),
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Delegate to MQSub4MoEMethod's per-expert GEMM."""
        if not getattr(layer, "_rtn_moe_packed", False):
            # Fallback: BF16 (shouldn't happen after process_weights)
            from vllm.model_executor.layers.fused_moe import fused_experts
            return fused_experts(
                x, layer.w13_weight, layer.w2_weight,
                topk_weights=topk_weights, topk_ids=topk_ids)

        from vllm.multiquant.weight_quant.mq_sub4_linear import _load_mq_gemm

        bits = layer._rtn_moe_bits
        group_size = self.group_size
        kernel = _load_mq_gemm(bits)
        if kernel is None:
            raise RuntimeError(f"mq_gemm_int{bits} not available")

        B, K = x.shape
        num_experts = layer.w13_qweight.shape[0]
        topk = topk_ids.shape[1]
        N_gate_up = layer.w13_scales.shape[2]

        output = torch.zeros(B, K, dtype=x.dtype, device=x.device)

        for b in range(B):
            for t in range(topk):
                expert_id = topk_ids[b, t].item()
                weight = topk_weights[b, t].item()
                if expert_id < 0 or expert_id >= num_experts:
                    continue

                x_row = x[b:b+1].to(torch.float16)

                # Gate+Up GEMM
                w13_q = layer.w13_qweight[expert_id]
                w13_s = layer.w13_scales[expert_id].to(torch.float16)
                w13_z = layer.w13_qzeros[expert_id]

                gate_up = torch.zeros(1, w13_q.shape[1],
                                      dtype=torch.float16, device=x.device)
                if bits == 2:
                    kernel.mq_gemm_int2(x_row, w13_q, w13_s, w13_z,
                                        gate_up, group_size)
                elif bits == 3:
                    kernel.mq_gemm_int3(x_row, w13_q, w13_s, w13_z,
                                        gate_up, K, group_size)

                # SiLU activation
                half_n = N_gate_up // 2
                gate = gate_up[0, :half_n]
                up = gate_up[0, half_n:]
                activated = torch.nn.functional.silu(gate) * up

                # Down GEMM
                act_row = activated.unsqueeze(0)
                w2_q = layer.w2_qweight[expert_id]
                w2_s = layer.w2_scales[expert_id].to(torch.float16)
                w2_z = layer.w2_qzeros[expert_id]

                down = torch.zeros(1, w2_q.shape[1],
                                   dtype=torch.float16, device=x.device)
                if bits == 2:
                    kernel.mq_gemm_int2(act_row, w2_q, w2_s, w2_z,
                                        down, group_size)
                elif bits == 3:
                    K_down = act_row.shape[1]
                    kernel.mq_gemm_int3(act_row, w2_q, w2_s, w2_z,
                                        down, K_down, group_size)

                output[b] += weight * down[0].to(output.dtype)

        if shared_experts_input is not None:
            return output, shared_experts_input
        return output
