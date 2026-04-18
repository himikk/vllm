# SPDX-License-Identifier: Apache-2.0
"""MultiQuant sub-4-bit MoE method — per-expert fused GEMM.

For FusedMoE layers with INT2/INT3 weights: calls mq_gemm_int2/int3
per expert instead of the Triton MoE kernel (which only supports INT4).

Weight loading reuses MoeWNA16Method (GPTQ format, shard handling).
process_weights_after_loading() transforms from MoE transposed-uint8
format to kernel-native format.  apply() runs per-expert GEMM.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method

logger = init_logger(__name__)


class MQSub4MoEMethod(MoeWNA16Method):
    """MoE method for sub-4-bit (INT2/INT3) using MultiQuant fused GEMM.

    Inherits weight loading from MoeWNA16Method.
    process_weights_after_loading() transforms tensors from MoE storage
    format (transposed uint8) to kernel-native format:
      qweight: [E, K_packed_i32, N] int32
      scales:  [E, n_groups, N] float16
      qzeros:  [E, n_groups, N_zp_i32] int32
    apply() runs per-expert fused GEMM.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Transform MoE weights from transposed-uint8 to kernel format.

        MoeWNA16Method weight loader stores GPTQ tensors transposed as uint8:
          qweight: [E, N, K_packed_u8] uint8  (from [K/16, N] int32 GPTQ)
          scales:  [E, N, n_groups]           (from [n_groups, N] GPTQ)
          qzeros:  [E, N_zp_u8, n_groups] uint8 (converted+transposed)

        Our kernel expects original GPTQ layout:
          qweight: [K/16, N] int32
          scales:  [n_groups, N] float16
          qzeros:  [n_groups, N_zp] int32
        """
        bits = self.quant_config.weight_bits

        for prefix in ('w13', 'w2'):
            # --- qweight: [E, N, K_u8] uint8 → [E, K_i32, N] int32 ---
            qw = getattr(layer, f'{prefix}_qweight')
            # view(int32) reinterprets last dim: [E, N, K_u8] → [E, N, K_u8/4]
            qw_i32 = qw.data.contiguous().view(torch.int32)
            # transpose dims 1,2: [E, N, K_i32] → [E, K_i32, N]
            qw_kernel = qw_i32.transpose(1, 2).contiguous()
            setattr(layer, f'{prefix}_qweight', torch.nn.Parameter(
                qw_kernel, requires_grad=False))

            # --- scales: [E, N, n_groups] → [E, n_groups, N] ---
            sc = getattr(layer, f'{prefix}_scales')
            sc_kernel = sc.data.transpose(1, 2).contiguous()
            setattr(layer, f'{prefix}_scales', torch.nn.Parameter(
                sc_kernel, requires_grad=False))

            # --- qzeros: undo convert_gptq_int4_qzeros + transpose ---
            qz_param = getattr(layer, f'{prefix}_qzeros', None)
            if qz_param is None or qz_param.numel() == 0:
                continue
            qz = qz_param.data  # [E, N_zp_u8, n_groups] uint8
            # undo transpose: [E, N_zp_u8, n_groups] → [E, n_groups, N_zp_u8]
            qz = qz.transpose(1, 2).contiguous()
            # undo convert_gptq_int4_qzeros (which added +1 to each 4-bit nibble)
            lo = (qz & 0xF).to(torch.int16) - 1
            hi = ((qz >> 4) & 0xF).to(torch.int16) - 1
            qz_orig = ((lo & 0xF) | ((hi & 0xF) << 4)).to(torch.uint8)
            # [E, n_groups, N_zp_u8] uint8 → [E, n_groups, N_zp_i32] int32
            qz_i32 = qz_orig.contiguous().view(torch.int32)
            setattr(layer, f'{prefix}_qzeros', torch.nn.Parameter(
                qz_i32, requires_grad=False))

        logger.info("MQSub4MoE: transformed weights to kernel format "
                     "(bits=%d, w13_qweight=%s, w13_scales=%s, w13_qzeros=%s)",
                     bits, layer.w13_qweight.shape, layer.w13_scales.shape,
                     layer.w13_qzeros.shape if hasattr(layer, 'w13_qzeros')
                     else 'N/A')

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Per-expert GEMM with INT2/INT3 fused dequant.

        After process_weights_after_loading, tensors are in kernel format:
          w13_qweight: [E, K_packed_i32, N] int32
          w13_scales:  [E, n_groups, N] float16
          w13_qzeros:  [E, n_groups, N_zp_i32] int32
        """
        from vllm.multiquant.weight_quant.mq_sub4_linear import _load_mq_gemm

        bits = self.quant_config.weight_bits
        group_size = self.quant_config.group_size
        kernel = _load_mq_gemm(bits)
        if kernel is None:
            raise RuntimeError(f"mq_gemm_int{bits} not available")

        B, K = x.shape  # [num_tokens, hidden_size]
        num_experts = layer.w13_qweight.shape[0]
        topk = topk_ids.shape[1]

        # After process_weights_after_loading:
        # w13_qweight [E, K_packed_i32, N_gate_up] — directly usable by kernel
        # w13_scales  [E, n_groups, N_gate_up]
        # w13_qzeros  [E, n_groups, N_zp_i32]
        N_gate_up = layer.w13_scales.shape[2]  # actual N (gate+up fused)

        output = torch.zeros(B, K, dtype=x.dtype, device=x.device)

        for b in range(B):
            for t in range(topk):
                expert_id = topk_ids[b, t].item()
                weight = topk_weights[b, t].item()
                if expert_id < 0 or expert_id >= num_experts:
                    continue

                x_row = x[b:b+1].to(torch.float16)  # [1, K]

                # Gate+Up GEMM: x[1,K] @ dequant(W13[K_packed,N]) → [1, N]
                w13_q = layer.w13_qweight[expert_id]  # [K_packed_i32, N]
                w13_s = layer.w13_scales[expert_id].to(torch.float16)  # [n_groups, N]
                w13_z = layer.w13_qzeros[expert_id]   # [n_groups, N_zp_i32]

                gate_up = torch.zeros(1, w13_q.shape[1], dtype=torch.float16,
                                      device=x.device)
                if bits == 2:
                    kernel.mq_gemm_int2(x_row, w13_q, w13_s, w13_z,
                                        gate_up, group_size)
                elif bits == 3:
                    kernel.mq_gemm_int3(x_row, w13_q, w13_s, w13_z,
                                        gate_up, K, group_size)

                # SiLU activation: silu(gate) * up
                half_n = N_gate_up // 2
                gate = gate_up[0, :half_n]
                up = gate_up[0, half_n:]
                activated = torch.nn.functional.silu(gate) * up  # [half_n]

                # Down GEMM: activated[1, half_n] @ dequant(W2) → [1, K]
                act_row = activated.unsqueeze(0)  # [1, half_n]
                w2_q = layer.w2_qweight[expert_id]  # [half_n_packed_i32, K]
                w2_s = layer.w2_scales[expert_id].to(torch.float16)
                w2_z = layer.w2_qzeros[expert_id]

                down = torch.zeros(1, w2_q.shape[1], dtype=torch.float16,
                                   device=x.device)
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
