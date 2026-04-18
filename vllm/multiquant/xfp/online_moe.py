# SPDX-License-Identifier: Apache-2.0
"""XFP online MoE method — BF16 → per-expert learned codebook at load time.

v3: fused CUDA MoE kernel. Single kernel launch per GEMM handles all experts
via sorted_token_ids / expert_ids (Marlin pattern). No Python expert loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.multiquant.policy import DTYPE_BITS

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)


# ─── Custom op for torch.compile / CUDA Graph compatibility ─────────
#
# The MoE forward contains moe_align_block_size (C++ op), dynamic
# tensor allocations, and Python control flow — all incompatible with
# CUDA Graph capture. Wrapping as a custom op makes torch.compile see
# an opaque operator with known output shape.

def _xfp_moe_forward_impl(
    x: torch.Tensor,              # [B, K]
    topk_weights: torch.Tensor,   # [B, topk]
    topk_ids: torch.Tensor,       # [B, topk]
    w13_packed: torch.Tensor,     # [E * fpe13] int32
    w13_codebook: torch.Tensor,   # [E * N13 * lut] fp16
    w2_packed: torch.Tensor,      # [E * fpe2] int32
    w2_codebook: torch.Tensor,    # [E * N2 * lut] fp16
    bits: int,
    K13: int, N13: int,
    K2: int, N2: int,
    E: int, fpe13: int, fpe2: int,
) -> torch.Tensor:
    """Real impl: full MoE forward (gate_up → SiLU → down → reduce)."""
    from vllm.multiquant.xfp.xfp_moe_kernel import _load_xfp_moe_gemm
    moe_kernel = _load_xfp_moe_gemm()
    if moe_kernel is None:
        raise RuntimeError("XFP MoE kernel not loaded")

    B = x.shape[0]
    topk = topk_ids.shape[1]
    half_n = N13 // 2
    BT = B * topk

    x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
    no_w = torch.empty(0, dtype=torch.float32, device=x.device)

    # Token sorting — pure torch ops (CUDA Graph safe, no C++ custom op)
    flat_topk = topk_ids.reshape(-1)  # [B*topk]
    sort_indices = flat_topk.argsort(stable=True)
    sorted_token_ids = sort_indices.to(torch.int32)
    sorted_expert_ids = flat_topk[sort_indices].to(torch.int32)
    num_valid = sorted_token_ids.shape[0]

    # Gate+Up
    gate_up = torch.zeros(BT, N13, dtype=torch.bfloat16, device=x.device)
    moe_kernel.xfp_moe_gemm(
        x_bf16, w13_packed, w13_codebook,
        gate_up, sorted_token_ids, sorted_expert_ids,
        no_w, int(bits), int(K13), int(N13), int(topk),
        int(fpe13), num_valid)

    # SiLU
    gate = F.silu(gate_up[:, :half_n])
    up = gate_up[:, half_n:]
    activated = gate * up

    # Down
    down = torch.zeros(BT, N2, dtype=torch.bfloat16, device=x.device)
    down_expert_ids = topk_ids.reshape(-1).to(torch.int32)
    down_sorted = torch.arange(BT, dtype=torch.int32, device=x.device)
    moe_kernel.xfp_moe_gemm(
        activated, w2_packed, w2_codebook,
        down, down_sorted, down_expert_ids,
        no_w, int(bits), int(K2), int(N2), 1,
        int(fpe2), BT)

    # Scatter-reduce
    orig = torch.arange(BT, device=x.device, dtype=torch.int64) // topk
    weighted = down.float() * topk_weights.reshape(-1).unsqueeze(1)
    output = torch.zeros(B, N2, dtype=torch.bfloat16, device=x.device)
    output.scatter_add_(
        0, orig.unsqueeze(1).expand_as(weighted),
        weighted.to(output.dtype))
    return output


def _xfp_moe_forward_fake(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_codebook: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_codebook: torch.Tensor,
    bits: int,
    K13: int, N13: int,
    K2: int, N2: int,
    E: int, fpe13: int, fpe2: int,
) -> torch.Tensor:
    """Fake impl: output shape [B, N2] bf16."""
    return torch.empty(x.shape[0], N2, dtype=torch.bfloat16, device=x.device)


try:
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="xfp_moe_forward",
        op_func=_xfp_moe_forward_impl,
        fake_impl=_xfp_moe_forward_fake,
    )
    _xfp_moe_op = torch.ops.vllm.xfp_moe_forward
    logger.info("XFP MoE custom op registered (torch.compile safe)")
except Exception as e:
    logger.warning("XFP MoE custom op registration failed: %s", e)
    _xfp_moe_op = _xfp_moe_forward_impl


class XFPMoEMethod(FusedMoEMethodBase):
    """Learned-codebook quant-on-load for FusedMoE layers.

    Per-expert Lloyd codebook + word-aligned packed indices.
    Apply uses fused CUDA kernel — one launch per GEMM, all experts.
    """

    def __init__(
        self,
        quant_config: "QuantizationConfig",
        dtype: str = "xfp4",
        moe_config: "FusedMoEConfig | None" = None,
    ):
        if dtype not in ("xfp", "xfp2", "xfp3", "xfp4"):
            raise ValueError(
                f"XFPMoEMethod: unsupported dtype '{dtype}', "
                f"supported: xfp (auto), xfp2, xfp3, xfp4"
            )
        if moe_config is not None:
            super().__init__(moe_config)
        self.quant_config = quant_config
        self.dtype = dtype
        self.bits = DTYPE_BITS[dtype]

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
        # When the base_loader has flagged streaming quant-on-load, allocate
        # the huge expert tensors on meta. initialize_streaming_quantload()
        # will materialize them on CUDA right before the loader touches them.
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
        layer._xfp_moe_hidden = hidden_size
        layer._xfp_moe_intermediate = intermediate_size_per_partition

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "_xfp_moe_packed", False):
            return

        from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
        from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm
        from vllm.multiquant.xfp.xfp_moe_kernel import _load_xfp_moe_gemm
        from vllm.multiquant.weight_cache import MultiQuantWeightCache
        from vllm.multiquant.xfp import xfp_weight_cache as xfp_cache

        device = layer.w13_weight.device

        # ─── Cache check (skip 256-expert Lloyd if we have it on disk) ──
        cache = MultiQuantWeightCache.get_active()
        layer_prefix = getattr(layer, "layer_name", "") or \
            getattr(layer, "prefix", "") or ""
        if cache is not None and layer_prefix and xfp_cache.load_moe(
                cache, layer_prefix, layer, device):
            # Kernel JIT-load still needed so forward path is graph-safe.
            _load_xfp_gemm(int(layer._xfp_moe_bits))
            _load_xfp_moe_gemm()
            logger.info(
                "XFP MoE %s ← cache (skip Lloyd) bits=%d E=%d "
                "K13=%d N13=%d K2=%d N2=%d",
                layer_prefix, layer._xfp_moe_bits, layer._xfp_moe_E,
                layer._xfp_moe_K13, layer._xfp_moe_N13,
                layer._xfp_moe_K2, layer._xfp_moe_N2,
            )
            try:
                del layer.w13_weight, layer.w2_weight
            except AttributeError:
                pass
            return

        bits = self.bits
        w13 = layer.w13_weight.data  # [E, N_gate_up, K]
        w2 = layer.w2_weight.data    # [E, N_down, K_down]
        E = int(w13.shape[0])

        # MoE Lloyd iters: defined BEFORE auto-select so both use the same.
        import os
        moe_lloyd_iters = int(os.environ.get("XFP_MOE_LLOYD_ITERS", "5"))

        # Auto bit-width: sample a few experts, run auto_select with the
        # SAME lloyd_iters as the actual packing to avoid quality mismatch.
        if bits == 0:
            from vllm.multiquant.xfp.xfp_pack import xfp_auto_select
            sample_experts = min(4, E)
            sample = w13[:sample_experts].reshape(-1, w13.shape[2]).float()
            bits = xfp_auto_select(
                sample,
                candidates=(2, 3, 4),
                min_cos=self.quant_config.auto_min_cos
                    if hasattr(self.quant_config, 'auto_min_cos') else 0.98,
                lloyd_iters=moe_lloyd_iters,
            )
            logger.info("XFP MoE auto-select: bits=%d (from %d expert sample, "
                        "lloyd=%d)", bits, sample_experts, moe_lloyd_iters)

        _load_xfp_gemm(bits)
        _load_xfp_moe_gemm()

        # Save shape metadata before freeing weights
        K13 = int(w13.shape[2])
        N13 = int(w13.shape[1])
        K2 = int(w2.shape[2])
        N2 = int(w2.shape[1])

        from vllm.multiquant.policy import MultiQuantPolicyRegistry
        reg = MultiQuantPolicyRegistry.get_active()

        def _expertwise_pack_and_repack(W_stack: torch.Tensor):
            """W_stack: [E, N, K] -> flat packed [E*fpe], flat codebook [E*N*lut].

            Packs one expert at a time: float32 transient = N×K×4 bytes
            (~25 MiB) instead of E×N×K×4 (~9 GiB). Critical for 122B+
            models on unified memory.
            """
            E_ = int(W_stack.shape[0])
            N_ = int(W_stack.shape[1])
            K_ = int(W_stack.shape[2])
            lut_size = 1 << bits

            repacked_list = []
            codebook_list = []
            last_stats = None

            for e in range(E_):
                W_e = W_stack[e].float()          # [N, K] float32, ~25 MiB
                packed_e, cb_e, _, _, stats_e = xfp_pack(
                    W_e, bits=bits, outlier_sigma=None,
                    lloyd_iters=moe_lloyd_iters,
                )
                del W_e
                repacked_list.append(xfp_repack(packed_e))
                codebook_list.append(cb_e)        # [N, lut_size]
                del packed_e
                last_stats = stats_e

            del W_stack
            flat_per_expert = repacked_list[0].numel()
            all_repacked = torch.cat(repacked_list, dim=0)
            del repacked_list
            all_codebook = torch.cat(codebook_list, dim=0)
            del codebook_list

            return all_repacked, all_codebook, flat_per_expert, last_stats

        # Pack w13, then free BF16 w13 before packing w2
        p13, cb13, fpe13, stats13 = _expertwise_pack_and_repack(w13)
        layer.w13_weight.data = torch.empty(0)  # free BF16

        p2, cb2, fpe2, stats2 = _expertwise_pack_and_repack(w2)
        layer.w2_weight.data = torch.empty(0)  # free BF16

        if reg is not None:
            reg.record_stats("routed_expert", stats13)
            reg.record_stats("routed_expert", stats2)

        # Store as flat tensors for fused kernel
        layer.w13_xfp_packed = nn.Parameter(p13.to(device), requires_grad=False)
        layer.w13_xfp_codebook = nn.Parameter(cb13.to(device), requires_grad=False)
        layer.w2_xfp_packed = nn.Parameter(p2.to(device), requires_grad=False)
        layer.w2_xfp_codebook = nn.Parameter(cb2.to(device), requires_grad=False)

        layer._xfp_moe_bits = bits
        layer._xfp_moe_dtype = self.dtype
        layer._xfp_moe_K13 = K13
        layer._xfp_moe_N13 = N13
        layer._xfp_moe_K2 = K2
        layer._xfp_moe_N2 = N2
        layer._xfp_moe_E = E
        layer._xfp_moe_fpe13 = fpe13
        layer._xfp_moe_fpe2 = fpe2
        layer._xfp_moe_packed = True

        try:
            del layer.w13_weight, layer.w2_weight
        except AttributeError:
            pass

        # Persist to disk cache (if enabled) so future loads skip Lloyd.
        if cache is not None and layer_prefix:
            xfp_cache.save_moe(cache, layer_prefix, layer)

        logger.info(
            "XFP MoE: %d experts w13[%dx%d] + w2[%dx%d] -> %s "
            "(fused, fpe=%d/%d, lloyd=%d)",
            E, layer._xfp_moe_N13, layer._xfp_moe_K13,
            layer._xfp_moe_N2, layer._xfp_moe_K2, self.dtype,
            fpe13, fpe2, moe_lloyd_iters,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not getattr(layer, "_xfp_moe_packed", False):
            from vllm.model_executor.layers.fused_moe import fused_experts
            return fused_experts(
                x, layer.w13_weight, layer.w2_weight,
                topk_weights=topk_weights, topk_ids=topk_ids,
            )

        return _xfp_moe_op(
            x, topk_weights, topk_ids,
            layer.w13_xfp_packed, layer.w13_xfp_codebook,
            layer.w2_xfp_packed, layer.w2_xfp_codebook,
            int(layer._xfp_moe_bits),
            int(layer._xfp_moe_K13), int(layer._xfp_moe_N13),
            int(layer._xfp_moe_K2), int(layer._xfp_moe_N2),
            int(layer._xfp_moe_E),
            int(layer._xfp_moe_fpe13), int(layer._xfp_moe_fpe2),
        )
