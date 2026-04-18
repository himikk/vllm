# SPDX-License-Identifier: Apache-2.0
"""MultiQuant attention backend — compressed uint8 KV-cache.

Generic backend for any KV-cache quantizer registered in the MultiQuant
registry (TurboQuant, RotorQuant, etc.). Does NOT inherit FlashInfer.

Cache: (num_blocks, 2, block_size, num_kv_heads, packed_size) uint8
Decode: compressed score + online softmax + V decompress
Prefill: naive causal attention on raw K/V
"""

import math
import os
import struct
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)

try:
    from vllm.attention import AttentionType
except ImportError:
    from vllm.v1.attention.backend import AttentionType

logger = init_logger(__name__)


# ============================================================
# Metadata
# ============================================================

@dataclass
class TQMetadata(AttentionMetadata):
    seq_lens: torch.Tensor          # [batch]
    block_table: torch.Tensor       # [batch, max_blocks]
    slot_mapping: torch.Tensor      # [num_tokens]
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    max_seq_len: int = 0
    query_start_loc: Optional[torch.Tensor] = None


# ============================================================
# Backend
# ============================================================

class MultiQuantAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "tq3", "tq4", "tq2w", "tq3w", "tq4w", "tq3r", "tq4r", "rq2", "rq3", "rq4",
    ]

    @staticmethod
    def get_name() -> str:
        return "MULTIQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16, 32, 64]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype=None) -> bool:
        if not kv_cache_dtype:
            return False
        from vllm.multiquant.registry import is_multiquant_dtype
        return is_multiquant_dtype(kv_cache_dtype)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return True  # packed_size is passed as head_size

    @staticmethod
    def get_impl_cls():
        return MultiQuantImpl

    @staticmethod
    def get_builder_cls():
        return TQMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="tq3"):
        # head_size is already packed_size from get_kv_cache_spec
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


# ============================================================
# Metadata Builder
# ============================================================

class TQMetadataBuilder(AttentionMetadataBuilder[TQMetadata]):
    from vllm.v1.attention.backend import AttentionCGSupport
    # TQ: CUDA kernel is graph-safe → full decode graphs
    # RQ: Clifford rotation has Python control flow → PIECEWISE only
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        from vllm.v1.attention.backend import AttentionCGSupport
        kv_dtype = str(getattr(vllm_config.cache_config, 'cache_dtype', ''))
        if kv_dtype.startswith("rq"):
            # RQ with fused Clifford kernel → graph-safe
            from vllm.v1.attention.ops.triton_mq_fused_decode import (
                _load_clifford_kernel,
            )
            if _load_clifford_kernel() is not None:
                return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
            return AttentionCGSupport.NEVER  # Python fallback → PIECEWISE
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE  # TQ

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        # Determine prefill vs decode
        num_tokens = cam.num_actual_tokens
        num_reqs = cam.num_reqs
        # Heuristic: if max_query_len > 1, there are prefill tokens
        num_prefill = 0
        num_decode = num_tokens
        if cam.max_query_len > 1:
            num_prefill = num_tokens
            num_decode = 0

        import os
        if os.environ.get("MQ_DEBUG"):
            if not hasattr(TQMetadataBuilder, '_build_cnt'):
                TQMetadataBuilder._build_cnt = 0
            TQMetadataBuilder._build_cnt += 1
            if TQMetadataBuilder._build_cnt <= 5:
                logger.info("[MQ_BUILD] #%d: tokens=%d reqs=%d "
                            "max_q_len=%d → pf=%d dc=%d",
                            TQMetadataBuilder._build_cnt,
                            num_tokens, num_reqs,
                            cam.max_query_len,
                            num_prefill, num_decode)

        return TQMetadata(
            seq_lens=cam.seq_lens,
            block_table=cam.block_table_tensor,
            slot_mapping=cam.slot_mapping,
            num_prefill_tokens=num_prefill,
            num_decode_tokens=num_decode,
            max_seq_len=cam.max_seq_len,
            query_start_loc=cam.query_start_loc,
        )


# ============================================================
# Impl
# ============================================================

class MultiQuantImpl:
    """Custom TQ attention — no FlashInfer dependency."""

    supports_quant_query_input: bool = False
    can_return_lse_for_decode: bool = False

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass

    @staticmethod
    def _recover_head_dim(head_size: int, kv_cache_dtype: str) -> int:
        """Recover real head_dim from head_size parameter.

        vLLM passes EITHER the real head_dim (256) OR the packed_size (100)
        depending on the code path. Check if head_size is already a valid D
        (i.e., its packed_size != head_size), else reverse-map from packed.
        """
        from vllm.multiquant.registry import get_kv_quantizer_config
        # Check if head_size IS already the real D
        try:
            cfg = get_kv_quantizer_config(kv_cache_dtype, head_size)
            if cfg.key_packed_size != head_size:
                # head_size is real D (packed would be different)
                return head_size
        except Exception:
            pass
        # head_size is packed_size — reverse map to real D
        for d in [64, 96, 128, 192, 256, 512]:
            try:
                cfg = get_kv_quantizer_config(kv_cache_dtype, d)
                if cfg.key_packed_size == head_size:
                    return d
            except Exception:
                continue
        # Fallback
        return head_size

    def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                 alibi_slopes=None, sliding_window=None, kv_cache_dtype="tq3",
                 logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **kwargs):
        self.num_heads = num_heads
        # head_size from Spec is packed_size (uint8 bytes), not real head_dim.
        # Recover real D: packed = ceil(D*mse_bits/8) + ceil(D/8) + 4
        from vllm.multiquant.registry import get_kv_quantizer_config
        real_head_dim = self._recover_head_dim(head_size, kv_cache_dtype)
        self.head_size = real_head_dim
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self._tq_config = get_kv_quantizer_config(kv_cache_dtype, real_head_dim)
        self._packed_size = self._tq_config.key_packed_size
        self._mse_bits = self._tq_config.mse_bits
        self._is_rq = kv_cache_dtype.startswith("rq")
        # WHT-family: block-based format (tq3w/tq4w = Hadamard, tq3r/tq4r = random rotation)
        self._is_wht = kv_cache_dtype.endswith("w") or kv_cache_dtype.endswith("r")
        self._is_block_rot = kv_cache_dtype.endswith("r")

        if self._is_wht:
            # WHT mode: no Pi/S matrices, no QJL correction
            from vllm.multiquant.turboquant.wht_config import TurboQuantWHTConfig
            assert isinstance(self._tq_config, TurboQuantWHTConfig)
            self._wht_block_size = self._tq_config.block_size
            self._mse_bytes = 0  # not used in WHT mode
            self._qjl_bytes = 0
            self._mask = 0
            self._correction = 0.0
        else:
            self._mse_bytes = (real_head_dim * self._mse_bits + 7) // 8
            self._qjl_bytes = (real_head_dim + 7) // 8
            self._mask = (1 << self._mse_bits) - 1
            self._correction = math.sqrt(math.pi / 2) / real_head_dim

        # Pre-allocate shift constants for _torch_pack (graph-safe bitpacking)
        if not self._is_wht:
            if self._mse_bits == 1:
                self._shifts_mse = torch.arange(8, device="cuda")
            elif self._mse_bits == 2:
                self._shifts_mse = torch.tensor([0, 2, 4, 6], device="cuda")
            else:
                self._shifts_mse = None
            self._shifts_sign = torch.arange(8, device="cuda")

        # Pre-load decode kernel at init (not in forward — graph-safe)
        if self._is_wht:
            # WHT path: CUDA kernels loaded lazily in _forward_decode
            # (tq_wht_splitkv_decode, tq_wht_fused_decode_attention or
            #  tq_blockrot_fused_decode_attention). PyTorch fallback only
            # triggers if the CUDA kernel fails to compile.
            decode_mode = "block-rot-CUDA" if self._is_block_rot else "WHT-CUDA"
        else:
            from vllm.v1.attention.ops.triton_mq_fused_decode import (
                mq_fused_decode_attention, _load_cuda_kernel,
            )
            cuda_kernel = _load_cuda_kernel()
            self._decode_fn = mq_fused_decode_attention
            decode_mode = "CUDA" if cuda_kernel else "Triton"

            # Pre-load RQ kernels at init (JIT compile before first forward)
            if self._is_rq:
                import vllm.multiquant.rotorquant.clifford  # noqa: F401
                from vllm.v1.attention.ops.triton_mq_fused_decode import (
                    _load_rq_decode_kernel, _load_clifford_kernel,
                )
                _load_rq_decode_kernel()
                _load_clifford_kernel()

        logger.info(
            "MultiQuant attention: D=%d (from spec %d), %s KV, "
            "decode=%s, %s",
            self.head_size, head_size,
            self.kv_cache_dtype,
            decode_mode,
            "RQ Clifford" if self._is_rq else "TQ rotation",
        )

    def _rotate_forward(self, x, Pi):
        """Forward rotation: TQ = x @ Pi.T, RQ = rotor sandwich."""
        if self._is_rq:
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors, rotor_sandwich,
                extract_vectors_from_multivectors,
            )
            D = x.shape[-1]
            mv = embed_vectors_as_multivectors(x)
            mv_rot = rotor_sandwich(Pi, mv)
            return extract_vectors_from_multivectors(mv_rot, D)
        return x @ Pi.T

    def _rotate_inverse(self, x, Pi):
        """Inverse rotation: TQ = x @ Pi, RQ = reverse rotor sandwich."""
        if self._is_rq:
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors, rotor_sandwich,
                extract_vectors_from_multivectors, reverse,
            )
            D = x.shape[-1]
            mv = embed_vectors_as_multivectors(x)
            rotor_rev = reverse(Pi)
            mv_recon = rotor_sandwich(rotor_rev, mv)
            return extract_vectors_from_multivectors(mv_recon, D)
        return x @ Pi

    def _torch_pack(self, vecs, Pi, S, centroids, D, device):
        """Graph-safe TQ/RQ packing — pure PyTorch tensor ops."""
        x = vecs.float()
        vn = x.norm(dim=-1)
        xh = x / (vn.unsqueeze(-1) + 1e-8)
        rot = self._rotate_forward(xh, Pi)
        idx = (rot.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        xm = self._rotate_inverse(centroids[idx], Pi)
        r = xh - xm
        rn = r.norm(dim=-1)
        sign_bits = (r @ S.T >= 0).to(torch.uint8)
        N = vecs.shape[0]
        mse_bits = self._mse_bits
        mse_bytes = self._mse_bytes
        qjl_bytes = self._qjl_bytes
        if mse_bits == 1:
            idx_g = idx[:, :mse_bytes * 8].reshape(N, mse_bytes, 8)
            mse_packed = (idx_g.to(torch.uint8) << self._shifts_sign
                          ).sum(dim=-1).to(torch.uint8)
        elif mse_bits == 2:
            idx_g = idx[:, :mse_bytes * 4].reshape(N, mse_bytes, 4)
            mse_packed = (idx_g.to(torch.uint8) << self._shifts_mse
                          ).sum(dim=-1).to(torch.uint8)
        else:
            mse_packed = torch.zeros(N, mse_bytes, dtype=torch.uint8, device=device)
            for j in range(D):
                bo = j * mse_bits; bi = bo // 8; bs = bo % 8
                val = idx[:, j].to(torch.uint8) & self._mask
                mse_packed[:, bi] |= (val << bs) & 0xFF
                if bs + mse_bits > 8 and bi + 1 < mse_bytes:
                    mse_packed[:, bi + 1] |= (val >> (8 - bs)) & 0xFF
        sign_g = sign_bits[:, :qjl_bytes * 8].reshape(N, qjl_bytes, 8)
        sign_packed = (sign_g << self._shifts_sign).sum(dim=-1).to(torch.uint8)
        vn_u8 = vn.to(torch.float16).view(torch.uint8).reshape(N, 2)
        rn_u8 = rn.to(torch.float16).view(torch.uint8).reshape(N, 2)
        return torch.cat([mse_packed, sign_packed, vn_u8, rn_u8], dim=-1)

    @torch.compiler.disable
    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Pack K+V into compressed uint8 cache.

        During CUDA Graph capture, this is a no-op — the actual packing
        happens at replay time with real data. The capture just sees
        the tensor shapes.
        """
        if self.kv_sharing_target_layer_name is not None:
            return

        D = self.head_size
        device = key.device
        block_size = kv_cache.shape[2]

        Pi, S, centroids = self._get_matrices(layer, device)

        # vLLM may pass key/value as 2D [N, Hkv*D] or 3D [N, Hkv, D]
        if key.dim() == 2:
            num_tokens = key.shape[0]
            num_heads = self.num_kv_heads
            key = key.reshape(num_tokens, num_heads, D)
            value = value.reshape(num_tokens, num_heads, D)
        else:
            num_tokens, num_heads = key.shape[0], key.shape[1]

        # Trace KV write (first call only per layer)
        # os imported at module level
        _mq_dbg = os.environ.get("MQ_DEBUG")
        if _mq_dbg:
            _lid = getattr(layer, '_mq_layer_id', -1)
            _kv_cnt = getattr(layer, '_mq_kv_write_cnt', 0)
            layer._mq_kv_write_cnt = _kv_cnt + 1
            if _kv_cnt < 3:
                sm = slot_mapping
                n_valid = (sm >= 0).sum().item()
                n_zero = (sm == 0).sum().item()
                # Check forward context for real slot_mapping
                fc_info = "N/A"
                try:
                    from vllm.forward_context import get_forward_context
                    fc = get_forward_context()
                    if hasattr(fc, 'slot_mapping') and fc.slot_mapping:
                        fc_sm = fc.slot_mapping
                        if isinstance(fc_sm, dict):
                            for k, v in list(fc_sm.items())[:1]:
                                fc_info = (f"dict[{k}]: valid="
                                           f"{int((v>=0).sum())} "
                                           f"first4={v[:4].tolist()}")
                except Exception as ex:
                    fc_info = f"err:{ex}"
                logger.info(
                    "[MQ_KV] L%02d write#%d: key=%s slots=%d "
                    "valid=%d zero=%d first8=%s "
                    "fc=%s",
                    _lid, _kv_cnt, list(key.shape), len(sm),
                    n_valid, n_zero,
                    sm[:8].tolist(), fc_info)

        # Pack K/V and write to KV cache
        k_flat = key.reshape(-1, D)
        v_flat = value.reshape(-1, D)
        s_b = kv_cache.stride(0)
        s_kv = kv_cache.stride(1)
        s_s = kv_cache.stride(2)
        s_h = kv_cache.stride(3)
        # slot_mapping must be int32 for CUDA kernels
        sm32 = slot_mapping.int() if slot_mapping.dtype != torch.int32 \
            else slot_mapping

        if self._is_wht:
            pi_blocks = self._get_pi_blocks(layer, device)

            # Try fused pack-to-cache (1 kernel = pack + scatter, no temp buffer)
            fused_ok = False
            if self._is_block_rot:
                from vllm.v1.attention.ops.triton_mq_fused_decode import (
                    _load_blockrot_pack_kernel,
                )
                pk = _load_blockrot_pack_kernel()
                if pk is not None and hasattr(pk, 'tq_blockrot_pack_to_cache'):
                    pk.tq_blockrot_pack_to_cache(
                        key, pi_blocks, kv_cache, sm32, 0,
                        s_b, s_kv, s_s, s_h)
                    pk.tq_blockrot_pack_to_cache(
                        value, pi_blocks, kv_cache, sm32, 1,
                        s_b, s_kv, s_s, s_h)
                    fused_ok = True
            else:
                from vllm.v1.attention.ops.triton_mq_fused_decode import (
                    _load_wht_pack_kernel,
                )
                pk = _load_wht_pack_kernel()
                if pk is not None and hasattr(pk, 'tq_wht_pack_to_cache'):
                    # Pack kernel expects bf16; cast if model uses fp16
                    k_bf = key.to(torch.bfloat16) if key.dtype != torch.bfloat16 else key
                    v_bf = value.to(torch.bfloat16) if value.dtype != torch.bfloat16 else value
                    pk.tq_wht_pack_to_cache(
                        k_bf, kv_cache, sm32, 0,
                        s_b, s_kv, s_s, s_h, self._mse_bits)
                    pk.tq_wht_pack_to_cache(
                        v_bf, kv_cache, sm32, 1,
                        s_b, s_kv, s_s, s_h, self._mse_bits)
                    fused_ok = True

            if not fused_ok:
                # Fallback: pack to buffer + scatter
                N_vecs = k_flat.shape[0]
                ps = self._packed_size
                if not hasattr(self, '_pack_k_buf') or \
                        self._pack_k_buf.shape[0] < N_vecs:
                    max_n = max(N_vecs, 256 * self.num_kv_heads)
                    self._pack_k_buf = torch.empty(
                        max_n, ps, dtype=torch.uint8, device=device)
                    self._pack_v_buf = torch.empty(
                        max_n, ps, dtype=torch.uint8, device=device)
                k_packed = self._pack_k_buf[:N_vecs]
                v_packed = self._pack_v_buf[:N_vecs]
                if self._is_block_rot and pk is not None:
                    pk.tq_blockrot_pack(k_flat, pi_blocks, k_packed)
                    pk.tq_blockrot_pack(v_flat, pi_blocks, v_packed)
                elif not self._is_block_rot and pk is not None:
                    pk.tq_wht_pack(k_flat, k_packed)
                    pk.tq_wht_pack(v_flat, v_packed)
                else:
                    from vllm.multiquant.turboquant.wht_quantizer import (
                        pack_wht,
                    )
                    k_packed = pack_wht(
                        k_flat.float(), self._tq_config, Pi_blocks=pi_blocks)
                    v_packed = pack_wht(
                        v_flat.float(), self._tq_config, Pi_blocks=pi_blocks)

                k_packed = k_packed.reshape(num_tokens, num_heads, -1)
                v_packed = v_packed.reshape(num_tokens, num_heads, -1)
                slots_clamped = slot_mapping.clamp(min=0)
                bi = slots_clamped // block_size
                bo = slots_clamped % block_size
                kv_cache[bi, 0, bo, :, :self._packed_size] = k_packed
                kv_cache[bi, 1, bo, :, :self._packed_size] = v_packed
        elif torch.cuda.is_current_stream_capturing():
            k_packed = self._torch_pack(k_flat, Pi, S, centroids, D, device)
            v_packed = self._torch_pack(v_flat, Pi, S, centroids, D, device)
            k_packed = k_packed.reshape(num_tokens, num_heads, -1)
            v_packed = v_packed.reshape(num_tokens, num_heads, -1)
            slots_clamped = slot_mapping.clamp(min=0)
            bi = slots_clamped // block_size
            bo = slots_clamped % block_size
            kv_cache[bi, 0, bo, :, :self._packed_size] = k_packed
            kv_cache[bi, 1, bo, :, :self._packed_size] = v_packed
        else:
            k_packed = self._pack_batch(k_flat, Pi, S, centroids, D)
            v_packed = self._pack_batch(v_flat, Pi, S, centroids, D)
            k_packed = k_packed.reshape(num_tokens, num_heads, -1)
            v_packed = v_packed.reshape(num_tokens, num_heads, -1)
            slots_clamped = slot_mapping.clamp(min=0)
            bi = slots_clamped // block_size
            bo = slots_clamped % block_size
            kv_cache[bi, 0, bo, :, :self._packed_size] = k_packed
            kv_cache[bi, 1, bo, :, :self._packed_size] = v_packed

        if _mq_dbg and _kv_cnt < 3:
            # Verify: read back what we just wrote
            if len(slots) > 0:
                s0 = slots[0]
                b0, o0 = s0 // block_size, s0 % block_size
                written = kv_cache[b0, 0, o0, 0, :self._packed_size]
                logger.info("[MQ_KV] L%02d verify slot=%d: written_k[0]=%s",
                            _lid, s0.item(),
                            written[:8].tolist())

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        """Dispatch to decode (graph-safe) or prefill (compiler-disabled).

        forward_includes_kv_cache_update=True: KV cache write happens here
        using attn_metadata.slot_mapping (correctly populated by vLLM scheduler),
        not via unified_kv_cache_update (which gets empty slot_mapping during prefill).
        """
        D = self.head_size
        N = query.shape[0]

        # vLLM passes 3D tensors [N, heads, D] — flatten to 2D
        if query.dim() == 3:
            query = query.reshape(N, -1)
        if key is not None and key.dim() == 3:
            key = key.reshape(key.shape[0], -1)
        if value is not None and value.dim() == 3:
            value = value.reshape(value.shape[0], -1)

        output_3d = False
        if output is not None and output.dim() == 3:
            output_3d = True
            out_shape = output.shape
            output = output.view(N, -1)
        elif output is None:
            output = torch.empty(N, self.num_heads * D,
                                 device=query.device, dtype=query.dtype)

        if attn_metadata is None:
            if output_3d:
                return output.fill_(0).view(out_shape)
            return output.fill_(0)

        # KV cache update: use attn_metadata.slot_mapping (from scheduler)
        # pack_wht uses ~0.4 MB per call at B=1 (graph-safe with GPU caches)
        if (key is not None and value is not None
                and self.kv_sharing_target_layer_name is None):
            slot_mapping = attn_metadata.slot_mapping
            num_actual = attn_metadata.num_prefill_tokens + \
                         attn_metadata.num_decode_tokens
            # Trace slot_mapping in forward (first few calls)
            if os.environ.get("MQ_DEBUG"):
                _lid = getattr(layer, '_mq_layer_id', -1)
                _fwd_cnt = getattr(self, '_fwd_kv_cnt', 0)
                self._fwd_kv_cnt = _fwd_cnt + 1
                if _fwd_cnt < 50:
                    sm = slot_mapping[:num_actual]
                    _v = int((sm >= 0).sum())
                    _u = int(sm[:_v].unique().numel()) if _v > 0 else 0
                    logger.info(
                        "[MQ_FWD_KV] L%02d #%d: actual=%d sm_u=%d "
                        "first8=%s kv_ptr=%x",
                        _lid, _fwd_cnt, num_actual,
                        _u, sm[:8].tolist(),
                        kv_cache.data_ptr())
            k3d = key[:num_actual].reshape(-1, self.num_kv_heads, D)
            v3d = value[:num_actual].reshape(-1, self.num_kv_heads, D)
            self.do_kv_cache_update(
                layer, k3d, v3d, kv_cache, slot_mapping[:num_actual])

        device = query.device
        Pi, S, centroids = self._get_matrices(layer, device)
        # Cache Pi_blocks for block-rot decode (needs layer reference)
        if self._is_block_rot:
            self._get_pi_blocks(layer, device)
        block_size = kv_cache.shape[2]

        num_prefill = attn_metadata.num_prefill_tokens
        num_decode = attn_metadata.num_decode_tokens

        # Layer tracing (MQ_DEBUG=1): log norms per layer per step
        # os imported at module level
        _mq_dbg = os.environ.get("MQ_DEBUG")
        if _mq_dbg:
            # Identify layer by Pi data_ptr (unique per layer)
            _lid = getattr(layer, '_mq_layer_id', None)
            if _lid is None:
                if not hasattr(MultiQuantImpl, '_mq_layer_counter'):
                    MultiQuantImpl._mq_layer_counter = 0
                _lid = MultiQuantImpl._mq_layer_counter
                MultiQuantImpl._mq_layer_counter += 1
                layer._mq_layer_id = _lid
            if num_decode > 0:
                sl = attn_metadata.seq_lens[:1].item()
                qn = query[:num_decode].float().norm().item()
                bt = attn_metadata.block_table[:1, :4].tolist()
                logger.info("[MQ] L%02d sl=%d q=%.1f bt=%s pf=%d dc=%d",
                            _lid, sl, qn, bt, num_prefill, num_decode)

        if num_prefill > 0:
            self._forward_prefill(
                query, key, value, output, Pi, S, centroids,
                num_decode, num_prefill, D, device, layer=layer,
            )

        if num_decode > 0:
            self._forward_decode(
                query, output, kv_cache, Pi, S, centroids,
                attn_metadata, num_decode, block_size, D, device,
            )
            if _mq_dbg and _lid is not None:
                on = output[:num_decode].float().norm().item()
                bt = attn_metadata.block_table[:1, :4].tolist()
                sl = attn_metadata.seq_lens[:1].item()
                logger.info("[MQ] L%02d sl=%d out=%.1f bt=%s",
                            _lid, sl, on, bt)
                # On first decode, compare with stored prefill K/V
                if not hasattr(layer, '_dbg_ref_done') and \
                        hasattr(layer, '_dbg_pf_k'):
                    layer._dbg_ref_done = True
                    # BF16 reference: use stored prefill K/V
                    pf_k = layer._dbg_pf_k  # [L, nkv, D]
                    pf_v = layer._dbg_pf_v
                    L = pf_k.shape[0]
                    dq = query[:1].reshape(1, self.num_heads, D)
                    # GQA expand
                    kg = pf_k.unsqueeze(2).expand(
                        -1,-1,self.num_kv_groups,-1
                    ).reshape(L, self.num_heads, D).float()
                    vg = pf_v.unsqueeze(2).expand(
                        -1,-1,self.num_kv_groups,-1
                    ).reshape(L, self.num_heads, D).float()
                    sc = torch.einsum('bhd,thd->bht', dq.float(), kg) * self.scale
                    wt = F.softmax(sc, dim=-1)
                    ref_out = torch.einsum('bht,thd->bhd', wt, vg)
                    ref_flat = ref_out.reshape(1, -1).to(output.dtype)
                    cos = F.cosine_similarity(
                        output[:1].float(), ref_flat.float(), dim=-1).item()
                    logger.info(
                        "[MQ_REF] L%02d DECODE vs BF16_REF: cos=%.4f "
                        "out_norm=%.1f ref_norm=%.1f",
                        _lid, cos, output[:1].float().norm().item(),
                        ref_flat.float().norm().item())

        if output_3d:
            return output.view(out_shape)
        return output

    @torch.compiler.disable
    def _forward_prefill(self, query, key, value, output, Pi, S, centroids,
                         num_decode, num_prefill, D, device, layer=None):
        """Prefill: naive causal bmm. Not graph-captured."""
        L = num_prefill
        pq = query[num_decode:num_decode + L].reshape(L, self.num_heads, D)
        pk = key[num_decode:num_decode + L].reshape(L, self.num_kv_heads, D)
        pv = value[num_decode:num_decode + L].reshape(L, self.num_kv_heads, D)

        # Store prefill K/V for decode reference comparison
        if layer is not None and os.environ.get("MQ_DEBUG") \
                and not hasattr(layer, '_dbg_pf_k'):
            layer._dbg_pf_k = pk.detach().clone()
            layer._dbg_pf_v = pv.detach().clone()

        if self.num_kv_groups > 1:
            pk = pk.repeat_interleave(self.num_kv_groups, dim=1)
            pv = pv.repeat_interleave(self.num_kv_groups, dim=1)

        scores = torch.bmm(
            pq.transpose(0, 1).float(),
            pk.transpose(0, 1).float().transpose(-2, -1)
        ) * self.scale
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0)
        weights = F.softmax(scores, dim=-1)
        prefill_out = torch.bmm(weights, pv.transpose(0, 1).float())
        output[num_decode:num_decode + L] = prefill_out.transpose(0, 1).reshape(
            L, -1).to(output.dtype)

    def _forward_decode(self, query, output, kv_cache, Pi, S, centroids,
                        attn_metadata, num_decode, block_size, D, device):
        """Decode: WHT → Python fallback, TQ → CUDA kernel, RQ → RQ kernel."""
        dq = query[:num_decode].reshape(num_decode, self.num_heads, D)

        if self._is_wht:
            s_b, s_kv, s_s, s_h = (kv_cache.stride(0), kv_cache.stride(1),
                                    kv_cache.stride(2), kv_cache.stride(3))
            out_3d = output[:num_decode].reshape(
                num_decode, self.num_heads, D)
            bt = attn_metadata.block_table[:num_decode]
            sl = attn_metadata.seq_lens[:num_decode]
            if sl.dtype != torch.int32:
                sl = sl.int()

            # Try split-KV kernel first (WHT only, not block-rot yet)
            splitkv_ok = False
            if not self._is_block_rot:
                from vllm.v1.attention.ops.triton_mq_fused_decode import (
                    _load_splitkv_decode_kernel,
                )
                skv_kernel = _load_splitkv_decode_kernel()
                if skv_kernel is not None:
                    NUM_SPLITS = 8
                    num_qh = num_decode * self.num_heads
                    # Pre-allocate ONCE. Max 32 batch × heads to limit memory.
                    # Larger batches fall through to original kernel.
                    MAX_DECODE_QH = 32 * self.num_heads
                    if not hasattr(self, '_split_buf'):
                        self._split_buf = torch.empty(
                            MAX_DECODE_QH, NUM_SPLITS, D + 2,
                            dtype=torch.float32, device=device)
                    if num_qh <= MAX_DECODE_QH:
                        split_buf = self._split_buf[:num_qh]
                        skv_kernel.tq_wht_splitkv_decode(
                            dq, kv_cache, bt, sl,
                            split_buf, out_3d,
                            D, self._mse_bits, self.scale,
                            s_b, s_kv, s_s, s_h, NUM_SPLITS)
                        splitkv_ok = True

            if not splitkv_ok:
                # Fallback: original single-block CUDA kernel
                kernel = None
                if self._is_block_rot:
                    from vllm.v1.attention.ops.triton_mq_fused_decode import (
                        _load_blockrot_decode_kernel,
                    )
                    kernel = _load_blockrot_decode_kernel()
                else:
                    from vllm.v1.attention.ops.triton_mq_fused_decode import (
                        _load_wht_cuda_kernel,
                    )
                    kernel = _load_wht_cuda_kernel()
                if kernel is not None:
                    if self._is_block_rot:
                        pi_blk = getattr(self, '_cached_pi_blocks', None)
                        kernel.tq_blockrot_fused_decode_attention(
                            dq, pi_blk, kv_cache, bt, sl,
                            out_3d, D, self._mse_bits, self.scale,
                            s_b, s_kv, s_s, s_h)
                    else:
                        kernel.tq_wht_fused_decode_attention(
                            dq, kv_cache, bt, sl,
                            out_3d, D, self._mse_bits, self.scale,
                            s_b, s_kv, s_s, s_h)
                else:
                    # Last resort: Python fallback
                    if not hasattr(self, '_wht_torch_logged'):
                        self._wht_torch_logged = True
                        mode = "block-rot" if self._is_block_rot \
                            else "WHT"
                        logger.warning("%s decode: PyTorch fallback",
                                       mode)
                    decode_out = self._wht_decode_torch(
                        dq, kv_cache, attn_metadata, num_decode,
                        block_size, D, device)
                    output[:num_decode] = decode_out.reshape(
                        num_decode, -1).to(output.dtype)
            return

        if self._is_rq:
            # RQ: separate CUDA kernel with Clifford V decompression
            from vllm.v1.attention.ops.triton_mq_fused_decode import (
                _load_rq_decode_kernel, _rq_rotate_forward,
            )
            rq_kernel = _load_rq_decode_kernel()
            if rq_kernel is not None:
                # Pre-allocate RQ decode buffers (once)
                rq_key = (num_decode, self.num_heads, D, device)
                if not hasattr(self, '_rq_decode_bufs') or \
                        self._rq_decode_bufs.get("key") != rq_key:
                    self._rq_decode_bufs = {
                        "key": rq_key,
                        "q_flat": torch.empty(
                            num_decode * self.num_heads, D,
                            device=device, dtype=torch.float32),
                        "q_rot": torch.empty(
                            num_decode * self.num_heads, D,
                            device=device, dtype=torch.float32),
                        "q_proj": torch.empty(
                            num_decode * self.num_heads, D,
                            device=device, dtype=torch.float32),
                        "rq_out": torch.empty(
                            num_decode, self.num_heads, D,
                            device=device, dtype=torch.float32),
                    }
                rb = self._rq_decode_bufs

                # Graph-safe: copy_ handles dtype conversion in-place
                rb["q_flat"].copy_(dq.reshape(num_decode * self.num_heads, D))
                _rq_rotate_forward(rb["q_flat"], Pi, out=rb["q_rot"])
                torch.mm(rb["q_flat"], S.T, out=rb["q_proj"])

                q_rot_3d = rb["q_rot"].reshape(num_decode, self.num_heads, D)
                q_proj_3d = rb["q_proj"].reshape(num_decode, self.num_heads, D)

                s_block = kv_cache.stride(0)
                s_kv = kv_cache.stride(1)
                s_slot = kv_cache.stride(2)
                s_head = kv_cache.stride(3)
                rq_kernel.rq_fused_decode_attention(
                    q_rot_3d, q_proj_3d, kv_cache,
                    Pi, S, centroids,
                    attn_metadata.block_table[:num_decode].int(),
                    attn_metadata.seq_lens[:num_decode].int(),
                    rb["rq_out"],
                    D, self._mse_bits, centroids.shape[0], self.scale,
                    s_block, s_kv, s_slot, s_head,
                )
                # copy_ handles float32→bfloat16 in-place (no allocation)
                output[:num_decode].copy_(
                    rb["rq_out"].reshape(num_decode, -1))
            else:
                # Fallback: Python loop (not graph-safe)
                self._decode_python_loop(
                    dq, kv_cache, Pi, S, centroids,
                    attn_metadata.seq_lens[:num_decode],
                    attn_metadata.block_table[:num_decode],
                    block_size, D, device, output,
                )
        else:
            # TQ: CUDA fused kernel with full V decompression via Pi/S GEMV
            if os.environ.get("MQ_DEBUG"):
                _lid2 = getattr(layer, '_mq_layer_id', -1)
                if not hasattr(layer, '_dc_trace_cnt'):
                    layer._dc_trace_cnt = 0
                layer._dc_trace_cnt += 1
                if layer._dc_trace_cnt <= 2:
                    bt = attn_metadata.block_table[:num_decode]
                    sl = attn_metadata.seq_lens[:num_decode]
                    logger.info(
                        "[MQ_DC] L%02d bt=%s sl=%s",
                        _lid2, bt[0, :4].tolist(), sl.tolist())
            decode_out = self._decode_fn(
                q=dq, kv_cache=kv_cache, Pi=Pi, S=S,
                centroids=centroids,
                block_table=attn_metadata.block_table[:num_decode],
                seq_lens=attn_metadata.seq_lens[:num_decode],
                scale=self.scale, block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                mse_bits=self._mse_bits,
                correction=self._correction,
                is_rq=False,
            )
            output[:num_decode] = decode_out.reshape(num_decode, -1).to(output.dtype)

    # --- Helpers ---

    @torch.compiler.disable
    def _decode_python_loop(self, dq, kv_cache, Pi, S, centroids,
                            seq_lens, block_table, block_size, D,
                            device, output):
        """RQ fallback decode — Python loop over batch × heads × tokens."""
        num_decode = dq.shape[0]
        for qi in range(num_decode):
            sl = seq_lens[qi].item()
            if sl <= 0:
                continue
            positions = torch.arange(sl, device=device)
            bi_log = positions // block_size
            bo = positions % block_size
            bi_phys = block_table[qi, bi_log.long()]

            for kv_h in range(self.num_kv_heads):
                k_packed = kv_cache[bi_phys, 0, bo, kv_h]
                v_packed = kv_cache[bi_phys, 1, bo, kv_h]

                # CUDA unpack with .contiguous()
                try:
                    from vllm.multiquant.weight_quant.archer_ops import cuda_unpack
                    k_result = cuda_unpack(k_packed.contiguous(), D, self._mse_bits)
                except Exception as e:
                    logger.debug("K cuda_unpack fallback: %s", e)
                    k_result = None

                if k_result is not None:
                    idx_all, signs_all, k_vn, k_rn = k_result
                    idx_all = idx_all.long()
                else:
                    no = self._mse_bytes + self._qjl_bytes
                    idx_all = torch.zeros(sl, D, dtype=torch.long, device=device)
                    for j in range(D):
                        boff = j * self._mse_bits
                        bi = boff // 8
                        bs = boff % 8
                        bv = k_packed[:, bi].long() >> bs
                        spill = bs + self._mse_bits - 8
                        if spill > 0 and bi + 1 < self._mse_bytes:
                            bv = bv | (k_packed[:, bi + 1].long() << (self._mse_bits - spill))
                        idx_all[:, j] = bv & self._mask
                    signs_all = torch.zeros(sl, D, dtype=torch.float32, device=device)
                    for b in range(self._qjl_bytes):
                        bv = k_packed[:, self._mse_bytes + b].long()
                        for k in range(8):
                            j = b * 8 + k
                            if j >= D:
                                break
                            signs_all[:, j] = torch.where(
                                ((bv >> k) & 1).bool(),
                                torch.ones(sl, device=device),
                                -torch.ones(sl, device=device))
                    vn_bytes = k_packed[:, no:no + 2].contiguous()
                    rn_bytes = k_packed[:, no + 2:no + 4].contiguous()
                    k_vn = vn_bytes.view(torch.float16).float().squeeze(-1)
                    k_rn = rn_bytes.view(torch.float16).float().squeeze(-1)

                no = self._mse_bytes + self._qjl_bytes
                c_idx = centroids[idx_all]

                for h in range(kv_h * self.num_kv_groups,
                               (kv_h + 1) * self.num_kv_groups):
                    q_rot_h = self._rotate_forward(
                        dq[qi, h].float().unsqueeze(0), Pi
                    ).squeeze(0)
                    q_proj_h = dq[qi, h].float() @ S.T
                    term1 = (q_rot_h.unsqueeze(0) * c_idx).sum(-1)
                    term2 = (q_proj_h.unsqueeze(0) * signs_all).sum(-1)
                    scores = k_vn * (
                        term1 + self._correction * k_rn * term2
                    ) * self.scale

                    # V unpack
                    try:
                        v_result = cuda_unpack(v_packed.contiguous(), D, self._mse_bits)
                    except Exception as e:
                        logger.debug("V cuda_unpack fallback: %s", e)
                        v_result = None

                    if v_result is not None:
                        v_idx, v_signs, v_vn, v_rn = v_result
                        v_idx = v_idx.long()
                    else:
                        v_idx = torch.zeros(sl, D, dtype=torch.long, device=device)
                        for j in range(D):
                            boff = j * self._mse_bits
                            bi = boff // 8
                            bs = boff % 8
                            bv = v_packed[:, bi].long() >> bs
                            spill = bs + self._mse_bits - 8
                            if spill > 0 and bi + 1 < self._mse_bytes:
                                bv = bv | (v_packed[:, bi + 1].long() << (self._mse_bits - spill))
                            v_idx[:, j] = bv & self._mask
                        v_signs = torch.zeros(sl, D, dtype=torch.float32, device=device)
                        for b in range(self._qjl_bytes):
                            bv = v_packed[:, self._mse_bytes + b].long()
                            for bk in range(8):
                                j = b * 8 + bk
                                if j >= D:
                                    break
                                v_signs[:, j] = torch.where(
                                    ((bv >> bk) & 1).bool(),
                                    torch.ones(sl, device=device),
                                    -torch.ones(sl, device=device))
                        v_vn_bytes = v_packed[:, no:no + 2].contiguous()
                        v_rn_bytes = v_packed[:, no + 2:no + 4].contiguous()
                        v_vn = v_vn_bytes.view(torch.float16).float().squeeze(-1)
                        v_rn = v_rn_bytes.view(torch.float16).float().squeeze(-1)
                    v_c = centroids[v_idx]
                    v_xm = self._rotate_inverse(v_c, Pi)
                    v_xq = self._correction * v_rn.unsqueeze(-1) * (v_signs @ S)
                    v_recon = v_vn.unsqueeze(-1) * (v_xm + v_xq)

                    weights = F.softmax(scores, dim=-1)
                    out_h = (weights.unsqueeze(-1) * v_recon).sum(0)
                    output[qi, h * D:(h + 1) * D] = out_h.to(output.dtype)

    def _wht_decode_torch(self, dq, kv_cache, attn_metadata,
                          num_decode, block_size, D, device):
        """Graph-safe WHT decode using pure PyTorch ops.

        Supports batch decode (num_decode >= 1) for CUDA Graph capture.
        No .item(), no data-dependent shapes, no Python loops.
        """
        from vllm.multiquant.turboquant.wht_quantizer import unpack_wht

        B = num_decode
        seq_lens = attn_metadata.seq_lens[:B]  # [B]
        block_table = attn_metadata.block_table[:B]  # [B, max_blocks]
        ps = self._packed_size
        Hkv = self.num_kv_heads
        Hq = self.num_heads

        # Use max_seq_len from metadata (set by scheduler, fixed per capture)
        max_sl = attn_metadata.max_seq_len
        if max_sl <= 0:
            max_sl = 1
        if not hasattr(self, '_wht_maxsl_logged'):
            self._wht_maxsl_logged = True
            logger.info("WHT decode torch: max_sl=%d B=%d", max_sl, B)

        # Positions [max_sl]
        positions = torch.arange(max_sl, device=device)
        bi = positions // block_size  # [max_sl]
        bo = positions % block_size

        # For batch: gather per-batch-item
        # block_table: [B, max_blocks] → phys_blocks: [B, max_sl]
        phys = block_table[:, bi.long()]  # [B, max_sl]

        # Gather K/V: [B, max_sl, Hkv, ps]
        bo_exp = bo.unsqueeze(0).expand(B, -1)  # [B, max_sl]
        k_packed = kv_cache[phys, 0, bo_exp, :, :ps]  # [B, max_sl, Hkv, ps]
        v_packed = kv_cache[phys, 1, bo_exp, :, :ps]

        # Unpack: [B*max_sl*Hkv, ps] → [B*max_sl*Hkv, D]
        k_flat = k_packed.reshape(B * max_sl * Hkv, ps)
        v_flat = v_packed.reshape(B * max_sl * Hkv, ps)
        pi_blk = getattr(self, '_cached_pi_blocks', None)
        k_recon = unpack_wht(
            k_flat, self._tq_config, Pi_blocks=pi_blk
        ).reshape(B, max_sl, Hkv, D)
        v_recon = unpack_wht(
            v_flat, self._tq_config, Pi_blocks=pi_blk
        ).reshape(B, max_sl, Hkv, D)

        # GQA expand: [B, T, Hkv, D] → [B, T, Hq, D]
        if self.num_kv_groups > 1:
            k_recon = k_recon.unsqueeze(3).expand(
                -1, -1, -1, self.num_kv_groups, -1).reshape(B, max_sl, Hq, D)
            v_recon = v_recon.unsqueeze(3).expand(
                -1, -1, -1, self.num_kv_groups, -1).reshape(B, max_sl, Hq, D)

        # Attention: Q[B, Hq, D] · K[B, T, Hq, D] → scores[B, Hq, T]
        q = dq.float()  # [B, Hq, D]
        k = k_recon.float()  # [B, T, Hq, D]
        v = v_recon.float()

        scores = torch.einsum('bhd,bthd->bht', q, k) * self.scale  # [B, Hq, T]

        # Mask: positions beyond seq_len get -inf (handles padding + seq_len=0)
        mask = positions.unsqueeze(0) >= seq_lens.unsqueeze(1)  # [B, max_sl]
        scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        weights = F.softmax(scores, dim=-1)  # [B, Hq, T]
        # NaN guard: seq_len=0 → softmax(all -inf) → NaN → replace with 0
        weights = weights.nan_to_num(0.0)
        out = torch.einsum('bht,bthd->bhd', weights, v)  # [B, Hq, D]

        return out  # [B, Hq, D]

    @torch.compiler.disable
    def _wht_decode_python(self, dq, kv_cache, centroids, attn_metadata,
                           num_decode, block_size, D, device):
        """WHT decode: Python fallback (no CUDA kernel yet).

        Reads packed WHT blocks from kv_cache, reconstructs K/V via
        inverse WHT, computes attention scores and weighted V sum.
        """
        from vllm.multiquant.turboquant.wht_quantizer import unpack_wht

        seq_lens = attn_metadata.seq_lens[:num_decode]
        block_table = attn_metadata.block_table[:num_decode]
        ps = self._packed_size

        outputs = []
        for qi in range(num_decode):
            sl = seq_lens[qi].item()
            if sl <= 0:
                outputs.append(torch.zeros(self.num_heads, D,
                                           device=device, dtype=torch.float32))
                continue

            # Gather K/V from cache via block_table
            positions = torch.arange(sl, device=device)
            bi = positions // block_size
            bo = positions % block_size
            phys_blocks = block_table[qi, bi.long()]
            # [sl, num_kv_heads, packed_size]
            k_packed = kv_cache[phys_blocks, 0, bo, :, :ps]
            v_packed = kv_cache[phys_blocks, 1, bo, :, :ps]

            # Unpack per head
            pi_blk = getattr(self, '_cached_pi_blocks', None)
            q_h = dq[qi]  # [num_heads, D]
            head_outputs = []
            for h in range(self.num_kv_heads):
                k_h = unpack_wht(k_packed[:, h, :], self._tq_config,
                                 Pi_blocks=pi_blk)  # [sl, D]
                v_h = unpack_wht(v_packed[:, h, :], self._tq_config,
                                 Pi_blocks=pi_blk)  # [sl, D]

                # GQA: expand to all q-heads in this group
                kv_group_size = self.num_kv_groups
                for g in range(kv_group_size):
                    qh_idx = h * kv_group_size + g
                    q_vec = q_h[qh_idx].float()  # [D]
                    scores = (k_h.float() @ q_vec) * self.scale  # [sl]
                    weights = F.softmax(scores, dim=0)  # [sl]
                    out_h = (weights.unsqueeze(-1) * v_h.float()).sum(0)  # [D]
                    head_outputs.append(out_h)

            outputs.append(torch.stack(head_outputs))  # [num_heads, D]

        return torch.stack(outputs)  # [num_decode, num_heads, D]

    def _get_pi_blocks(self, layer, device):
        """Get cached block rotation matrices for tq3r/tq4r. None for WHT."""
        if not self._is_block_rot:
            return None
        if not hasattr(layer, '_tq_Pi_blocks_f32'):
            layer._tq_Pi_blocks_f32 = (
                layer._tq_Pi_blocks.to(device).float().contiguous())
        # Also cache on self for use in decode (which doesn't get layer)
        self._cached_pi_blocks = layer._tq_Pi_blocks_f32
        return layer._tq_Pi_blocks_f32

    def _get_matrices(self, layer, device):
        if self._is_wht:
            # WHT/block-rot mode: no Pi/S, only centroids (+ Pi_blocks for tq3r)
            if not hasattr(layer, '_tq_c_f32'):
                from vllm.multiquant.shared.centroids import get_wht_centroids
                c = get_wht_centroids(self._mse_bits).to(device)
                layer._tq_c_f32 = c.float().contiguous()
            # Return None for Pi/S — callers must check _is_wht
            return None, None, layer._tq_c_f32
        if not hasattr(layer, '_tq_Pi_f32'):
            layer._tq_Pi_f32 = layer._tq_Pi.to(device).float().contiguous()
            layer._tq_S_f32 = layer._tq_S.to(device).float().contiguous()
            layer._tq_c_f32 = layer._tq_centroids.to(device).float().contiguous()
        return layer._tq_Pi_f32, layer._tq_S_f32, layer._tq_c_f32

    def _pack_single(self, vec, Pi, S, centroids, D):
        """Pack a single vector — slow, per-vector. Use _pack_batch instead."""
        return self._pack_batch(vec.unsqueeze(0), Pi, S, centroids, D).squeeze(0)

    def _pack_batch(self, vecs, Pi, S, centroids, D):
        """Pack a batch of vectors — vectorized, no .item() calls."""
        x = vecs.float()
        vn = x.norm(dim=-1)
        xh = x / (vn.unsqueeze(-1) + 1e-8)

        rot = self._rotate_forward(xh, Pi)
        idx = (rot.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        xm = self._rotate_inverse(centroids[idx], Pi)
        r = xh - xm
        rn = r.norm(dim=-1)
        signs = (r @ S.T >= 0).float()
        signs[signs == 0] = -1.0

        from vllm.multiquant.shared.bitpack import pack_vectors_batched
        return pack_vectors_batched(idx, signs, vn, rn, D, self._mse_bits)

    def _score_packed(self, q_rot, q_proj, packed, centroids):
        D = self.head_size
        t1 = 0.0
        if self._mse_bits == 2:
            for b in range(self._mse_bytes):
                bv = packed[b].item()
                for k in range(4):
                    j = b*4+k
                    if j >= D: break
                    t1 += q_rot[j].item() * centroids[(bv >> (k*2)) & self._mask].item()
        t2 = 0.0
        for b in range(self._qjl_bytes):
            bv = packed[self._mse_bytes + b].item()
            for k in range(8):
                j = b*8+k
                if j >= D: break
                t2 += q_proj[j].item() * (1.0 if ((bv >> k) & 1) else -1.0)
        no = self._mse_bytes + self._qjl_bytes
        vn = struct.unpack('e', bytes([packed[no].item(), packed[no+1].item()]))[0]
        rn = struct.unpack('e', bytes([packed[no+2].item(), packed[no+3].item()]))[0]
        return vn * (t1 + self._correction * rn * t2) * self.scale

    def _unpack(self, packed, Pi, S, centroids):
        D = self.head_size
        idx = torch.zeros(D, dtype=torch.long, device=packed.device)
        if self._mse_bits == 2:
            for j in range(D):
                b, k = j//4, j%4
                idx[j] = (packed[b].item() >> (k*2)) & self._mask
        signs = torch.zeros(D, dtype=torch.float32, device=packed.device)
        for j in range(D):
            b, k = j//8, j%8
            signs[j] = 1.0 if ((packed[self._mse_bytes+b].item() >> k) & 1) else -1.0
        no = self._mse_bytes + self._qjl_bytes
        vn = struct.unpack('e', bytes([packed[no].item(), packed[no+1].item()]))[0]
        rn = struct.unpack('e', bytes([packed[no+2].item(), packed[no+3].item()]))[0]
        c_idx = centroids[idx]
        xm = c_idx @ Pi
        xq = self._correction * rn * (signs @ S)
        return vn * (xm + xq)
