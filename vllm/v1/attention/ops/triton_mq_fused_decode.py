# SPDX-License-Identifier: Apache-2.0
"""Fused MultiQuant Decode Kernel — single Triton launch for full decode.

Replaces the Python loop in MultiQuantImpl.forward() with a single kernel
that does: K-unpack → Score → Online Softmax → V-unpack → V-accumulate.

Key insight (Post-Loop GEMV): the D×D inverse rotation is moved OUTSIDE the
token loop. Inside the kernel we accumulate centroid values and signs in
rotated space. After the kernel, two batched GEMVs reconstruct the output:
    output = centroid_acc @ Pi + correction * sign_acc @ S

Total kernel launches: 5 (constant, independent of batch/heads/seq_len):
    1. Q @ Pi.T  (pre-rotate query)
    2. Q @ S.T   (pre-project query)
    3. Triton fused decode (this kernel)
    4. centroid_acc @ Pi  (V reconstruction)
    5. sign_acc @ S       (QJL correction)

TQ only (TQ3/TQ4). RQ keeps Python fallback.
"""

import math
import os

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mq_fused_decode_kernel(
    # Pre-computed query projections [B*Hq, D]
    q_rot_ptr,
    q_proj_ptr,
    # KV cache [num_blocks, 2, block_size, num_kv_heads, packed_size] uint8
    kv_cache_ptr,
    # Centroids [n_centroids] float32
    centroids_ptr,
    # Block table [B, max_blocks] int32
    block_table_ptr,
    # Sequence lengths [B] int32
    seq_lens_ptr,
    # Outputs [B, Hq, D] float32
    centroid_acc_ptr,
    sign_acc_ptr,
    # Strides for kv_cache: [s_block, s_kv, s_bslot, s_head, s_pack]
    s_cache_block: tl.constexpr,   # stride for block dimension
    s_cache_kv: tl.constexpr,      # stride for K/V dimension (=1 means K=0,V=1)
    s_cache_slot: tl.constexpr,    # stride for slot within block
    s_cache_head: tl.constexpr,    # stride for kv_head
    # Strides for block_table
    s_bt_batch: tl.constexpr,      # stride for batch dim in block_table
    # Dims
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,   # Hq // Hkv
    PACKED_SIZE: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    QJL_BYTES: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    # Constants
    correction_scale,              # sqrt(pi/2) / D
    attn_scale,                    # 1/sqrt(D)
):
    """One program per (batch, q_head). Processes all tokens sequentially."""
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // KV_GROUP_SIZE

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if seq_len <= 0:
        return

    # Load pre-rotated/projected query vectors [D]
    d_offs = tl.arange(0, HEAD_DIM)
    q_base = (batch_idx * tl.num_programs(1) + q_head_idx) * HEAD_DIM
    q_rot = tl.load(q_rot_ptr + q_base + d_offs)     # [D]
    q_proj = tl.load(q_proj_ptr + q_base + d_offs)   # [D]

    # Load centroids
    c_offs = tl.arange(0, N_CENTROIDS)
    centroids = tl.load(centroids_ptr + c_offs)       # [n_centroids]

    # Online softmax state
    m_prev = float("-inf")
    l_prev = 0.0

    # V accumulators [D] — in rotated space (no Pi/S needed)
    centroid_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    sign_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Process all cached tokens
    for pos in range(0, seq_len):
        # Block table gather: logical → physical block
        block_logical = pos // BLOCK_SIZE
        slot_in_block = pos % BLOCK_SIZE
        phys_block = tl.load(
            block_table_ptr + batch_idx * s_bt_batch + block_logical
        ).to(tl.int32)

        # ── K: Unpack + Score ───────────────────────────────────
        k_base = (
            phys_block * s_cache_block
            + 0 * s_cache_kv           # K is index 0
            + slot_in_block * s_cache_slot
            + kv_head_idx * s_cache_head
        )

        # Unpack MSE indices → dot(q_rot, centroids[idx])
        term1 = tl.zeros([], dtype=tl.float32)
        if MSE_BITS == 2:
            for b in tl.static_range(MSE_BYTES):
                byte_val = tl.load(kv_cache_ptr + k_base + b).to(tl.int32)
                for k in tl.static_range(4):
                    j = b * 4 + k
                    if j < HEAD_DIM:
                        idx_j = (byte_val >> (k * 2)) & 0x3
                        c_val = tl.sum(tl.where(c_offs == idx_j, centroids, 0.0))
                        q_r_j = tl.sum(tl.where(d_offs == j, q_rot, 0.0))
                        term1 += q_r_j * c_val
        elif MSE_BITS == 3:
            for j in tl.static_range(HEAD_DIM):
                bit_off = j * 3
                byte_idx = bit_off // 8
                bit_idx = bit_off % 8
                b0 = tl.load(kv_cache_ptr + k_base + byte_idx).to(tl.int32)
                idx_j = (b0 >> bit_idx) & 0x7
                if bit_idx > 5 and byte_idx + 1 < MSE_BYTES:
                    b1 = tl.load(
                        kv_cache_ptr + k_base + byte_idx + 1
                    ).to(tl.int32)
                    idx_j = idx_j | ((b1 << (8 - bit_idx)) & 0x7)
                c_val = tl.sum(tl.where(c_offs == (idx_j & 0x7), centroids, 0.0))
                q_r_j = tl.sum(tl.where(d_offs == j, q_rot, 0.0))
                term1 += q_r_j * c_val

        # Unpack QJL signs → dot(q_proj, signs)
        term2 = tl.zeros([], dtype=tl.float32)
        sign_base_k = k_base + MSE_BYTES
        for b in tl.static_range(QJL_BYTES):
            byte_val = tl.load(kv_cache_ptr + sign_base_k + b).to(tl.int32)
            for k in tl.static_range(8):
                j = b * 8 + k
                if j < HEAD_DIM:
                    sign_bit = (byte_val >> k) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    q_p_j = tl.sum(tl.where(d_offs == j, q_proj, 0.0))
                    term2 += q_p_j * sign_val

        # Load K norms (float16 packed as 2 uint8 each)
        k_norm_base = k_base + MSE_BYTES + QJL_BYTES
        vn_lo = tl.load(kv_cache_ptr + k_norm_base).to(tl.uint16)
        vn_hi = tl.load(kv_cache_ptr + k_norm_base + 1).to(tl.uint16)
        vn_u16 = vn_lo | (vn_hi << 8)
        k_vn = vn_u16.to(tl.float16, bitcast=True).to(tl.float32)

        rn_lo = tl.load(kv_cache_ptr + k_norm_base + 2).to(tl.uint16)
        rn_hi = tl.load(kv_cache_ptr + k_norm_base + 3).to(tl.uint16)
        rn_u16 = rn_lo | (rn_hi << 8)
        k_rn = rn_u16.to(tl.float16, bitcast=True).to(tl.float32)

        # Score
        score = k_vn * (term1 + correction_scale * k_rn * term2) * attn_scale

        # ── Online Softmax ──────────────────────────────────────
        m_new = tl.maximum(m_prev, score)
        exp_prev = tl.exp(m_prev - m_new)
        exp_cur = tl.exp(score - m_new)
        l_new = l_prev * exp_prev + exp_cur

        # Rescale accumulators
        centroid_acc = centroid_acc * exp_prev
        sign_acc = sign_acc * exp_prev

        # ── V: Unpack + Accumulate ──────────────────────────────
        v_base = (
            phys_block * s_cache_block
            + 1 * s_cache_kv           # V is index 1
            + slot_in_block * s_cache_slot
            + kv_head_idx * s_cache_head
        )

        # Unpack V MSE indices → centroid gather [D]
        if MSE_BITS == 2:
            for b in tl.static_range(MSE_BYTES):
                byte_val = tl.load(kv_cache_ptr + v_base + b).to(tl.int32)
                for kk in tl.static_range(4):
                    j = b * 4 + kk
                    if j < HEAD_DIM:
                        v_idx_j = (byte_val >> (kk * 2)) & 0x3
                        v_c_val = tl.sum(
                            tl.where(c_offs == v_idx_j, centroids, 0.0)
                        )
                        # Accumulate: centroid_acc[j] += w * v_vn * c_val
                        # Use masked update
                        mask_j = (d_offs == j)
                        # Load v_vn once (will be loaded below for full code)
                        # For now, accumulate centroid value per dim
                        centroid_acc = tl.where(
                            mask_j,
                            centroid_acc + v_c_val,  # placeholder, scaled below
                            centroid_acc,
                        )
        # NOTE: This per-element approach via tl.where is correct but slow for
        # large HEAD_DIM. It will be optimized in a future iteration with
        # vectorized loads. For correctness-first, this pattern matches the
        # existing triton_tq_attention_score.py approach.

        # Actually, let's do the full V unpack + accumulate properly.
        # We need v_vn and v_rn first, then scale.
        # Reset centroid_acc update — do it properly below.

        # The above partial update was incorrect. Let me restructure:
        # Load V norms first, then unpack indices and signs together.
        pass  # Restructure below

    # Placeholder — kernel needs restructuring for V path
    # Store results
    out_base = (batch_idx * tl.num_programs(1) + q_head_idx) * HEAD_DIM
    tl.store(centroid_acc_ptr + out_base + d_offs, centroid_acc)
    tl.store(sign_acc_ptr + out_base + d_offs, sign_acc)


# ============================================================
# The above kernel sketch has a structural problem: Triton's
# tl.where per-element approach for V accumulation is too slow
# for D=128. We need a different strategy.
#
# REVISED APPROACH: Scalar score loop + vectorized V accumulate
# using explicit per-dimension arrays built from byte loads.
# ============================================================

@triton.jit
def _mq_fused_decode_kernel_v2(
    # Pre-computed query projections [B*Hq, D]
    q_rot_ptr,
    q_proj_ptr,
    # KV cache — flat uint8 pointer
    kv_cache_ptr,
    # Centroids [n_centroids] float32
    centroids_ptr,
    # Block table [B, max_blocks] int32
    block_table_ptr,
    # Sequence lengths [B] int32
    seq_lens_ptr,
    # Outputs [B*Hq, D] float32
    centroid_acc_ptr,
    sign_acc_ptr,
    # KV cache strides (in bytes, since uint8)
    stride_cache_block,   # num_blocks dim
    stride_cache_kv,      # K/V dim (2)
    stride_cache_slot,    # block_size dim
    stride_cache_head,    # num_kv_heads dim
    # Block table stride
    stride_bt_batch,
    # Scalar params
    num_q_heads: tl.constexpr,
    # Dims
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PACKED_SIZE: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    QJL_BYTES: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    # Constants
    correction_scale,
    attn_scale,
):
    """Fused MQ decode: score K + softmax + accumulate V centroids/signs.

    Grid: (B, Hq). One program per (batch, query_head).
    Each program iterates over all seq_len tokens sequentially.
    """
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // KV_GROUP_SIZE

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if seq_len <= 0:
        return

    # Load query vectors
    d_offs = tl.arange(0, HEAD_DIM)
    q_base = (batch_idx * num_q_heads + q_head_idx) * HEAD_DIM
    q_rot = tl.load(q_rot_ptr + q_base + d_offs)
    q_proj = tl.load(q_proj_ptr + q_base + d_offs)

    # Load centroids to registers
    c_offs = tl.arange(0, N_CENTROIDS)
    centroids = tl.load(centroids_ptr + c_offs)

    # Online softmax state
    m_prev = float("-inf")
    l_prev = 0.0

    # V accumulators [D]
    v_cent_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    v_sign_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for pos in range(0, seq_len):
        block_log = pos // BLOCK_SIZE
        slot = pos % BLOCK_SIZE
        phys_block = tl.load(
            block_table_ptr + batch_idx * stride_bt_batch + block_log
        ).to(tl.int32)

        base_k = (
            phys_block * stride_cache_block
            + 0 * stride_cache_kv
            + slot * stride_cache_slot
            + kv_head_idx * stride_cache_head
        )

        # ── K Score (scalar) ────────────────────────────────
        # Load packed bytes and compute score serially
        # For 2-bit: 4 indices per byte, MSE_BYTES bytes total
        term1 = tl.zeros([], dtype=tl.float32)
        term2 = tl.zeros([], dtype=tl.float32)

        if MSE_BITS == 2:
            for b in tl.static_range(MSE_BYTES):
                bv = tl.load(kv_cache_ptr + base_k + b).to(tl.int32)
                for ki in tl.static_range(4):
                    j = b * 4 + ki
                    if j < HEAD_DIM:
                        idx = (bv >> (ki * 2)) & 0x3
                        c_val = tl.sum(tl.where(c_offs == idx, centroids, 0.0))
                        q_r = tl.sum(tl.where(d_offs == j, q_rot, 0.0))
                        term1 += q_r * c_val
        elif MSE_BITS == 3:
            for j in tl.static_range(HEAD_DIM):
                bo = j * 3
                bi = bo // 8
                bs = bo % 8
                b0 = tl.load(kv_cache_ptr + base_k + bi).to(tl.int32)
                idx = (b0 >> bs) & 0x7
                if bs > 5 and bi + 1 < MSE_BYTES:
                    b1 = tl.load(kv_cache_ptr + base_k + bi + 1).to(tl.int32)
                    idx = idx | ((b1 << (8 - bs)) & 0x7)
                c_val = tl.sum(tl.where(c_offs == (idx & 0x7), centroids, 0.0))
                q_r = tl.sum(tl.where(d_offs == j, q_rot, 0.0))
                term1 += q_r * c_val

        # QJL signs for K
        for b in tl.static_range(QJL_BYTES):
            bv = tl.load(kv_cache_ptr + base_k + MSE_BYTES + b).to(tl.int32)
            for ki in tl.static_range(8):
                j = b * 8 + ki
                if j < HEAD_DIM:
                    sv = tl.where(((bv >> ki) & 1) == 1, 1.0, -1.0)
                    q_p = tl.sum(tl.where(d_offs == j, q_proj, 0.0))
                    term2 += q_p * sv

        # K norms
        k_no = base_k + MSE_BYTES + QJL_BYTES
        vn_lo = tl.load(kv_cache_ptr + k_no).to(tl.uint16)
        vn_hi = tl.load(kv_cache_ptr + k_no + 1).to(tl.uint16)
        k_vn = (vn_lo | (vn_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        rn_lo = tl.load(kv_cache_ptr + k_no + 2).to(tl.uint16)
        rn_hi = tl.load(kv_cache_ptr + k_no + 3).to(tl.uint16)
        k_rn = (rn_lo | (rn_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        score = k_vn * (term1 + correction_scale * k_rn * term2) * attn_scale

        # ── Online Softmax ──────────────────────────────────
        m_new = tl.maximum(m_prev, score)
        rescale = tl.exp(m_prev - m_new)
        w = tl.exp(score - m_new)
        l_new = l_prev * rescale + w

        v_cent_acc = v_cent_acc * rescale
        v_sign_acc = v_sign_acc * rescale

        # ── V: Unpack + Accumulate [D] ──────────────────────
        base_v = (
            phys_block * stride_cache_block
            + 1 * stride_cache_kv          # V=1
            + slot * stride_cache_slot
            + kv_head_idx * stride_cache_head
        )

        # V norms
        v_no = base_v + MSE_BYTES + QJL_BYTES
        v_vn_lo = tl.load(kv_cache_ptr + v_no).to(tl.uint16)
        v_vn_hi = tl.load(kv_cache_ptr + v_no + 1).to(tl.uint16)
        v_vn = (v_vn_lo | (v_vn_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_rn_lo = tl.load(kv_cache_ptr + v_no + 2).to(tl.uint16)
        v_rn_hi = tl.load(kv_cache_ptr + v_no + 3).to(tl.uint16)
        v_rn = (v_rn_lo | (v_rn_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        # V centroid values per dimension (same unpack as K but vectorized [D])
        v_centroids_d = tl.zeros([HEAD_DIM], dtype=tl.float32)
        if MSE_BITS == 2:
            for b in tl.static_range(MSE_BYTES):
                bv = tl.load(kv_cache_ptr + base_v + b).to(tl.int32)
                for ki in tl.static_range(4):
                    j = b * 4 + ki
                    if j < HEAD_DIM:
                        idx = (bv >> (ki * 2)) & 0x3
                        c_val = tl.sum(tl.where(c_offs == idx, centroids, 0.0))
                        v_centroids_d = tl.where(
                            d_offs == j, c_val, v_centroids_d
                        )
        elif MSE_BITS == 3:
            for j in tl.static_range(HEAD_DIM):
                bo_v = j * 3
                bi_v = bo_v // 8
                bs_v = bo_v % 8
                b0 = tl.load(kv_cache_ptr + base_v + bi_v).to(tl.int32)
                idx = (b0 >> bs_v) & 0x7
                if bs_v > 5 and bi_v + 1 < MSE_BYTES:
                    b1 = tl.load(kv_cache_ptr + base_v + bi_v + 1).to(tl.int32)
                    idx = idx | ((b1 << (8 - bs_v)) & 0x7)
                c_val = tl.sum(tl.where(c_offs == (idx & 0x7), centroids, 0.0))
                v_centroids_d = tl.where(d_offs == j, c_val, v_centroids_d)

        # V signs per dimension [D]
        v_signs_d = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for b in tl.static_range(QJL_BYTES):
            bv = tl.load(kv_cache_ptr + base_v + MSE_BYTES + b).to(tl.int32)
            for ki in tl.static_range(8):
                j = b * 8 + ki
                if j < HEAD_DIM:
                    sv = tl.where(((bv >> ki) & 1) == 1, 1.0, -1.0)
                    v_signs_d = tl.where(d_offs == j, sv, v_signs_d)

        # Accumulate: weighted V components
        v_cent_acc += w * v_vn * v_centroids_d
        v_sign_acc += w * v_vn * v_rn * v_signs_d

        m_prev = m_new
        l_prev = l_new

    # Normalize by softmax denominator
    v_cent_acc = v_cent_acc / l_prev
    v_sign_acc = v_sign_acc / l_prev

    # Store
    out_base = (batch_idx * num_q_heads + q_head_idx) * HEAD_DIM
    tl.store(centroid_acc_ptr + out_base + d_offs, v_cent_acc)
    tl.store(sign_acc_ptr + out_base + d_offs, v_sign_acc)


# ============================================================
# Python wrapper
# ============================================================

_cuda_kernel = None
_cuda_kernel_tried = False
_decode_path_logged = False
_decode_buffers = {}  # Pre-allocated buffers keyed by (B, Hq, D, device)

from vllm.logger import init_logger
logger = init_logger(__name__)


def _load_cuda_kernel():
    """JIT compile the CUDA fused decode kernel."""
    global _cuda_kernel, _cuda_kernel_tried
    if _cuda_kernel_tried:
        return _cuda_kernel
    _cuda_kernel_tried = True

    # os imported at module level
    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "kernels", "turboquant"
    )
    if not os.path.exists(src_dir):
        src_dir = "/opt/tq_build"
    src_file = os.path.join(src_dir, "tq_compressed_attention.cu")
    if not os.path.exists(src_file):
        logger.warning("TQ fused decode: source not found at %s", src_file)
        return None

    try:
        from torch.utils.cpp_extension import load
        _cuda_kernel = load(
            name="tq_fused_decode",
            sources=[src_file],
            extra_cuda_cflags=[
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
                "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("TQ fused decode CUDA kernel compiled and loaded")
        return _cuda_kernel
    except Exception as e:
        logger.warning("TQ fused decode CUDA kernel FAILED: %s", e)
        return None


# ============================================================
# WHT Pack Kernel (fused: WHT → amax → quantize → bitpack)
# ============================================================

_wht_pack_kernel = None
_wht_pack_kernel_tried = False


def _load_wht_pack_kernel():
    """JIT compile the fused WHT pack kernel."""
    global _wht_pack_kernel, _wht_pack_kernel_tried
    if _wht_pack_kernel_tried:
        return _wht_pack_kernel
    _wht_pack_kernel_tried = True

    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "kernels", "turboquant"
    )
    if not os.path.exists(src_dir):
        src_dir = "/opt/tq_build"
    src_file = os.path.join(src_dir, "tq_wht_pack.cu")
    if not os.path.exists(src_file):
        logger.warning("WHT fused pack: source not found at %s", src_file)
        return None

    try:
        from torch.utils.cpp_extension import load
        _wht_pack_kernel = load(
            name="tq_wht_fused_pack",
            sources=[src_file],
            extra_cuda_cflags=[
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
                "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("WHT fused pack CUDA kernel compiled and loaded")
        return _wht_pack_kernel
    except Exception as e:
        logger.warning("WHT fused pack CUDA kernel FAILED: %s", e)
        return None


# ============================================================
# WHT Decode Kernel loader + decode wrapper
# ============================================================

_wht_cuda_kernel = None
_wht_cuda_kernel_tried = False
_wht_decode_path_logged = False
_wht_decode_buffers = {}


def _load_wht_cuda_kernel():
    """JIT compile the WHT fused decode kernel."""
    global _wht_cuda_kernel, _wht_cuda_kernel_tried
    if _wht_cuda_kernel_tried:
        return _wht_cuda_kernel
    _wht_cuda_kernel_tried = True

    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "kernels", "turboquant"
    )
    if not os.path.exists(src_dir):
        src_dir = "/opt/tq_build"
    src_file = os.path.join(src_dir, "tq_wht_decode.cu")
    if not os.path.exists(src_file):
        logger.warning("WHT fused decode: source not found at %s", src_file)
        return None

    try:
        from torch.utils.cpp_extension import load
        _wht_cuda_kernel = load(
            name="tq_wht_fused_decode",
            sources=[src_file],
            extra_cuda_cflags=[
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
                "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("WHT fused decode CUDA kernel compiled and loaded")
        return _wht_cuda_kernel
    except Exception as e:
        logger.warning("WHT fused decode CUDA kernel FAILED: %s", e)
        return None


def wht_fused_decode_attention(
    q: torch.Tensor,           # [B, Hq, D] bfloat16/float16
    kv_cache: torch.Tensor,    # [num_blocks, 2, block_size, Hkv, packed_size] uint8
    block_table: torch.Tensor, # [B, max_blocks] int32
    seq_lens: torch.Tensor,    # [B] int32
    scale: float,              # 1/sqrt(D)
    mse_bits: int,
    block_size_wht: int = 32,  # WHT block size
) -> torch.Tensor:
    """Fused WHT decode: pre-WHT query + CUDA kernel.

    Falls back to Python if CUDA kernel unavailable.
    """
    B, Hq, D = q.shape
    device = q.device

    # Pre-allocated buffers
    global _wht_decode_buffers
    buf_key = (B, Hq, D, device)
    if buf_key not in _wht_decode_buffers:
        _wht_decode_buffers[buf_key] = {
            "q_wht": torch.empty(B, Hq, D, device=device, dtype=torch.float32),
            "output": torch.empty(B, Hq, D, device=device, dtype=torch.float32),
        }
    buf = _wht_decode_buffers[buf_key]

    # Pre-compute WHT of query (graph-safe: pre-allocated buffers)
    from vllm.multiquant.shared.wht import wht_forward
    if 'q_flat' not in buf or buf['q_flat'].shape[0] != B * Hq:
        buf['q_flat'] = torch.empty(B * Hq, D, device=device, dtype=torch.float32)
    buf['q_flat'].copy_(q.reshape(B * Hq, D))
    q_wht_flat = wht_forward(buf['q_flat'], block_size=block_size_wht)
    q_wht = buf["q_wht"]
    q_wht.copy_(q_wht_flat.reshape(B, Hq, D))

    output = buf["output"]
    output.zero_()

    # KV cache strides
    s_block = kv_cache.stride(0)
    s_kv = kv_cache.stride(1)
    s_slot = kv_cache.stride(2)
    s_head = kv_cache.stride(3)

    # Debug: check if kv_cache has any data
    if os.environ.get("MQ_DEBUG"):
        if not hasattr(wht_fused_decode_attention, '_kv_dbg_cnt'):
            wht_fused_decode_attention._kv_dbg_cnt = 0
        wht_fused_decode_attention._kv_dbg_cnt += 1
        if wht_fused_decode_attention._kv_dbg_cnt <= 50:
            sl0 = seq_lens[0].item()
            bt0 = block_table[0, :3].tolist()
            # Read what the kernel would read at pos=0
            phys_b = bt0[0]
            k_bytes = kv_cache[phys_b, 0, 0, 0, :14]  # First WHT block of K
            nz_bytes = int((k_bytes != 0).sum())
            logger.info("[WHT_KV] #%d sl=%d bt=%s "
                        "k_block0_bytes=[%s] nz=%d out_norm=%.2f",
                        wht_fused_decode_attention._kv_dbg_cnt,
                        sl0, bt0,
                        ','.join(str(x) for x in k_bytes[:8].tolist()),
                        nz_bytes, output.norm().item())

    # Try CUDA kernel (MQ_WHT_PYTHON=1 forces Python fallback)
    cuda_ok = False
    kernel = _load_wht_cuda_kernel() if not os.environ.get("MQ_WHT_PYTHON") else None
    if kernel is not None:
        try:
            # Pre-allocate int32 buffers (graph-safe: no .int() alloc)
            if 'bt_int' not in buf or buf['bt_int'].shape != block_table.shape:
                buf['bt_int'] = torch.empty_like(block_table, dtype=torch.int32)
                buf['sl_int'] = torch.empty_like(seq_lens, dtype=torch.int32)
            buf['bt_int'].copy_(block_table)
            buf['sl_int'].copy_(seq_lens)
            kernel.tq_wht_fused_decode_attention(
                q_wht,
                kv_cache,
                buf['bt_int'],
                buf['sl_int'],
                output,
                D, mse_bits, scale,
                s_block, s_kv, s_slot, s_head,
            )
            # JIT kernels run async — sync in eager mode only.
            # During graph capture: kernel is recorded, sync would break capture.
            if not torch.cuda.is_current_stream_capturing():
                torch.cuda.synchronize()
            cuda_ok = True
        except Exception as e:
            logger.warning("WHT CUDA decode call failed: %s", e)

    global _wht_decode_path_logged
    if not _wht_decode_path_logged:
        if cuda_ok:
            logger.info("WHT decode: using CUDA fused kernel")
        else:
            logger.warning("WHT decode: CUDA kernel unavailable, "
                           "using Python fallback (SLOW)")
        _wht_decode_path_logged = True

    if not cuda_ok:
        # Python fallback — correct but very slow
        logger.warning("WHT decode: falling back to Python (SLOW)")
        return None  # signal caller to use Python fallback

    # Debug: log output norm on first few calls
    if os.environ.get("MQ_DEBUG"):
        if not hasattr(wht_fused_decode_attention, '_call_cnt'):
            wht_fused_decode_attention._call_cnt = 0
        wht_fused_decode_attention._call_cnt += 1
        if wht_fused_decode_attention._call_cnt <= 3:
            logger.info("[WHT_CUDA] out_norm=%.2f q_norm=%.2f sl=%s",
                        output.float().norm().item(),
                        q.float().norm().item(),
                        seq_lens[:1].tolist())

    return output


def mq_fused_decode_attention(
    q: torch.Tensor,           # [B, Hq, D] bfloat16/float16
    kv_cache: torch.Tensor,    # [num_blocks, 2, block_size, Hkv, packed_size] uint8
    Pi: torch.Tensor,          # [D, D] float32 (TQ) or [n_groups, 8] (RQ rotors)
    S: torch.Tensor,           # [D, D] float32 (QJL projection matrix)
    centroids: torch.Tensor,   # [n_centroids] float32
    block_table: torch.Tensor, # [B, max_blocks] int32
    seq_lens: torch.Tensor,    # [B] int32
    scale: float,              # 1/sqrt(D)
    block_size: int,
    num_kv_heads: int,
    mse_bits: int,
    correction: float,         # sqrt(pi/2) / D
    is_rq: bool = False,       # True for RotorQuant (Clifford rotation)
) -> torch.Tensor:
    """Fused MQ decode: 2 pre-rotations + 1 CUDA kernel + 2 post-rotations.

    Works for both TQ (dense Pi matrix) and RQ (Clifford rotors).
    The CUDA kernel is identical — only pre/post rotation differs.
    """
    B, Hq, D = q.shape
    device = q.device
    n_centroids = centroids.shape[0]

    # Pre-allocated buffers (image-compatible + CUDA kernel fix)
    global _decode_buffers
    buf_key = (B, Hq, D, device)
    if buf_key not in _decode_buffers:
        _decode_buffers[buf_key] = {
            "q_rot": torch.empty(B * Hq, D, device=device, dtype=torch.float32),
            "q_proj": torch.empty(B * Hq, D, device=device, dtype=torch.float32),
            "centroid_acc": torch.zeros(B, Hq, D, device=device, dtype=torch.float32),
            "sign_acc": torch.zeros(B, Hq, D, device=device, dtype=torch.float32),
            "v_mse": torch.empty(B * Hq, D, device=device, dtype=torch.float32),
            "qjl_corr": torch.empty(B * Hq, D, device=device, dtype=torch.float32),
            "output": torch.empty(B, Hq, D, device=device, dtype=torch.float32),
        }
    buf = _decode_buffers[buf_key]

    q_flat = q.reshape(B * Hq, D).float()
    if is_rq:
        _rq_rotate_forward(q_flat, Pi, out=buf["q_rot"])
    else:
        torch.mm(q_flat, Pi.T, out=buf["q_rot"])
    q_rot = buf["q_rot"]
    torch.mm(q_flat, S.T, out=buf["q_proj"])
    q_proj = buf["q_proj"]

    q_rot_3d = q_rot.reshape(B, Hq, D)
    q_proj_3d = q_proj.reshape(B, Hq, D)

    centroid_acc = buf["centroid_acc"]
    sign_acc = buf["sign_acc"]
    centroid_acc.zero_()
    sign_acc.zero_()

    # KV cache strides
    s_block = kv_cache.stride(0)
    s_kv = kv_cache.stride(1)
    s_slot = kv_cache.stride(2)
    s_head = kv_cache.stride(3)

    # Try CUDA kernel (full V decompression — no post-GEMV needed)
    cuda_ok = False
    kernel = _load_cuda_kernel()
    if kernel is not None:
        try:
            output = buf["output"]
            kernel.tq_fused_decode_attention(
                q_rot_3d,
                q_proj_3d,
                kv_cache,
                Pi,             # [D, D] float32 rotation matrix
                S,              # [D, D] float32 QJL projection matrix
                centroids,
                block_table.int(),
                seq_lens.int(),
                output,         # kernel writes fully reconstructed V here
                D, mse_bits, n_centroids, scale,
                s_block, s_kv, s_slot, s_head,
            )
            cuda_ok = True
        except Exception as e:
            logger.warning("CUDA fused decode call failed: %s", e)

    global _decode_path_logged
    if not _decode_path_logged:
        if cuda_ok:
            logger.info("MQ decode: using CUDA fused kernel (full V decompression)")
        else:
            logger.warning("MQ decode: CUDA kernel unavailable, "
                           "using Triton fallback + post-loop GEMV")
        _decode_path_logged = True

    if cuda_ok:
        return output

    # Triton fallback with post-loop GEMV
    packed_size = kv_cache.shape[-1]
    mse_bytes = (D * mse_bits + 7) // 8
    qjl_bytes = (D + 7) // 8
    kv_group_size = Hq // num_kv_heads
    s_bt = block_table.stride(0)

    grid = (B, Hq)
    _mq_fused_decode_kernel_v2[grid](
        q_rot, q_proj,
        kv_cache,
        centroids,
        block_table, seq_lens,
        centroid_acc.reshape(B * Hq, D),
        sign_acc.reshape(B * Hq, D),
        s_block, s_kv, s_slot, s_head,
        s_bt,
        num_q_heads=Hq,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        KV_GROUP_SIZE=kv_group_size,
        PACKED_SIZE=packed_size,
        MSE_BITS=mse_bits,
        MSE_BYTES=mse_bytes,
        QJL_BYTES=qjl_bytes,
        N_CENTROIDS=n_centroids,
        correction_scale=correction,
        attn_scale=scale,
        num_warps=4,
        num_stages=1,
    )

    # Post-loop V reconstruction
    c_flat = centroid_acc.reshape(B * Hq, D)
    s_flat = sign_acc.reshape(B * Hq, D)
    v_mse = buf["v_mse"]
    qjl_corr = buf["qjl_corr"]
    if is_rq:
        _rq_rotate_inverse(c_flat, Pi, out=v_mse)
    else:
        torch.mm(c_flat, Pi, out=v_mse)
    torch.mm(s_flat, S, out=qjl_corr)
    v_mse.add_(qjl_corr, alpha=correction)

    return v_mse.reshape(B, Hq, D).clone()


# ============================================================
# RQ (RotorQuant) — Clifford rotation helpers + decode kernel
# ============================================================

_clifford_kernel = None
_clifford_kernel_tried = False


def _load_clifford_kernel():
    """JIT compile the fused Clifford sandwich kernel."""
    global _clifford_kernel, _clifford_kernel_tried
    if _clifford_kernel_tried:
        return _clifford_kernel
    _clifford_kernel_tried = True

    # os imported at module level
    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "kernels", "rotorquant"
    )
    if not os.path.exists(src_dir):
        src_dir = "/opt/rq_build"
    src_file = os.path.join(src_dir, "clifford_sandwich.cu")
    if not os.path.exists(src_file):
        return None

    try:
        from torch.utils.cpp_extension import load
        _clifford_kernel = load(
            name="clifford_sandwich",
            sources=[src_file],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
            ],
            verbose=False,
        )
        logger.info("Clifford sandwich CUDA kernel loaded")
        return _clifford_kernel
    except Exception as e:
        logger.warning("Clifford sandwich kernel FAILED: %s", e)
        return None


def _rq_rotate_forward(x, rotors, out=None):
    """RQ forward rotation: CUDA kernel or Python fallback."""
    D = x.shape[-1]
    kernel = _load_clifford_kernel()
    if kernel is not None:
        if out is None:
            out = torch.empty_like(x)
        kernel.clifford_sandwich_forward(x.float(), rotors.float(), out, D)
        return out
    from vllm.multiquant.rotorquant.clifford import (
        embed_vectors_as_multivectors, rotor_sandwich,
        extract_vectors_from_multivectors,
    )
    mv = embed_vectors_as_multivectors(x)
    mv_rot = rotor_sandwich(rotors, mv)
    return extract_vectors_from_multivectors(mv_rot, D)


def _rq_rotate_inverse(x, rotors, out=None):
    """RQ inverse rotation: CUDA kernel or Python fallback."""
    D = x.shape[-1]
    kernel = _load_clifford_kernel()
    if kernel is not None:
        if out is None:
            out = torch.empty_like(x)
        kernel.clifford_sandwich_inverse(x.float(), rotors.float(), out, D)
        return out
    from vllm.multiquant.rotorquant.clifford import (
        embed_vectors_as_multivectors, rotor_sandwich,
        extract_vectors_from_multivectors, reverse,
    )
    mv = embed_vectors_as_multivectors(x)
    rotor_rev = reverse(rotors)
    mv_recon = rotor_sandwich(rotor_rev, mv)
    return extract_vectors_from_multivectors(mv_recon, D)


_rq_decode_kernel = None
_rq_decode_kernel_tried = False


def _load_rq_decode_kernel():
    """JIT compile the RQ fused decode kernel (Clifford V decompression)."""
    global _rq_decode_kernel, _rq_decode_kernel_tried
    if _rq_decode_kernel_tried:
        return _rq_decode_kernel
    _rq_decode_kernel_tried = True

    # os imported at module level
    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "kernels", "rotorquant"
    )
    if not os.path.exists(src_dir):
        src_dir = "/opt/rq_build"
    src_file = os.path.join(src_dir, "rq_compressed_attention.cu")
    if not os.path.exists(src_file):
        return None

    try:
        from torch.utils.cpp_extension import load
        _rq_decode_kernel = load(
            name="rq_fused_decode",
            sources=[src_file],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
            ],
            verbose=False,
        )
        logger.info("RQ fused decode CUDA kernel loaded")
        return _rq_decode_kernel
    except Exception as e:
        logger.warning("RQ fused decode kernel FAILED: %s", e)
        return None


# ============================================================
# Block-Rotation Kernel loaders (tq3r/tq4r)
# ============================================================

_blockrot_pack_kernel = None
_blockrot_pack_kernel_tried = False

_blockrot_decode_kernel = None
_blockrot_decode_kernel_tried = False


def _load_blockrot_pack_kernel():
    """JIT compile the fused block-rotation pack kernel."""
    global _blockrot_pack_kernel, _blockrot_pack_kernel_tried
    if _blockrot_pack_kernel_tried:
        return _blockrot_pack_kernel
    _blockrot_pack_kernel_tried = True

    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "kernels", "turboquant"
    )
    if not os.path.exists(src_dir):
        src_dir = "/opt/tq_build"
    src_file = os.path.join(src_dir, "tq_blockrot_pack.cu")
    if not os.path.exists(src_file):
        logger.warning("Block-rot pack: source not found at %s", src_file)
        return None

    try:
        from torch.utils.cpp_extension import load
        _blockrot_pack_kernel = load(
            name="tq_blockrot_pack",
            sources=[src_file],
            extra_cuda_cflags=[
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
                "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("Block-rot fused pack CUDA kernel compiled and loaded")
        return _blockrot_pack_kernel
    except Exception as e:
        logger.warning("Block-rot fused pack CUDA kernel FAILED: %s", e)
        return None


def _load_blockrot_decode_kernel():
    """JIT compile the block-rotation fused decode kernel."""
    global _blockrot_decode_kernel, _blockrot_decode_kernel_tried
    if _blockrot_decode_kernel_tried:
        return _blockrot_decode_kernel
    _blockrot_decode_kernel_tried = True

    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "kernels", "turboquant"
    )
    if not os.path.exists(src_dir):
        src_dir = "/opt/tq_build"
    src_file = os.path.join(src_dir, "tq_blockrot_decode.cu")
    if not os.path.exists(src_file):
        logger.warning("Block-rot decode: source not found at %s", src_file)
        return None

    try:
        from torch.utils.cpp_extension import load
        _blockrot_decode_kernel = load(
            name="tq_blockrot_decode",
            sources=[src_file],
            extra_cuda_cflags=[
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
                "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("Block-rot fused decode CUDA kernel compiled and loaded")
        return _blockrot_decode_kernel
    except Exception as e:
        logger.warning("Block-rot fused decode CUDA kernel FAILED: %s", e)
        return None


# ============================================================
# Split-KV WHT Decode Kernel loader
# ============================================================

_splitkv_decode_kernel = None
_splitkv_decode_kernel_tried = False


def _load_splitkv_decode_kernel():
    """JIT compile the split-KV WHT decode kernel."""
    global _splitkv_decode_kernel, _splitkv_decode_kernel_tried
    if _splitkv_decode_kernel_tried:
        return _splitkv_decode_kernel
    _splitkv_decode_kernel_tried = True

    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "kernels", "turboquant"
    )
    if not os.path.exists(src_dir):
        src_dir = "/opt/tq_build"
    src_file = os.path.join(src_dir, "tq_wht_decode_splitkv.cu")
    if not os.path.exists(src_file):
        logger.warning("Split-KV decode: source not found at %s", src_file)
        return None

    try:
        from torch.utils.cpp_extension import load
        _splitkv_decode_kernel = load(
            name="tq_wht_splitkv_decode",
            sources=[src_file],
            extra_cuda_cflags=[
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
                "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("Split-KV WHT decode CUDA kernel compiled and loaded")
        return _splitkv_decode_kernel
    except Exception as e:
        logger.warning("Split-KV WHT decode CUDA kernel FAILED: %s", e)
        return None
