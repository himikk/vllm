# SPDX-License-Identifier: Apache-2.0
"""MultiQuant fused MLA decode — Triton kernel.

Key optimization: pre-rotate query (q_rot = q @ Pi), then score
against raw centroids. Eliminates per-token D×D rotation entirely.

Math: q · (Pi^T @ centroids[idx]) = (q @ Pi) · centroids[idx]

Per-token cost: just gather + dot-product (no matrix multiply).
"""

import math

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mq_mla_decode_kernel(
    Q_rot,                  # [B, num_heads, D] — pre-rotated query
    KV_Cache,               # [num_pages, page_size, packed_size] uint8
    Centroids,              # [n_centroids] float32
    S_proj,                 # [D] float32 — pre-computed q_rot @ S for QJL
    Block_table,            # [B, max_blocks] int32
    Seq_lens,               # [B] int32
    Out,                    # [B, num_heads, kv_lora_rank]
    Lse,                    # [B, num_heads] float32
    sm_scale,
    stride_qb, stride_qh,
    stride_cache_page, stride_cache_slot,
    stride_bt_b,
    stride_ob, stride_oh,
    D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    seq_len = tl.load(Seq_lens + cur_batch)
    if seq_len <= 0:
        return

    # Load pre-rotated query
    offs_d = tl.arange(0, D)
    q_rot = tl.load(
        Q_rot + cur_batch * stride_qb + cur_head * stride_qh + offs_d,
        mask=offs_d < D, other=0.0,
    )

    COORDS_PER_BYTE: tl.constexpr = 8 // MSE_BITS
    MASK: tl.constexpr = (1 << MSE_BITS) - 1
    MSE_BYTES: tl.constexpr = (D * MSE_BITS + 7) // 8
    QJL_BYTES: tl.constexpr = (D + 7) // 8
    NORM_OFFSET: tl.constexpr = MSE_BYTES + QJL_BYTES

    # Load centroids to registers
    offs_c = tl.arange(0, N_CENTROIDS)
    centroids = tl.load(Centroids + offs_c, mask=offs_c < N_CENTROIDS)

    # Online softmax state
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([KV_LORA_RANK], dtype=tl.float32)
    offs_v = tl.arange(0, KV_LORA_RANK)

    for token_idx in range(0, seq_len):
        # Page table lookup
        page_idx = tl.load(
            Block_table + cur_batch * stride_bt_b + token_idx // PAGE_SIZE
        )
        slot = token_idx % PAGE_SIZE
        base = page_idx * stride_cache_page + slot * stride_cache_slot

        # ── Unpack MSE indices + compute score ──────────────────
        # score = q_rot · centroids[idx] (no rotation needed!)
        score = tl.zeros([1], dtype=tl.float32)

        for byte_i in range(MSE_BYTES):
            packed_byte = tl.load(KV_Cache + base + byte_i).to(tl.int32)
            for k in range(COORDS_PER_BYTE):
                j = byte_i * COORDS_PER_BYTE + k
                if j < D:
                    idx = (packed_byte >> (k * MSE_BITS)) & MASK
                    # Centroid value for this coordinate
                    c_val = tl.sum(tl.where(offs_c == idx, centroids, 0.0))
                    # Accumulate dot product: q_rot[j] * centroid
                    q_j = tl.sum(tl.where(offs_d == j, q_rot, 0.0))
                    score += q_j * c_val

        score = score * sm_scale

        # ── Online softmax ──────────────────────────────────────
        m_new = tl.maximum(m_prev, score)
        exp_prev = tl.exp(m_prev - m_new)
        exp_new = tl.exp(score - m_new)
        l_new = l_prev * exp_prev + exp_new

        # ── Value: unpack first kv_lora_rank coordinates ────────
        # (Value is stored in the same packed format as Key)
        v_val = tl.zeros([KV_LORA_RANK], dtype=tl.float32)
        for byte_i in range(MSE_BYTES):
            packed_byte = tl.load(KV_Cache + base + byte_i).to(tl.int32)
            for k in range(COORDS_PER_BYTE):
                j = byte_i * COORDS_PER_BYTE + k
                if j < KV_LORA_RANK:
                    idx = (packed_byte >> (k * MSE_BITS)) & MASK
                    c_val = tl.sum(tl.where(offs_c == idx, centroids, 0.0))
                    v_val = tl.where(offs_v == j, c_val, v_val)

        # Weighted accumulation
        scale_prev = l_prev * exp_prev / l_new
        scale_new = exp_new / l_new
        acc = acc * scale_prev + v_val * scale_new

        m_prev = m_new
        l_prev = l_new

    # Write output
    offs_out = cur_batch * stride_ob + cur_head * stride_oh + offs_v
    tl.store(Out + offs_out, acc.to(Out.dtype.element_ty),
             mask=offs_v < KV_LORA_RANK)
    tl.store(Lse + cur_batch * tl.num_programs(1) + cur_head,
             m_prev + tl.log(l_prev))


def mq_mla_decode_attention(
    q: torch.Tensor,            # [B, num_heads, qk_dim]
    kv_cache: torch.Tensor,     # [num_pages, page_size, packed_size] uint8
    centroids: torch.Tensor,    # [n_centroids] float32
    Pi: torch.Tensor,           # [D, D] float32
    S: torch.Tensor,            # [D, D] float32
    block_table: torch.Tensor,  # [B, max_blocks]
    seq_lens: torch.Tensor,     # [B]
    scale: float,
    page_size: int,
    head_size: int,
    kv_lora_rank: int,
    mse_bits: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MQ-MLA decode attention.

    Pre-rotates query: q_rot = q @ Pi (one matmul per batch).
    Then scores against raw centroids: score = q_rot · centroids[idx].
    No per-token D×D rotation.
    """
    B, num_heads, qk_dim = q.shape
    n_centroids = centroids.shape[0]

    # Pre-rotate query: eliminates per-token rotation
    # q_rot[b,h,:] = q[b,h,:] @ Pi
    q_rot = torch.bmm(
        q.reshape(B * num_heads, 1, qk_dim).float(),
        Pi.unsqueeze(0).expand(B * num_heads, -1, -1),
    ).reshape(B, num_heads, head_size).contiguous()

    o = torch.zeros(B, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.zeros(B, num_heads, dtype=torch.float32, device=q.device)

    # Triton grid: one program per (batch, head)
    grid = (B, num_heads)

    _mq_mla_decode_kernel[grid](
        q_rot.float(), kv_cache, centroids.float(),
        torch.zeros(head_size, device=q.device),  # S_proj placeholder
        block_table, seq_lens, o, lse,
        scale,
        q_rot.stride(0), q_rot.stride(1),
        kv_cache.stride(0), kv_cache.stride(1),
        block_table.stride(0),
        o.stride(0), o.stride(1),
        D=head_size,
        MSE_BITS=mse_bits,
        KV_LORA_RANK=kv_lora_rank,
        PAGE_SIZE=page_size,
        BLOCK_N=1,
        N_CENTROIDS=n_centroids,
    )

    return o, lse
