# SPDX-License-Identifier: Apache-2.0
"""XFP pack utility — Lloyd codebook + word-aligned sub-byte packing.

One call per weight matrix at load time:
    packed, codebook, stats = xfp_pack(W, bits, also_score_widths=(2,3,4))

Per output channel (row of W), a 2^bits-entry codebook is fit via Lloyd
iteration on the 1-D weight distribution. Each weight is then replaced by the
index of its nearest codebook entry. Indices are bit-packed word-aligned into
uint32 words so the decode kernel can extract them with shift+mask and no
cross-word reads.

Packing per bits (all word-aligned on uint32):

    bits | values per uint32 | K_packed       | reserve bits per word
    -----+-------------------+----------------+----------------------
    2    | 16                | ceil(K / 16)   | 0
    3    | 10                | ceil(K / 10)   | 2
    4    |  8                | ceil(K / 8)    | 0

Output shape: packed [K_packed, N_out] uint32 (K-major, matches the layout
convention of mq_gemm_int2/int3 kernels).

Statistics:
    Per-layer XFPPackStats captures weight distribution moments, outlier
    ratios at k=3σ and k=4σ, reconstruction MSE/cos_sim at the chosen bits,
    and (when also_score_widths is set) MSE at alternative bit widths so
    callers can recommend a per-layer optimal bit width for future auto-sizing.

v1 scope: no outlier extraction (bulk-only codebook). Outlier-split path is v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class XFPPackStats:
    """Per-layer distribution + reconstruction stats from Lloyd packing."""

    bits: int
    shape: tuple  # (N_out, K)

    # Weight distribution moments (raw input)
    w_mean: float
    w_std: float
    w_abs_max: float

    # Outlier fraction estimates (|w - mean| > k * std)
    outlier_ratio_k3: float
    outlier_ratio_k4: float

    # Reconstruction quality at the chosen `bits`
    mse: float
    rmse_rel: float  # sqrt(mse) / w_std
    max_abs_err: float
    cos_sim: float

    # Outlier split (0 when outlier extraction is disabled)
    outlier_count: int = 0          # number of (row, col) pairs extracted
    outlier_sigma: float = 0.0      # threshold used (0 = none)
    outlier_fraction: float = 0.0   # count / numel

    # Candidate scoring — MSE at other bit widths (empty unless requested)
    mse_per_bits: dict[int, float] = field(default_factory=dict)

    # Auto-size recommendation derived from mse_per_bits
    # (falls back to `bits` when mse_per_bits is empty)
    recommended_bits: int = 0
    recommended_gap: float = 1.0  # mse[chosen] / mse[recommended], ≥1.0


# ─── Lloyd iteration ──────────────────────────────────────────────────


def _lloyd_per_channel(
    W: torch.Tensor,  # [N_out, K] fp32
    n_centroids: int,
    n_iters: int,
    row_chunk: int = 4096,
) -> torch.Tensor:
    """Return codebook [N_out, n_centroids] fp32.

    Lloyd's algorithm (1-D k-means) per row of W. Min-max linspace init,
    then alternating assignment/update. Large row_chunk (default 4096) so
    typical sub-64k-row matrices run in a single chunk — the outer Python
    loop then has one iteration per Lloyd pass, not dozens.
    """
    device = W.device
    N_out, K = W.shape

    # Min-max linspace initialization per row — O(N_out * K), avoids the
    # O(N_out * K * log K) cost of torch.quantile. On a smooth distribution
    # Lloyd converges from this within ~15 iterations; on heavy-tailed
    # distributions the first few iterations already pull centroids toward
    # the CDF-uniform optimum.
    w_min = W.min(dim=1, keepdim=True).values  # [N_out, 1]
    w_max = W.max(dim=1, keepdim=True).values  # [N_out, 1]
    t = torch.linspace(
        0.0, 1.0, n_centroids, device=device, dtype=torch.float32
    )  # [n_centroids]
    codebook = w_min + (w_max - w_min) * t.unsqueeze(0)  # [N_out, n_centroids]

    # Break ties so identical rows don't collapse all centroids
    jitter = torch.linspace(
        -1e-6, 1e-6, n_centroids, device=device, dtype=torch.float32,
    )
    codebook = codebook + jitter.unsqueeze(0)

    for _ in range(n_iters):
        new_codebook = torch.empty_like(codebook)
        for r0 in range(0, N_out, row_chunk):
            r1 = min(r0 + row_chunk, N_out)
            W_chunk = W[r0:r1]  # [C, K]
            cb_chunk = codebook[r0:r1]  # [C, n_centroids]

            # Nearest assignment — distance in [C, K, n_centroids]
            dist = (W_chunk.unsqueeze(-1) - cb_chunk.unsqueeze(1)).abs()
            idx = dist.argmin(-1)  # [C, K]

            # Mean-of-cluster update
            # one-hot scatter-add accumulation
            C = r1 - r0
            sums = torch.zeros(
                C, n_centroids, dtype=torch.float32, device=device
            )
            counts = torch.zeros(
                C, n_centroids, dtype=torch.float32, device=device
            )
            sums.scatter_add_(1, idx, W_chunk)
            counts.scatter_add_(1, idx, torch.ones_like(W_chunk))

            # Empty clusters → keep old centroid (count=1 with old centroid value)
            empty = counts == 0
            if empty.any():
                counts = counts + empty.float()
                sums = sums + torch.where(
                    empty, cb_chunk, torch.zeros_like(cb_chunk)
                )

            new_codebook[r0:r1] = sums / counts

        # Re-sort each row's codebook so indices are monotone
        # (helps logging and some decode optimizations)
        codebook = torch.sort(new_codebook, dim=1).values

    return codebook


def _assign_indices(
    W: torch.Tensor,  # [N_out, K] fp32
    codebook: torch.Tensor,  # [N_out, 2^bits] fp32
    row_chunk: int = 128,
) -> torch.Tensor:
    """Return argmin indices [N_out, K] int64."""
    N_out, K = W.shape
    idx = torch.empty(N_out, K, dtype=torch.int64, device=W.device)
    for r0 in range(0, N_out, row_chunk):
        r1 = min(r0 + row_chunk, N_out)
        dist = (W[r0:r1].unsqueeze(-1) - codebook[r0:r1].unsqueeze(1)).abs()
        idx[r0:r1] = dist.argmin(-1)
    return idx


# ─── Word-aligned sub-byte packing ────────────────────────────────────


def _pack_indices(
    idx: torch.Tensor,  # [N_out, K] int, values in [0, 2^bits)
    bits: int,
) -> torch.Tensor:
    """Pack into [K_packed, N_out] uint32.

    Word layout (bits → values/word):
        bits=2: 16 values, shift 2*i, no reserve
        bits=3: 10 values, shift 3*i, 2 reserve bits (bit 30-31 unused)
        bits=4:  8 values, shift 4*i, no reserve
    """
    N_out, K = idx.shape
    vals_per_word = {2: 16, 3: 10, 4: 8}[bits]
    K_packed = (K + vals_per_word - 1) // vals_per_word

    # Pad K so it's a multiple of vals_per_word
    if K < K_packed * vals_per_word:
        pad = K_packed * vals_per_word - K
        idx = F.pad(idx, (0, pad), value=0)

    # idx is [N_out, K_packed * vals_per_word]; view as [N_out, K_packed, vals_per_word]
    idx = idx.view(N_out, K_packed, vals_per_word).to(torch.int64)

    # Build per-word packed value via shifted OR
    packed = torch.zeros(
        N_out, K_packed, dtype=torch.int64, device=idx.device
    )
    for slot in range(vals_per_word):
        packed |= (idx[:, :, slot] & ((1 << bits) - 1)) << (slot * bits)

    # Transpose to K-major layout [K_packed, N_out] and cast to int32
    # (int32 is the closest "uint32" torch offers; bit pattern is identical
    # as long as no arithmetic interprets it as signed)
    return packed.t().contiguous().to(torch.int32)


# ─── Reference dequant (used by tests and by the Python reconstruction
#     path that feeds stats.mse / stats.cos_sim) ──────────────────────


def dequant_xfp(
    packed: torch.Tensor,  # [K_packed, N_out] int32
    codebook: torch.Tensor,  # [N_out, 2^bits] any float dtype
    K: int,
    bits: int,
) -> torch.Tensor:
    """Reconstruct W [N_out, K] from packed + codebook.

    Reference implementation — torch ops only, no CUDA kernel. Used by
    xfp_pack() itself (to compute MSE) and by tests.
    """
    vals_per_word = {2: 16, 3: 10, 4: 8}[bits]
    mask = (1 << bits) - 1
    K_packed = packed.shape[0]
    N_out = packed.shape[1]
    assert K_packed * vals_per_word >= K

    packed_nk = packed.t().to(torch.int64)  # [N_out, K_packed]

    # Unpack to [N_out, K_padded]
    unpacked = torch.zeros(
        N_out, K_packed * vals_per_word,
        dtype=torch.int64, device=packed.device,
    )
    for slot in range(vals_per_word):
        unpacked[:, slot::vals_per_word] = (packed_nk >> (slot * bits)) & mask

    idx = unpacked[:, :K]  # drop padding

    # Per-row gather: codebook is [N_out, 2^bits], index per (n, k)
    return torch.gather(codebook, 1, idx)


# ─── Stats ──────────────────────────────────────────────────────────


def _distribution_stats(W: torch.Tensor) -> tuple[float, float, float, float, float]:
    """Return (mean, std, abs_max, outlier_ratio_k3, outlier_ratio_k4)."""
    mean = W.mean().item()
    std = W.std().item()
    abs_max = W.abs().max().item()
    centered = (W - mean).abs()
    total = float(W.numel())
    outlier_k3 = float((centered > 3.0 * std).sum().item()) / total
    outlier_k4 = float((centered > 4.0 * std).sum().item()) / total
    return mean, std, abs_max, outlier_k3, outlier_k4


def _reconstruction_stats(
    W: torch.Tensor, W_rec: torch.Tensor
) -> tuple[float, float, float]:
    """Return (mse, max_abs_err, cos_sim)."""
    diff = W - W_rec
    mse = (diff * diff).mean().item()
    max_abs_err = diff.abs().max().item()
    cos = F.cosine_similarity(
        W.reshape(-1).unsqueeze(0), W_rec.reshape(-1).unsqueeze(0), dim=1
    ).item()
    return mse, max_abs_err, cos


def _score_mse_only(W: torch.Tensor, bits: int, lloyd_iters: int) -> float:
    """Fit a codebook at `bits` and return reconstruction MSE (no packing).

    Used to populate stats.mse_per_bits for the auto-size signal.
    """
    codebook = _lloyd_per_channel(W, 1 << bits, lloyd_iters)
    idx = _assign_indices(W, codebook)
    W_rec = torch.gather(codebook, 1, idx)
    return ((W - W_rec) * (W - W_rec)).mean().item()


# ─── Weight repack for coalesced warp reads ────────────────────────


def xfp_repack(packed: torch.Tensor, warp_size: int = 32) -> torch.Tensor:
    """Repack [K_packed, N] → [K_groups * N * warp_size] int32.

    v4opt kernel's access pattern: warp n, lane i reads
    B_packed[kw * N + n] where kw = lane, lane+32, lane+64...

    Without repack: consecutive lane reads (lane 0..31) at the same kw
    are at addresses kw*N+n — all in one cache line IF different warps
    happen to sync. In practice warps drift → poor L2 utilization.

    After repack: the K dimension is interleaved over warp_size so that
    one warp's consecutive lane reads at the same kw_group form a
    contiguous 128-byte block:

      repacked[kw_group * N * WS + n * WS + lane]

    All 32 lane reads are consecutive → 1 cache line → 100% utilization.

    The tensor is returned as a flat [K_groups * N * warp_size] int32
    to avoid 3D stride complications in the kernel. The kernel computes:
      idx = kw_group * (N * WS) + n * WS + lane
    where kw_group = (kw_original / WS), and lane = kw_original % WS.
    """
    K_packed, N = packed.shape
    K_groups = (K_packed + warp_size - 1) // warp_size

    # Pad K_packed to a multiple of warp_size
    if K_packed % warp_size != 0:
        pad = warp_size - K_packed % warp_size
        packed = F.pad(packed, (0, 0, 0, pad), value=0)

    # [K_groups, warp_size, N] → [K_groups, N, warp_size] → flatten
    repacked = (
        packed.reshape(K_groups, warp_size, N)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(-1)
    )
    return repacked


# ─── Auto bit-width selection ──────────────────────────────────────


def xfp_auto_select(
    W: torch.Tensor,
    candidates: tuple[int, ...] = (2, 3, 4),
    min_cos: float = 0.98,
    lloyd_iters: int = 20,
    outlier_sigma: Optional[float] = 4.0,
    outlier_max_fraction: float = 0.02,
) -> int:
    """Pick the lowest bit width where reconstruction meets the cos gate.

    Runs Lloyd at each candidate width, computes per-channel cosine
    similarity, and returns the first (lowest) bits where the median
    per-channel cos >= min_cos.

    The cos gate is the sole discriminator. MSE ratio was tested and
    rejected: on real MoE models, XFP2 has ~12× higher MSE than XFP4
    but identical math accuracy (the error is spread uniformly and
    doesn't concentrate in model-critical channels). Cos similarity
    captures this: it measures directional preservation per channel,
    which is what matters for downstream quality.

    Falls back to max(candidates) if no lower width qualifies.

    Returns:
        Chosen bits (one of the candidates).
    """
    if W.dim() != 2:
        raise ValueError(f"xfp_auto_select: W must be 2D, got {W.dim()}D")

    W = W.to(torch.float32)
    candidates = tuple(sorted(candidates))
    best_bits = candidates[-1]  # fallback

    # Outlier split (shared across all candidates — same mask)
    if outlier_sigma is not None and outlier_sigma > 0:
        mu = W.mean()
        sigma = W.std()
        threshold = float(outlier_sigma) * sigma
        mask = (W - mu).abs() > threshold
        total = W.numel()
        max_allowed = int(outlier_max_fraction * total)
        nnz = int(mask.sum().item())
        if nnz > max_allowed and max_allowed > 0:
            flat_abs = (W - mu).abs().reshape(-1)
            _, top_idx = torch.topk(flat_abs, max_allowed, largest=True, sorted=False)
            mask = torch.zeros_like(flat_abs, dtype=torch.bool)
            mask[top_idx] = True
            mask = mask.reshape_as(W)
        if nnz > 0:
            W_bulk = W.clone()
            W_bulk[mask] = mu
        else:
            W_bulk = W
            mask = None
    else:
        W_bulk = W
        mask = None

    # Test each candidate from lowest to highest
    for bits in candidates:
        if bits == best_bits:
            # Highest candidate always qualifies as fallback
            return bits

        n_centroids = 1 << bits
        cb = _lloyd_per_channel(W_bulk, n_centroids, lloyd_iters)
        idx = _assign_indices(W_bulk, cb)
        rec = torch.gather(cb, 1, idx)
        if mask is not None:
            flat_r = rec.reshape(-1).clone()
            flat_r[mask.reshape(-1)] = W.reshape(-1)[mask.reshape(-1)]
            rec = flat_r.reshape_as(W)

        # Per-channel cos similarity — sole quality gate
        cos_per_ch = F.cosine_similarity(W, rec, dim=1)  # [N_out]
        median_cos = float(cos_per_ch.median().item())

        if median_cos >= min_cos:
            return bits

    return best_bits


# ─── Public entry point ────────────────────────────────────────────


def xfp_pack(
    W: torch.Tensor,
    bits: int,
    lloyd_iters: int = 20,
    also_score_widths: tuple[int, ...] = (),
    outlier_sigma: Optional[float] = None,
    outlier_max_fraction: float = 0.02,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    XFPPackStats,
]:
    """Pack a weight matrix via per-channel Lloyd codebook + bit-packed indices.

    Args:
        W: [N_out, K] weight tensor. Caller converts from bf16/fp16 to fp32.
        bits: target bit width; one of {2, 3, 4}. Determines codebook size
            (2^bits entries) and pack layout per §3.3 of XFP.PAPER.md.
        lloyd_iters: number of Lloyd refinement iterations (20–50 typical).
        also_score_widths: additional bit widths to fit a codebook for and
            score only for MSE, without packing. Used to populate
            stats.mse_per_bits as a per-layer auto-size signal.
        outlier_sigma: if set, extract weights with |w - mean| > sigma * std
            as sparse residuals BEFORE fitting the codebook. Paper §4 Step 2.
            Typical values 3.0–4.0. None disables outlier extraction.
            Based on GLM-4.7-Flash weight inspection (tests/xfp/inspect_
            distribution.py), 4.0 is the sweet spot for MoE models: it
            catches the 40σ attention outliers (kv_b_proj, q_b_proj) while
            marking only ~0.01–0.8 % of weights, comfortably below the
            15 % sparse-path threshold of Paper §4 Step 1.
        outlier_max_fraction: safety cap. If the k-sigma threshold would
            mark more than this fraction of weights as outliers (e.g. for
            a broad-spectrum distribution where the split is counter-
            productive), keep only the top-by-magnitude
            `outlier_max_fraction * numel` weights and reclassify the rest
            as bulk. Default 0.02 (= 2 %). Paper §4 says ratios above ~30 %
            should drop sparse extraction entirely — we use a tighter
            cap because v1 has no inline sparse kernel yet and larger
            outlier sets hurt the apply-path throughput.
        seed: random seed (reserved for future stochastic variants).

    Returns:
        packed: [K_packed, N_out] int32 (bit-identical to uint32)
        codebook: [N_out, 2^bits] fp16
        outlier_indices: [n_outliers] int64 flat index (row*K + col), or None
        outlier_values: [n_outliers] fp16 original weight values, or None
        stats: XFPPackStats

    When `outlier_sigma` is None the two outlier tensors are None and the
    returned pipeline matches the v1 bulk-only encoder exactly.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"xfp_pack: unsupported bits={bits}, must be in {{2,3,4}}")
    if W.dim() != 2:
        raise ValueError(f"xfp_pack: W must be 2D [N_out, K], got shape {tuple(W.shape)}")

    del seed  # reserved

    W = W.to(torch.float32)
    N_out, K = W.shape
    n_centroids = 1 << bits

    # Outlier split (Paper §4 Step 2) — optional. Runs BEFORE Lloyd so the
    # codebook fits the cleaned bulk distribution.
    outlier_indices: Optional[torch.Tensor] = None
    outlier_values: Optional[torch.Tensor] = None
    outlier_count = 0
    outlier_fraction = 0.0
    if outlier_sigma is not None and outlier_sigma > 0:
        mu = W.mean()
        sigma = W.std()
        total_numel = W.numel()
        centered_abs = (W - mu).abs()
        threshold = float(outlier_sigma) * sigma
        mask = centered_abs > threshold  # [N_out, K] bool
        nnz = int(mask.sum().item())
        max_allowed = int(outlier_max_fraction * total_numel)

        if nnz > max_allowed and max_allowed > 0:
            # Safety cap: keep only the top-by-magnitude weights. Anything
            # beyond outlier_max_fraction is clamped back into the bulk.
            # Uses a top-k on the flattened abs-centered tensor.
            flat_abs = centered_abs.reshape(-1)
            _, top_flat_idx = torch.topk(
                flat_abs, max_allowed, largest=True, sorted=False
            )
            mask = torch.zeros_like(flat_abs, dtype=torch.bool)
            mask[top_flat_idx] = True
            mask = mask.reshape_as(W)
            nnz = max_allowed

        if nnz > 0:
            # Flat indices so the apply path can split (row, col) cheaply.
            flat_mask = mask.reshape(-1)
            outlier_indices = flat_mask.nonzero(as_tuple=False).squeeze(1).to(torch.int64)
            outlier_values = W.reshape(-1)[outlier_indices].to(torch.float16)
            # Replace outlier positions with the layer mean so Lloyd fits
            # the bulk without being pulled by extreme values. Using mean
            # (not zero) avoids creating a new cluster at zero that would
            # consume a codebook entry.
            W_bulk = W.clone()
            W_bulk[mask] = mu
            outlier_count = nnz
            outlier_fraction = float(nnz) / float(total_numel)
        else:
            W_bulk = W
    else:
        W_bulk = W

    # 1. Lloyd codebook on the (possibly cleaned) bulk
    codebook_fp32 = _lloyd_per_channel(W_bulk, n_centroids, lloyd_iters)

    # 2. Assign indices (still on the bulk — outlier positions will be
    #    reconstructed via the scatter-add path at apply time, so their
    #    codebook index is irrelevant)
    idx = _assign_indices(W_bulk, codebook_fp32)

    # 3. Reconstruction (for MSE / cos sim) — include outlier correction
    #    so stats reflect the full XFP reconstruction, not just the bulk.
    W_rec_bulk = torch.gather(codebook_fp32, 1, idx)
    if outlier_indices is not None and outlier_values is not None:
        W_rec = W_rec_bulk.clone()
        flat_rec = W_rec.reshape(-1)
        flat_rec[outlier_indices] = outlier_values.to(torch.float32)
        W_rec = flat_rec.reshape(N_out, K)
    else:
        W_rec = W_rec_bulk

    # 4. Pack indices to word-aligned uint32
    packed = _pack_indices(idx, bits)

    # 5. Stats — distribution stats use the ORIGINAL W (incl. outliers)
    w_mean, w_std, w_abs_max, k3, k4 = _distribution_stats(W)
    mse, max_abs_err, cos_sim = _reconstruction_stats(W, W_rec)
    rmse_rel = (mse ** 0.5) / max(w_std, 1e-12)

    mse_per_bits: dict[int, float] = {bits: mse}
    for b in also_score_widths:
        if b == bits or b in mse_per_bits:
            continue
        if b not in (2, 3, 4):
            continue
        mse_per_bits[b] = _score_mse_only(W_bulk, b, lloyd_iters)

    # Recommendation: lowest MSE among scored widths, ties → prefer lower bits
    if mse_per_bits:
        recommended_bits = min(
            mse_per_bits.items(), key=lambda kv: (kv[1], kv[0])
        )[0]
        recommended_gap = mse_per_bits[bits] / max(
            mse_per_bits[recommended_bits], 1e-30
        )
    else:
        recommended_bits = bits
        recommended_gap = 1.0

    stats = XFPPackStats(
        bits=bits,
        shape=(N_out, K),
        w_mean=w_mean,
        w_std=w_std,
        w_abs_max=w_abs_max,
        outlier_ratio_k3=k3,
        outlier_ratio_k4=k4,
        mse=mse,
        rmse_rel=rmse_rel,
        max_abs_err=max_abs_err,
        cos_sim=cos_sim,
        outlier_count=outlier_count,
        outlier_sigma=float(outlier_sigma or 0.0),
        outlier_fraction=outlier_fraction,
        mse_per_bits=mse_per_bits,
        recommended_bits=recommended_bits,
        recommended_gap=recommended_gap,
    )

    return (
        packed,
        codebook_fp32.to(torch.float16),
        outlier_indices,
        outlier_values,
        stats,
    )
