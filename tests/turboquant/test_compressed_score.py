#!/usr/bin/env python3
"""Standalone test: TQ compressed attention score kernel.

Validates that attention scores from packed TQ cache match
scores from decompressed BF16 keys.

Run inside container:
  python3 /opt/vllm-riy/tests/turboquant/test_compressed_score.py
"""
import math
import sys
import time

import torch
import torch.nn.functional as F


def build_kernels():
    """Build both TQ kernels via JIT."""
    from torch.utils.cpp_extension import load

    # Score kernel (reads compressed cache directly)
    score_ext = load(
        name="tq_score",
        sources=[
            "/opt/tq_build/tq_compressed_score_ext.cu",
            "/opt/tq_build/tq_compressed_score.cu",
        ],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                           "--use_fast_math",
                           "-gencode=arch=compute_121,code=sm_121",
                           "-diag-suppress=177,3288"],
        extra_include_paths=["/opt/tq_build"],
        verbose=False,
    )

    # Round-trip kernel (for packing keys)
    rt_ext = load(
        name="tq_serve",
        sources=["/opt/tq_build/tq_ext.cu", "/opt/tq_build/tq_round_trip.cu"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                           "--use_fast_math",
                           "-gencode=arch=compute_121,code=sm_121",
                           "-diag-suppress=177,3288"],
        extra_include_paths=["/opt/tq_build"],
        verbose=False,
    )
    return score_ext, rt_ext


def pack_key_tq3(key_vec, Pi, S, centroids, D):
    """Pack a single key vector into TQ3 format (PyTorch)."""
    mse_bits = 2
    x = key_vec.float()
    vn = x.norm()
    xh = x / (vn + 1e-8)
    rot = xh @ Pi.T
    idx = (rot.unsqueeze(-1) - centroids).abs().argmin(dim=-1).to(torch.uint8)
    yh = centroids[idx.long()]
    xm = yh @ Pi
    r = xh - xm
    rn = r.norm()
    signs = (r @ S.T >= 0).to(torch.uint8)

    # Pack MSE indices (2-bit, 4 per byte)
    mse_bytes = (D * mse_bits + 7) // 8
    packed_mse = torch.zeros(mse_bytes, dtype=torch.uint8, device=key_vec.device)
    for j in range(0, D, 4):
        val = torch.zeros(1, dtype=torch.uint8, device=key_vec.device)
        for k in range(min(4, D - j)):
            val |= (idx[j + k] & 0x3) << (k * 2)
        packed_mse[j // 4] = val

    # Pack signs (1-bit, 8 per byte)
    qjl_bytes = (D + 7) // 8
    packed_signs = torch.zeros(qjl_bytes, dtype=torch.uint8, device=key_vec.device)
    for j in range(0, D, 8):
        val = torch.zeros(1, dtype=torch.uint8, device=key_vec.device)
        for k in range(min(8, D - j)):
            val |= (signs[j + k] & 1) << k
        packed_signs[j // 8] = val

    # Norms
    vn_bytes = vn.half().reshape(1).view(torch.uint8)
    rn_bytes = rn.half().reshape(1).view(torch.uint8)

    return torch.cat([packed_mse, packed_signs, vn_bytes, rn_bytes])


def test_compressed_score():
    """Test: CUDA compressed score vs PyTorch decompressed score."""
    print("=" * 60)
    print("TQ Compressed Attention Score Kernel Test")
    print("=" * 60)

    device = "cuda"
    D = 64  # GLM-4.7 head_dim
    NC = 4  # TQ3 centroids
    mse_bits = 2
    num_q_heads = 20
    num_kv_heads = 20
    seq_len = 128
    block_size = 16
    num_blocks = (seq_len + block_size - 1) // block_size
    packed_size = (D * mse_bits + 7) // 8 + (D + 7) // 8 + 4  # 28 for D=64

    torch.manual_seed(42)

    # Generate matrices
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    Pi = torch.randn(D, D, generator=gen, dtype=torch.float32, device="cpu")
    Pi, _ = torch.linalg.qr(Pi)
    Pi = Pi.contiguous().to(device)
    gen.manual_seed(43)
    S = torch.randn(D, D, generator=gen, dtype=torch.float32).to(device)

    sigma = 1.0 / math.sqrt(D)
    centroids = torch.tensor([-1.51*sigma, -0.453*sigma, 0.453*sigma, 1.51*sigma],
                              dtype=torch.float32, device=device)

    # Generate random keys and pack them
    keys = torch.randn(seq_len, D, device=device, dtype=torch.bfloat16)
    k_cache = torch.zeros(num_blocks, block_size, 1, packed_size,
                          dtype=torch.uint8, device=device)

    for t in range(seq_len):
        bi, bo = t // block_size, t % block_size
        packed = pack_key_tq3(keys[t], Pi, S, centroids, D)
        k_cache[bi, bo, 0, :len(packed)] = packed

    # Generate query
    query = torch.randn(1, 1, D, device=device, dtype=torch.bfloat16)

    # Precompute q_rot = query @ Pi^T, q_proj = query @ S^T
    q_float = query.float()
    q_rot = (q_float @ Pi.T).contiguous()    # (1, 1, D)
    q_proj = (q_float @ S.T).contiguous()    # (1, 1, D)

    # Expand for all q_heads (GQA: all q heads use same kv head for now)
    q_rot_full = q_rot.expand(1, num_q_heads, D).contiguous()
    q_proj_full = q_proj.expand(1, num_q_heads, D).contiguous()

    # Replicate k_cache for all kv heads (must be contiguous for CUDA kernel)
    k_cache_full = k_cache.repeat(1, 1, num_kv_heads, 1).contiguous()

    # Block table and seq lens
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    seq_lens_t = torch.tensor([seq_len], device=device, dtype=torch.int32)

    attn_scale = 1.0 / math.sqrt(D)

    # === Reference: compute scores from PACKED cache (same data as kernel) ===
    correction = math.sqrt(math.pi / 2) / D
    mse_bytes = (D * mse_bits + 7) // 8
    qjl_bytes = (D + 7) // 8
    ref_scores = torch.zeros(seq_len, device=device)
    for t in range(seq_len):
        bi, bo = t // block_size, t % block_size
        packed = k_cache[bi, bo, 0]  # the PACKED data

        # Unpack MSE indices
        idx = torch.zeros(D, dtype=torch.int32, device=device)
        for j in range(D):
            b_idx = j // 4; bit = (j % 4) * 2
            idx[j] = (packed[b_idx].item() >> bit) & 0x3

        # Unpack signs
        sign_start = mse_bytes
        signs = torch.zeros(D, dtype=torch.float32, device=device)
        for j in range(D):
            b_idx = j // 8; bit = j % 8
            signs[j] = 1.0 if ((packed[sign_start + b_idx].item() >> bit) & 1) else -1.0

        # Unpack norms
        norm_start = mse_bytes + qjl_bytes
        vn_u16 = packed[norm_start].item() | (packed[norm_start + 1].item() << 8)
        rn_u16 = packed[norm_start + 2].item() | (packed[norm_start + 3].item() << 8)
        import struct
        vn = struct.unpack('e', struct.pack('H', vn_u16))[0]
        rn = struct.unpack('e', struct.pack('H', rn_u16))[0]

        c_idx = centroids[idx.long()]
        t1 = (q_rot[0, 0] * c_idx).sum().item()
        t2 = (q_proj[0, 0] * signs).sum().item()
        ref_scores[t] = vn * (t1 + correction * rn * t2) * attn_scale

    # === CUDA kernel scores ===
    scores_out = torch.full((1, num_q_heads, seq_len), float("-inf"),
                            device=device, dtype=torch.float32)

    # Write the JIT extension wrapper
    import os
    ext_src = "/opt/tq_build/tq_compressed_score_ext.cu"
    if not os.path.exists(ext_src):
        with open(ext_src, "w") as f:
            f.write('''
#include <torch/extension.h>
namespace turboquant {
void tq_compressed_attention_score(
    torch::Tensor q_rot, torch::Tensor q_proj,
    torch::Tensor k_cache, torch::Tensor centroids,
    torch::Tensor block_table, torch::Tensor seq_lens,
    torch::Tensor scores,
    int head_dim, int mse_bits, int n_centroids, float attn_scale);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("score", &turboquant::tq_compressed_attention_score);
}
''')

    score_ext, _ = build_kernels()

    score_ext.score(
        q_rot_full, q_proj_full,
        k_cache_full, centroids,
        block_table, seq_lens_t,
        scores_out,
        D, mse_bits, NC, attn_scale)

    # Compare (use head 0, all heads should be identical for GQA=1)
    cuda_scores = scores_out[0, 0, :seq_len]

    corr = torch.corrcoef(torch.stack([ref_scores, cuda_scores]))[0, 1].item()
    cos = F.cosine_similarity(ref_scores.unsqueeze(0), cuda_scores.unsqueeze(0)).item()
    max_diff = (ref_scores - cuda_scores).abs().max().item()

    print(f"\n  D={D}, seq_len={seq_len}, TQ3")
    print(f"  Score correlation: {corr:.6f}")
    print(f"  Score cosine:      {cos:.6f}")
    print(f"  Max diff:          {max_diff:.6f}")
    print(f"  Ref[:5]:  {ref_scores[:5]}")
    print(f"  CUDA[:5]: {cuda_scores[:5]}")

    assert corr > 0.95, f"Score correlation too low: {corr}"
    print("  PASSED\n")

    # === Benchmark ===
    print("  Benchmark:")
    # Warmup
    for _ in range(10):
        score_ext.score(q_rot_full, q_proj_full, k_cache_full, centroids,
                        block_table, seq_lens_t, scores_out,
                        D, mse_bits, NC, attn_scale)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(1000):
        score_ext.score(q_rot_full, q_proj_full, k_cache_full, centroids,
                        block_table, seq_lens_t, scores_out,
                        D, mse_bits, NC, attn_scale)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 1000

    print(f"  {seq_len} tokens × {num_q_heads} heads: {t*1e6:.1f} µs")
    print(f"  Throughput: {seq_len * num_q_heads / t:.0f} score-computations/s")

    # Memory comparison
    compressed_bytes = k_cache_full.numel()
    bf16_bytes = seq_len * num_kv_heads * D * 2
    print(f"\n  Memory: {compressed_bytes / 1024:.1f} KB (compressed)")
    print(f"          {bf16_bytes / 1024:.1f} KB (BF16)")
    print(f"          {compressed_bytes / bf16_bytes:.2f}x compression")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    test_compressed_score()
    print("=" * 60)
    print("COMPRESSED SCORE KERNEL TEST PASSED")
    print("=" * 60)
