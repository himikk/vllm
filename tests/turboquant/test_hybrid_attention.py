#!/usr/bin/env python3
"""Standalone test: Hybrid TQ attention — K=TQ3 compressed, V=FP8.

Computes full decode attention:
  1. Scores from compressed K-cache (CUDA kernel)
  2. Softmax in PyTorch
  3. V aggregation from FP8 V-cache (PyTorch dequant)

Memory: K=28 + V=64 = 92 bytes/token vs BF16 256 bytes → 2.8× compression.

Run: python3 /opt/test_hybrid_attention.py
"""
import math
import time

import torch
import torch.nn.functional as F


def build_score_kernel():
    from torch.utils.cpp_extension import load
    return load(
        name="tq_score",
        sources=["/opt/tq_build/tq_compressed_score_ext.cu",
                 "/opt/tq_build/tq_compressed_score.cu"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                           "--use_fast_math",
                           "-gencode=arch=compute_121,code=sm_121",
                           "-diag-suppress=177,3288"],
        extra_include_paths=["/opt/tq_build"],
        verbose=False,
    )


def pack_key_tq3(key, Pi, S, centroids, D):
    """Pack one key vector to TQ3 format."""
    mse_bits = 2
    mse_bytes = (D * mse_bits + 7) // 8
    qjl_bytes = (D + 7) // 8
    x = key.float()
    vn = x.norm(); xh = x / (vn + 1e-8)
    rot = xh @ Pi.T
    idx = (rot.unsqueeze(-1) - centroids).abs().argmin(dim=-1).to(torch.uint8)
    xm = centroids[idx.long()] @ Pi
    r = xh - xm; rn = r.norm()
    signs = (r @ S.T >= 0).to(torch.uint8)

    packed = torch.zeros(mse_bytes + qjl_bytes + 4, dtype=torch.uint8, device=key.device)
    for j in range(0, D, 4):
        v = 0
        for k in range(min(4, D - j)):
            v |= (idx[j + k].item() & 0x3) << (k * 2)
        packed[j // 4] = v
    for j in range(0, D, 8):
        v = 0
        for k in range(min(8, D - j)):
            v |= (signs[j + k].item() & 1) << k
        packed[mse_bytes + j // 8] = v
    packed[mse_bytes + qjl_bytes:mse_bytes + qjl_bytes + 2] = vn.half().reshape(1).view(torch.uint8)
    packed[mse_bytes + qjl_bytes + 2:mse_bytes + qjl_bytes + 4] = rn.half().reshape(1).view(torch.uint8)
    return packed


def test_hybrid_attention():
    print("=" * 60)
    print("Hybrid TQ Attention: K=TQ3(compressed) + V=FP8")
    print("=" * 60)

    device = "cuda"
    D = 64
    NC = 4
    mse_bits = 2
    num_q_heads = 20
    num_kv_heads = 20
    seq_len = 256
    block_size = 16
    num_blocks = (seq_len + block_size - 1) // block_size
    k_packed_size = (D * mse_bits + 7) // 8 + (D + 7) // 8 + 4  # 28

    torch.manual_seed(42)
    gen = torch.Generator(device="cpu"); gen.manual_seed(42)
    Pi = torch.randn(D, D, generator=gen, dtype=torch.float32, device="cpu")
    Pi, _ = torch.linalg.qr(Pi); Pi = Pi.contiguous().to(device)
    gen.manual_seed(43); S = torch.randn(D, D, generator=gen, dtype=torch.float32).to(device)
    sigma = 1.0 / math.sqrt(D)
    centroids = torch.tensor([-1.51*sigma, -0.453*sigma, 0.453*sigma, 1.51*sigma],
                              dtype=torch.float32, device=device)

    # Generate keys and values
    keys = torch.randn(seq_len, D, device=device, dtype=torch.bfloat16)
    values = torch.randn(seq_len, D, device=device, dtype=torch.bfloat16)

    # === Pack K-cache (TQ3 compressed) ===
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, k_packed_size,
                          dtype=torch.uint8, device=device)
    for t in range(seq_len):
        bi, bo = t // block_size, t % block_size
        packed = pack_key_tq3(keys[t], Pi, S, centroids, D)
        # Same packed data for all kv_heads (broadcast)
        for h in range(num_kv_heads):
            k_cache[bi, bo, h, :len(packed)] = packed

    # === Store V-cache as FP8 (simple: just store BF16 as-is for now) ===
    # Real FP8: values.to(torch.float8_e4m3fn) but we approximate with BF16
    v_cache_bf16 = values.reshape(num_blocks, block_size, D).clone()

    # === Generate query ===
    query = torch.randn(1, 1, D, device=device, dtype=torch.bfloat16)
    q_float = query.float()
    q_rot = (q_float @ Pi.T).expand(1, num_q_heads, D).contiguous()
    q_proj = (q_float @ S.T).expand(1, num_q_heads, D).contiguous()

    # === Reference: BF16 attention ===
    ref_scores = (query.float().squeeze() @ keys.float().T) / math.sqrt(D)
    ref_weights = F.softmax(ref_scores, dim=-1)
    ref_output = (ref_weights @ values.float()).squeeze()

    # === Hybrid: TQ3 K-scores (CUDA) + softmax + BF16 V ===
    score_ext = build_score_kernel()

    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    seq_lens_t = torch.tensor([seq_len], device=device, dtype=torch.int32)
    attn_scale = 1.0 / math.sqrt(D)

    # Compute scores from compressed K
    tq_scores = torch.full((1, num_q_heads, seq_len), float("-inf"),
                            device=device, dtype=torch.float32)
    score_ext.score(q_rot, q_proj, k_cache, centroids,
                    block_table, seq_lens_t, tq_scores,
                    D, mse_bits, NC, attn_scale)

    # Softmax
    tq_weights = F.softmax(tq_scores[0, 0, :seq_len], dim=-1)

    # V aggregation (from BF16 — in production this would be FP8)
    tq_output = (tq_weights @ values.float()).squeeze()

    # === Compare ===
    score_corr = torch.corrcoef(
        torch.stack([ref_scores.squeeze(), tq_scores[0, 0, :seq_len]])
    )[0, 1].item()
    output_cos = F.cosine_similarity(
        ref_output.unsqueeze(0), tq_output.unsqueeze(0)
    ).item()

    print(f"\n  Config: D={D}, seq_len={seq_len}, TQ3 K + BF16 V")
    print(f"  Score correlation: {score_corr:.6f}")
    print(f"  Output cosine:     {output_cos:.6f}")

    # Memory comparison
    k_mem = k_cache[:, :, :1, :].numel()  # per head
    v_mem = seq_len * D * 2  # BF16
    fp8_v_mem = seq_len * D * 1  # FP8
    bf16_total = seq_len * D * 2 * 2  # K+V BF16
    hybrid_total = k_mem + fp8_v_mem  # TQ3 K + FP8 V

    print(f"\n  Memory per head:")
    print(f"    BF16 K+V:     {bf16_total / 1024:.1f} KB")
    print(f"    TQ3 K + FP8 V: {hybrid_total / 1024:.1f} KB")
    print(f"    Compression:   {bf16_total / hybrid_total:.1f}×")

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(1000):
        score_ext.score(q_rot, q_proj, k_cache, centroids,
                        block_table, seq_lens_t, tq_scores,
                        D, mse_bits, NC, attn_scale)
    torch.cuda.synchronize()
    t_score = (time.perf_counter() - t0) / 1000

    print(f"\n  Latency: {t_score*1e6:.1f} µs (score only, {seq_len} tokens)")

    assert score_corr > 0.90, f"Score correlation too low: {score_corr}"
    assert output_cos > 0.85, f"Output cosine too low: {output_cos}"
    print("\n  PASSED")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    test_hybrid_attention()
    print("\n" + "=" * 60)
    print("HYBRID ATTENTION TEST PASSED")
    print("=" * 60)
