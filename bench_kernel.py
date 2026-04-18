#!/usr/bin/env python3
"""
Direct kernel benchmarking — bypasses vLLM, measures pack + decode kernel times.
Run inside container with GPU.
"""
import os
import sys
import time
import torch

def main():
    os.environ.setdefault("VLLM_MLA_DISABLE", "1")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "CutlassFP8ScaledMMLinearKernel")

    # Import WHT kernels
    sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
    from vllm.v1.attention.ops.triton_mq_fused_decode import (
        _load_wht_pack_kernel, _load_wht_cuda_kernel,
    )

    print("Loading kernels...")
    pack_kernel = _load_wht_pack_kernel()
    decode_kernel = _load_wht_cuda_kernel()
    assert pack_kernel is not None, "Pack kernel failed to load"
    assert decode_kernel is not None, "Decode kernel failed to load"
    print("Kernels loaded.")

    device = torch.device("cuda")
    D = 256  # GLM-4.7-Flash head dim
    num_heads = 4  # num_kv_heads for GLM
    block_size = 16
    packed_size = (D // 32) * 14  # 112 bytes per head
    mse_bits = 3  # tq3w = 3-bit WHT

    # Simulate decode: batch=1, 40 attention layers
    num_layers = 40
    seq_len = 50  # typical during generation

    # KV cache: [num_blocks, 2, block_size, num_heads, max_packed]
    num_blocks = (seq_len + block_size - 1) // block_size
    max_packed = 256  # padded
    kv_cache = torch.zeros(num_blocks * 2, 2, block_size, num_heads, max_packed,
                          dtype=torch.uint8, device=device)

    # Fill KV cache with some data (simulate having written KV)
    # Pack random bf16 vectors
    for b in range(num_blocks):
        for slot in range(block_size):
            if b * block_size + slot >= seq_len:
                break
            for kv_idx in range(2):
                vec = torch.randn(num_heads, D, device=device, dtype=torch.bfloat16)
                v_flat = vec.reshape(-1, D)
                packed = torch.empty(num_heads, packed_size, dtype=torch.uint8, device=device)
                pack_kernel.tq_wht_pack(v_flat, packed)
                kv_cache[b, kv_idx, slot, :, :packed_size] = packed

    # Query: bf16, shape [1, num_query_heads, D]
    num_q_heads = 32  # GQA
    query = torch.randn(1, num_q_heads, D, device=device, dtype=torch.bfloat16)

    # Block table: [1, num_blocks] int32
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

    # Seq lens: [1] int32
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    # Output buffer
    output = torch.empty(1, num_q_heads, D, device=device, dtype=torch.bfloat16)

    # Strides
    s_b = kv_cache.stride(0)
    s_kv = kv_cache.stride(1)
    s_s = kv_cache.stride(2)
    s_h = kv_cache.stride(3)

    scale = 1.0 / (D ** 0.5)

    # Pack kernel benchmark
    print(f"\n{'='*60}")
    print(f"Pack kernel benchmark (D={D}, num_heads={num_heads})")
    print(f"{'='*60}")

    k_input = torch.randn(num_heads, D, device=device, dtype=torch.bfloat16)
    k_packed = torch.empty(num_heads, packed_size, dtype=torch.uint8, device=device)

    # Warmup
    for _ in range(10):
        pack_kernel.tq_wht_pack(k_input, k_packed)
    torch.cuda.synchronize()

    # Measure
    n_iters = 1000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iters):
        pack_kernel.tq_wht_pack(k_input, k_packed)
    end.record()
    torch.cuda.synchronize()
    pack_ms = start.elapsed_time(end) / n_iters
    print(f"  Pack:   {pack_ms*1000:.1f} us per call ({num_heads} heads × D={D})")
    print(f"  Per layer (2 packs): {pack_ms*2*1000:.1f} us")
    print(f"  Per token (40 layers): {pack_ms*2*40*1000:.1f} us = {pack_ms*2*40:.3f} ms")

    # Decode kernel benchmark
    print(f"\n{'='*60}")
    print(f"Decode kernel benchmark (D={D}, seq_len={seq_len}, {num_q_heads} q_heads)")
    print(f"{'='*60}")

    # Warmup
    for _ in range(10):
        decode_kernel.tq_wht_fused_decode_attention(
            query, kv_cache, block_table, seq_lens,
            output, D, mse_bits, scale, s_b, s_kv, s_s, s_h)
    torch.cuda.synchronize()

    # Measure
    start.record()
    for _ in range(n_iters):
        decode_kernel.tq_wht_fused_decode_attention(
            query, kv_cache, block_table, seq_lens,
            output, D, mse_bits, scale, s_b, s_kv, s_s, s_h)
    end.record()
    torch.cuda.synchronize()
    decode_ms = start.elapsed_time(end) / n_iters
    print(f"  Decode: {decode_ms*1000:.1f} us per call (B=1, sl={seq_len}, {num_q_heads} heads)")
    print(f"  Per token (40 layers): {decode_ms*40*1000:.1f} us = {decode_ms*40:.3f} ms")

    # Total per-token time for attention
    total_attn_ms = (pack_ms * 2 + decode_ms) * num_layers
    print(f"\n{'='*60}")
    print(f"Total attention time per token (40 layers):")
    print(f"  Pack:   {pack_ms*2*num_layers:.3f} ms")
    print(f"  Decode: {decode_ms*num_layers:.3f} ms")
    print(f"  TOTAL:  {total_attn_ms:.3f} ms")
    print(f"  (at 34 tok/s, total token time = {1000/34:.1f} ms)")
    print(f"  Attention share: {total_attn_ms/(1000/34)*100:.1f}%")

    # Also measure at longer sequence lengths
    print(f"\n{'='*60}")
    print(f"Decode kernel scaling with seq_len")
    print(f"{'='*60}")
    for sl in [10, 50, 100, 200, 500, 1000]:
        nb = (sl + block_size - 1) // block_size
        kvc = torch.zeros(nb * 2, 2, block_size, num_heads, max_packed,
                         dtype=torch.uint8, device=device)
        bt = torch.arange(nb, dtype=torch.int32, device=device).unsqueeze(0)
        sls = torch.tensor([sl], dtype=torch.int32, device=device)
        out = torch.empty(1, num_q_heads, D, device=device, dtype=torch.bfloat16)

        for _ in range(5):
            decode_kernel.tq_wht_fused_decode_attention(
                query, kvc, bt, sls, out, D, mse_bits, scale,
                kvc.stride(0), kvc.stride(1), kvc.stride(2), kvc.stride(3))
        torch.cuda.synchronize()

        start.record()
        for _ in range(200):
            decode_kernel.tq_wht_fused_decode_attention(
                query, kvc, bt, sls, out, D, mse_bits, scale,
                kvc.stride(0), kvc.stride(1), kvc.stride(2), kvc.stride(3))
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 200
        print(f"  sl={sl:5d}: {ms*1000:.1f} us per call, "
              f"40 layers = {ms*40:.3f} ms")

if __name__ == "__main__":
    main()
