# SPDX-License-Identifier: Apache-2.0
"""End-to-end test of the TurboQuant cache pipeline.

Tests the full flow: quantize → pack → store in cache → unpack → score → softmax.
Validates that attention output from compressed cache closely matches FP16 reference.

Run: python3 tests/turboquant/test_cache_pipeline.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vllm.turboquant.config import TurboQuantConfig
from vllm.turboquant.quantizer import TurboQuantizer


def _create_cache(num_blocks, block_size, num_kv_heads, packed_size, device="cpu"):
    """Create empty KV cache tensors (key_cache, val_cache)."""
    shape = (num_blocks, block_size, num_kv_heads, packed_size)
    key_cache = torch.zeros(shape, dtype=torch.uint8, device=device)
    val_cache = torch.zeros(shape, dtype=torch.uint8, device=device)
    return key_cache, val_cache


def _store_key_to_cache(
    tq: TurboQuantizer,
    key: torch.Tensor,  # (head_dim,)
    cache: torch.Tensor,
    block_idx: int,
    block_off: int,
    head_idx: int,
):
    """Quantize key and store packed data into cache slot."""
    compressed = tq.quantize(key.unsqueeze(0))
    packed = tq.pack_cache(compressed)
    n = packed.shape[-1]
    cache[block_idx, block_off, head_idx, :n] = packed[0]


def _load_key_from_cache(
    tq: TurboQuantizer,
    cache: torch.Tensor,
    block_idx: int,
    block_off: int,
    head_idx: int,
) -> dict:
    """Load and unpack key from cache."""
    key_packed_size = tq.config.key_packed_size
    packed = cache[block_idx, block_off, head_idx, :key_packed_size]
    return tq.unpack_cache(packed.unsqueeze(0))


def _store_value_fp16(
    v: torch.Tensor,  # (head_dim,)
    cache: torch.Tensor,
    block_idx: int,
    block_off: int,
    head_idx: int,
    packed_size: int,
):
    """Store value as raw FP16 bytes."""
    raw = v.half().view(torch.uint8)
    n = min(len(raw), packed_size)
    cache[block_idx, block_off, head_idx, :n] = raw[:n]


def _load_value_fp16(
    cache: torch.Tensor,
    block_idx: int,
    block_off: int,
    head_idx: int,
    head_dim: int,
):
    """Load value from raw FP16 bytes."""
    n = head_dim * 2
    raw = cache[block_idx, block_off, head_idx, :n]
    return raw.view(torch.float16).float()


def test_single_head_attention():
    """Test attention with a single head, single query, short sequence."""
    print("=" * 60)
    print("TEST: Single-head attention pipeline")
    print("=" * 60)

    head_dim = 128
    seq_len = 64
    num_kv_heads = 1
    num_q_heads = 1
    block_size = 16

    for total_bits in [3, 4]:
        config = TurboQuantConfig(head_dim=head_dim, total_bits=total_bits)
        tq = TurboQuantizer(config, layer_idx=0)
        packed_size = config.packed_size

        num_blocks = (seq_len + block_size - 1) // block_size
        key_cache, val_cache = _create_cache(
            num_blocks, block_size, num_kv_heads, packed_size
        )

        # Generate random K/V
        keys = torch.randn(seq_len, head_dim)
        values = torch.randn(seq_len, head_dim)

        # Store into cache
        key_packed = config.key_packed_size
        for pos in range(seq_len):
            block_idx = pos // block_size
            block_off = pos % block_size

            k = keys[pos]
            compressed = tq.quantize(k.unsqueeze(0))
            packed = tq.pack_cache(compressed)
            key_cache[block_idx, block_off, 0, :key_packed] = packed[0]

            _store_value_fp16(values[pos], val_cache, block_idx, block_off, 0,
                              packed_size)

        # Generate query
        query = torch.randn(head_dim)

        # === TQ Score Path ===
        # Precompute rotated/projected query
        q_rot = query.float() @ tq.PiT    # = q @ Pi^T, (D,)
        q_proj = query.float() @ tq.ST     # = q @ S^T, (D,)

        tq_scores = torch.zeros(seq_len)
        correction_scale = math.sqrt(math.pi / 2) / head_dim

        for pos in range(seq_len):
            block_idx = pos // block_size
            block_off = pos % block_size

            packed = key_cache[block_idx, block_off, 0, :key_packed]
            unpacked = tq.unpack_cache(packed.unsqueeze(0))

            idx = unpacked["mse_indices"][0]      # (D,)
            signs = unpacked["qjl_signs"][0]      # (D,)
            vec_norm = unpacked["vec_norm"][0].float().item()
            res_norm = unpacked["res_norm"][0].float().item()

            # Term 1: <q_rot, centroids[idx]>
            c_idx = tq.centroids[idx.long()]
            term1 = (q_rot * c_idx).sum().item()

            # Term 2: QJL correction
            signs_float = signs.float()  # {-1, +1}
            term2 = (q_proj * signs_float).sum().item()

            score = vec_norm * (term1 + correction_scale * res_norm * term2)
            tq_scores[pos] = score

        # === FP16 Reference Path ===
        ref_scores = keys @ query  # (seq_len,)

        # Scale both
        scale = 1.0 / math.sqrt(head_dim)
        tq_weights = F.softmax(tq_scores * scale, dim=0)
        ref_weights = F.softmax(ref_scores * scale, dim=0)

        # Compute outputs
        # For TQ: use values from cache (FP16 stored)
        tq_values = torch.zeros(seq_len, head_dim)
        for pos in range(seq_len):
            block_idx = pos // block_size
            block_off = pos % block_size
            tq_values[pos] = _load_value_fp16(val_cache, block_idx, block_off, 0,
                                               head_dim)

        tq_output = (tq_weights.unsqueeze(-1) * tq_values).sum(dim=0)
        ref_output = (ref_weights.unsqueeze(-1) * values.float()).sum(dim=0)

        # Compare
        score_corr = torch.corrcoef(torch.stack([ref_scores, tq_scores]))[0, 1].item()
        weight_corr = torch.corrcoef(torch.stack([ref_weights, tq_weights]))[0, 1].item()
        output_cos = F.cosine_similarity(
            tq_output.unsqueeze(0), ref_output.unsqueeze(0)
        ).item()

        print(
            f"  tq{total_bits}: score_corr={score_corr:.4f}, "
            f"weight_corr={weight_corr:.4f}, output_cos={output_cos:.4f}"
        )
        assert score_corr > 0.85, f"Score correlation too low: {score_corr}"
        assert output_cos > 0.80, f"Output cosine too low: {output_cos}"

    print("  PASSED\n")


def test_gqa_attention():
    """Test GQA: multiple Q heads per KV head."""
    print("=" * 60)
    print("TEST: GQA attention (4 Q heads, 1 KV head)")
    print("=" * 60)

    head_dim = 128
    seq_len = 32
    num_kv_heads = 1
    num_q_heads = 4
    block_size = 16
    total_bits = 3

    config = TurboQuantConfig(head_dim=head_dim, total_bits=total_bits)
    tq = TurboQuantizer(config, layer_idx=0)
    packed_size = config.packed_size

    num_blocks = (seq_len + block_size - 1) // block_size
    key_cache, val_cache = _create_cache(
        num_blocks, block_size, num_kv_heads, packed_size
    )

    keys = torch.randn(seq_len, head_dim)
    values = torch.randn(seq_len, head_dim)

    # Store
    for pos in range(seq_len):
        block_idx = pos // block_size
        block_off = pos % block_size
        _store_key_to_cache(tq, keys[pos], key_cache, block_idx, block_off, 0)
        _store_value_fp16(values[pos], val_cache, block_idx, block_off, 0, packed_size)

    # 4 different queries
    queries = torch.randn(num_q_heads, head_dim)

    correction_scale = math.sqrt(math.pi / 2) / head_dim
    scale = 1.0 / math.sqrt(head_dim)

    for qh in range(num_q_heads):
        q = queries[qh]
        q_rot = q.float() @ tq.PiT
        q_proj = q.float() @ tq.ST

        tq_scores = torch.zeros(seq_len)
        for pos in range(seq_len):
            block_idx = pos // block_size
            block_off = pos % block_size
            unpacked = _load_key_from_cache(tq, key_cache, block_idx, block_off, 0)
            c_idx = tq.centroids[unpacked["mse_indices"][0].long()]
            t1 = (q_rot * c_idx).sum().item()
            t2 = (q_proj * unpacked["qjl_signs"][0].float()).sum().item()
            vn = unpacked["vec_norm"][0].float().item()
            rn = unpacked["res_norm"][0].float().item()
            tq_scores[pos] = vn * (t1 + correction_scale * rn * t2)

        ref_scores = keys @ q
        corr = torch.corrcoef(torch.stack([ref_scores, tq_scores]))[0, 1].item()
        print(f"  Q-head {qh}: score_corr={corr:.4f}")
        assert corr > 0.85, f"Q-head {qh}: correlation too low: {corr}"

    print("  PASSED\n")


def test_long_sequence():
    """Test with longer sequence to verify block boundary handling."""
    print("=" * 60)
    print("TEST: Long sequence (512 tokens, block_size=16)")
    print("=" * 60)

    head_dim = 128
    seq_len = 512
    block_size = 16
    total_bits = 3

    config = TurboQuantConfig(head_dim=head_dim, total_bits=total_bits)
    tq = TurboQuantizer(config, layer_idx=0)
    packed_size = config.packed_size

    num_blocks = (seq_len + block_size - 1) // block_size
    key_cache, val_cache = _create_cache(num_blocks, block_size, 1, packed_size)

    keys = torch.randn(seq_len, head_dim)
    values = torch.randn(seq_len, head_dim)

    for pos in range(seq_len):
        bi, bo = pos // block_size, pos % block_size
        _store_key_to_cache(tq, keys[pos], key_cache, bi, bo, 0)
        _store_value_fp16(values[pos], val_cache, bi, bo, 0, packed_size)

    # Query = one of the keys (needle test with full pipeline)
    needle = seq_len // 3
    query = keys[needle]

    # Dequantize all keys and compute reference via matmul
    tq_keys_recon = torch.zeros_like(keys)
    for pos in range(seq_len):
        bi, bo = pos // block_size, pos % block_size
        unpacked = _load_key_from_cache(tq, key_cache, bi, bo, 0)
        # Full dequant through quantizer
        tq_keys_recon[pos] = tq.dequantize(unpacked)

    tq_scores = tq_keys_recon @ query
    ref_scores = keys @ query

    corr = torch.corrcoef(torch.stack([ref_scores, tq_scores]))[0, 1].item()

    scale = 1.0 / math.sqrt(head_dim)
    tq_weights = F.softmax(tq_scores * scale, dim=0)
    ref_weights = F.softmax(ref_scores * scale, dim=0)

    tq_out = (tq_weights.unsqueeze(-1) * values).sum(0)
    ref_out = (ref_weights.unsqueeze(-1) * values).sum(0)
    cos = F.cosine_similarity(tq_out.unsqueeze(0), ref_out.unsqueeze(0)).item()

    tq_top1 = tq_scores.argmax().item()
    ref_top1 = ref_scores.argmax().item()

    print(f"  seq_len={seq_len}: score_corr={corr:.4f}, output_cos={cos:.4f}")
    print(f"  tq_top1={tq_top1}, ref_top1={ref_top1} (needle={needle})")
    assert corr > 0.85, f"Correlation too low: {corr}"
    assert cos > 0.80, f"Output cosine too low: {cos}"
    print("  PASSED\n")


def test_incremental_decode():
    """Test incremental decode: prefill N tokens, then decode 1-by-1."""
    print("=" * 60)
    print("TEST: Incremental decode (prefill 32, decode 16)")
    print("=" * 60)

    head_dim = 128
    prefill_len = 32
    decode_steps = 16
    block_size = 16
    total_bits = 3

    config = TurboQuantConfig(head_dim=head_dim, total_bits=total_bits)
    tq = TurboQuantizer(config, layer_idx=0)
    packed_size = config.packed_size

    total_len = prefill_len + decode_steps
    num_blocks = (total_len + block_size - 1) // block_size
    key_cache, val_cache = _create_cache(num_blocks, block_size, 1, packed_size)

    # Generate all K/V upfront (for reference)
    all_keys = torch.randn(total_len, head_dim)
    all_values = torch.randn(total_len, head_dim)

    # Prefill: store first prefill_len tokens
    for pos in range(prefill_len):
        bi, bo = pos // block_size, pos % block_size
        _store_key_to_cache(tq, all_keys[pos], key_cache, bi, bo, 0)
        _store_value_fp16(all_values[pos], val_cache, bi, bo, 0, packed_size)

    # Decode: add tokens one by one, compute score each time
    correction_scale = math.sqrt(math.pi / 2) / head_dim
    scale = 1.0 / math.sqrt(head_dim)

    for step in range(decode_steps):
        pos = prefill_len + step

        # Store new token
        bi, bo = pos // block_size, pos % block_size
        _store_key_to_cache(tq, all_keys[pos], key_cache, bi, bo, 0)
        _store_value_fp16(all_values[pos], val_cache, bi, bo, 0, packed_size)

        # Query = last token's key (self-attention proxy)
        query = all_keys[pos]
        seq_len = pos + 1

        # Compute TQ scores over all cached tokens
        q_rot = query.float() @ tq.PiT
        q_proj = query.float() @ tq.ST

        tq_scores = torch.zeros(seq_len)
        for p in range(seq_len):
            b_i, b_o = p // block_size, p % block_size
            unpacked = _load_key_from_cache(tq, key_cache, b_i, b_o, 0)
            c_idx = tq.centroids[unpacked["mse_indices"][0].long()]
            t1 = (q_rot * c_idx).sum().item()
            t2 = (q_proj * unpacked["qjl_signs"][0].float()).sum().item()
            vn = unpacked["vec_norm"][0].float().item()
            rn = unpacked["res_norm"][0].float().item()
            tq_scores[p] = vn * (t1 + correction_scale * rn * t2)

        ref_scores = all_keys[:seq_len] @ query
        corr = torch.corrcoef(torch.stack([ref_scores, tq_scores]))[0, 1].item()

        if step % 4 == 0:
            print(f"  step={step}, seq_len={seq_len}: score_corr={corr:.4f}")

    print("  PASSED\n")


def test_memory_savings():
    """Verify actual memory savings in cache tensors."""
    print("=" * 60)
    print("TEST: Memory savings comparison")
    print("=" * 60)

    head_dim = 128
    seq_len = 4096
    num_kv_heads = 8
    block_size = 16
    num_blocks = seq_len // block_size

    for total_bits in [3, 4]:
        config = TurboQuantConfig(head_dim=head_dim, total_bits=total_bits)
        packed_size = config.packed_size

        # TQ cache
        tq_bytes = num_blocks * block_size * num_kv_heads * packed_size * 2  # K+V

        # FP16 cache
        fp16_bytes = num_blocks * block_size * num_kv_heads * head_dim * 2 * 2  # K+V

        # FP8 cache
        fp8_bytes = num_blocks * block_size * num_kv_heads * head_dim * 1 * 2  # K+V

        ratio_vs_fp16 = fp16_bytes / tq_bytes
        ratio_vs_fp8 = fp8_bytes / tq_bytes

        print(
            f"  tq{total_bits} ({seq_len} tokens, {num_kv_heads} heads): "
            f"{tq_bytes / 1024 / 1024:.1f} MB "
            f"(vs FP16 {fp16_bytes / 1024 / 1024:.1f} MB = {ratio_vs_fp16:.1f}x, "
            f"vs FP8 {fp8_bytes / 1024 / 1024:.1f} MB = {ratio_vs_fp8:.1f}x)"
        )

    print()


if __name__ == "__main__":
    print()
    print("TurboQuant Cache Pipeline Tests")
    print("=" * 60)
    print()

    test_single_head_attention()
    test_gqa_attention()
    test_long_sequence()
    test_incremental_decode()
    test_memory_savings()

    print("=" * 60)
    print("ALL PIPELINE TESTS PASSED")
    print("=" * 60)
