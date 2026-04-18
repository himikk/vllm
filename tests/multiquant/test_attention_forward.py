#!/usr/bin/env python3
"""MultiQuant Attention Forward — full pipeline integration tests.

Tests the actual MultiQuantImpl.forward() path with real KV-cache,
simulating the exact data flow from vLLM serving:
  Prefill: bmm causal attention on raw K/V
  KV Write: vectorized _pack_batch → slot_mapping → cache write
  Decode: block_table → gather → unpack → compressed score → softmax → output

These tests bridge the gap between isolated component tests and vllm serve.
"""

import math
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_impl(D=128, num_heads=4, num_kv_heads=4, kv_cache_dtype="tq3"):
    """Create a MultiQuantImpl + fake attention layer with buffers."""
    from vllm.v1.attention.backends.multiquant_attn import MultiQuantImpl

    impl = MultiQuantImpl(
        num_heads=num_heads,
        head_size=D,
        scale=1.0 / math.sqrt(D),
        num_kv_heads=num_kv_heads,
        kv_cache_dtype=kv_cache_dtype,
    )

    # Create fake attention layer with required buffers
    layer = nn.Module()
    is_rq = kv_cache_dtype.startswith("rq")

    if is_rq:
        from vllm.multiquant.rotorquant.quantizer import generate_rotors
        Pi = generate_rotors(D, seed=42).to("cuda").float()
    else:
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        Pi = generate_rotation_matrix(D, seed=42).to("cuda").float()

    from vllm.multiquant.shared.qjl import generate_qjl_matrix
    from vllm.multiquant.shared.centroids import get_centroids

    S = generate_qjl_matrix(D, seed=43).to("cuda").float()
    mse_bits = impl._mse_bits
    centroids = get_centroids(D, mse_bits).to("cuda").float()

    layer.register_buffer("_tq_Pi", Pi)
    layer.register_buffer("_tq_S", S)
    layer.register_buffer("_tq_centroids", centroids)

    return impl, layer


def _make_cache(num_blocks, block_size, num_kv_heads, packed_size, device="cuda"):
    """Allocate KV-cache: (num_blocks, 2, block_size, num_kv_heads, packed_size)."""
    return torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, packed_size,
        dtype=torch.uint8, device=device,
    )


def _make_metadata(seq_lens, block_table, slot_mapping,
                   num_prefill=0, num_decode=0, max_seq_len=0):
    """Create a TQMetadata-like object with required fields."""
    from vllm.v1.attention.backends.multiquant_attn import TQMetadata

    return TQMetadata(
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        num_prefill_tokens=num_prefill,
        num_decode_tokens=num_decode,
        max_seq_len=max_seq_len,
    )


# ── 0. Init Smoke Tests ──────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestImplInit:
    """MultiQuantImpl.__init__ must not crash for any KV dtype."""

    @pytest.mark.parametrize("dtype", ["tq3", "tq4", "rq2", "rq3", "rq4"])
    def test_init_all_dtypes(self, dtype):
        """_is_rq must be set before pre-load block references it."""
        impl, layer = _make_impl(D=128, kv_cache_dtype=dtype)
        assert impl._is_rq == dtype.startswith("rq")
        assert impl._decode_fn is not None
        assert impl._mse_bits > 0


# ── 0b. KV-Cache Integrity ───────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestKVCacheIntegrity:
    """Verify KV-Cache write (prefill) → read (decode) roundtrip."""

    def test_prefill_cache_then_decode_quality(self):
        """Write KV via do_kv_cache_update, decode, compare vs reference."""
        D = 128
        num_heads = 32
        num_kv_heads = 4
        block_size = 16
        seq_len = 64

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        n_blocks = (seq_len + block_size - 1) // block_size
        cache = _make_cache(n_blocks, block_size, num_kv_heads, packed_size)

        torch.manual_seed(42)
        key = torch.randn(seq_len, num_kv_heads, D,
                          dtype=torch.bfloat16, device="cuda")
        value = torch.randn(seq_len, num_kv_heads, D,
                            dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(seq_len, device="cuda")
        impl.do_kv_cache_update(layer, key, value, cache, slot_mapping)

        query = torch.randn(1, num_heads * D,
                            dtype=torch.bfloat16, device="cuda")
        block_table = torch.arange(n_blocks, device="cuda").unsqueeze(0)
        meta = _make_metadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=block_table,
            slot_mapping=torch.tensor([seq_len], device="cuda"),
            num_prefill=0, num_decode=1, max_seq_len=seq_len,
        )
        output = impl.forward(layer, query, None, None, cache, meta)

        # Reference: naive bmm attention on raw K/V
        q = query.reshape(1, num_heads, D).float()
        k = key.float()
        v = value.float()
        if num_kv_heads < num_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
        scale = 1.0 / math.sqrt(D)
        scores = torch.einsum("bhd,shd->bhs", q, k) * scale
        weights = F.softmax(scores, dim=-1)
        ref_out = torch.einsum("bhs,shd->bhd", weights, v).reshape(1, -1)

        cos = F.cosine_similarity(output.float(), ref_out, dim=-1).mean()
        assert cos > 0.3, f"Decode quality: cos={cos:.4f} (expected >0.3)"
        assert not output.isnan().any()
        out_var = output.float().var().item()
        assert out_var > 0.001, f"Output degenerate: var={out_var:.6f}"

    def test_cache_readback_matches_written(self):
        """Unpack what was written to cache, compare vs original K/V."""
        D = 128
        num_kv_heads = 4
        block_size = 16

        impl, layer = _make_impl(D, num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        cache = _make_cache(1, block_size, num_kv_heads, packed_size)

        torch.manual_seed(42)
        key = torch.randn(1, num_kv_heads, D,
                          dtype=torch.bfloat16, device="cuda")
        slot_mapping = torch.tensor([0], device="cuda")
        impl.do_kv_cache_update(layer, key, key, cache, slot_mapping)

        Pi, S, centroids = impl._get_matrices(layer, key.device)

        from vllm.multiquant.weight_quant.archer_ops import cuda_unpack
        for kv_h in range(num_kv_heads):
            packed = cache[0, 0, 0, kv_h].unsqueeze(0)
            result = cuda_unpack(packed, D, impl._mse_bits)
            if result is None:
                pytest.skip("CUDA unpack not available")
            idx, signs, vn, rn = result

            c_vals = centroids[idx.long()]
            W_mse = c_vals @ Pi
            correction = math.sqrt(math.pi / 2) / D
            W_qjl = correction * rn.unsqueeze(-1) * (signs @ S)
            recon = vn.unsqueeze(-1) * (W_mse + W_qjl)

            original = key[0, kv_h].float()
            cos = F.cosine_similarity(
                original.unsqueeze(0), recon, dim=-1
            ).mean()
            assert cos > 0.8, f"Cache readback kv_h={kv_h}: cos={cos:.4f}"

    def test_10_step_autoregressive_not_degenerate(self):
        """10 decode steps must produce diverse outputs (not all '1')."""
        D = 128
        num_heads = 32
        num_kv_heads = 4
        block_size = 16
        seq_len = 32

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        n_blocks = 3
        cache = _make_cache(n_blocks, block_size, num_kv_heads, packed_size)

        torch.manual_seed(42)
        key = torch.randn(seq_len, num_kv_heads, D,
                          dtype=torch.bfloat16, device="cuda")
        value = torch.randn(seq_len, num_kv_heads, D,
                            dtype=torch.bfloat16, device="cuda")
        slot_mapping = torch.arange(seq_len, device="cuda")
        impl.do_kv_cache_update(layer, key, value, cache, slot_mapping)

        outputs = []
        for step in range(10):
            query = torch.randn(1, num_heads * D,
                                dtype=torch.bfloat16, device="cuda")
            block_table = torch.arange(n_blocks, device="cuda").unsqueeze(0)
            meta = _make_metadata(
                seq_lens=torch.tensor([seq_len + step], device="cuda"),
                block_table=block_table,
                slot_mapping=torch.tensor([seq_len + step], device="cuda"),
                num_prefill=0, num_decode=1, max_seq_len=seq_len + step,
            )
            out = impl.forward(layer, query, None, None, cache, meta)
            outputs.append(out.float().clone())

        all_same = all(
            torch.allclose(outputs[0], o, atol=1e-3) for o in outputs[1:]
        )
        assert not all_same, "All 10 decode outputs identical — degenerate!"

        stacked = torch.stack(outputs)
        var = stacked.var().item()
        assert var > 0.0001, f"Output variance too low: {var}"


# ── 1. KV Cache Update (Pack + Write) ────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestKVCacheUpdate:
    """Test do_kv_cache_update: pack → slot_mapping → cache write."""

    @pytest.mark.parametrize("dtype", ["tq3", "rq3", "tq4", "rq4"])
    def test_pack_write_readback(self, dtype):
        """Pack K/V, write to cache via slot_mapping, read back packed data."""
        D = 128
        num_tokens = 8
        num_kv_heads = 4
        block_size = 16

        impl, layer = _make_impl(D, num_kv_heads=num_kv_heads,
                                 kv_cache_dtype=dtype)
        packed_size = impl._packed_size
        cache = _make_cache(2, block_size, num_kv_heads, packed_size)

        torch.manual_seed(42)
        key = torch.randn(num_tokens, num_kv_heads, D,
                          dtype=torch.bfloat16, device="cuda")
        value = torch.randn(num_tokens, num_kv_heads, D,
                            dtype=torch.bfloat16, device="cuda")

        # slot_mapping: token i → block=i//16, offset=i%16
        slot_mapping = torch.arange(num_tokens, device="cuda")

        impl.do_kv_cache_update(layer, key, value, cache, slot_mapping)

        # Verify cache was written (non-zero)
        for i in range(num_tokens):
            bi = i // block_size
            bo = i % block_size
            for kv_h in range(num_kv_heads):
                k_packed = cache[bi, 0, bo, kv_h]
                v_packed = cache[bi, 1, bo, kv_h]
                assert k_packed.any(), f"K cache empty at token={i}, head={kv_h}"
                assert v_packed.any(), f"V cache empty at token={i}, head={kv_h}"

    def test_slot_mapping_noncontiguous(self):
        """Slot mapping with gaps (like vLLM block allocator)."""
        D = 128
        num_tokens = 4
        num_kv_heads = 2
        block_size = 16

        impl, layer = _make_impl(D, num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        cache = _make_cache(4, block_size, num_kv_heads, packed_size)

        torch.manual_seed(42)
        key = torch.randn(num_tokens, num_kv_heads, D,
                          dtype=torch.bfloat16, device="cuda")
        value = torch.randn(num_tokens, num_kv_heads, D,
                            dtype=torch.bfloat16, device="cuda")

        # Scattered slots: tokens go to blocks 0, 2, 1, 3
        slot_mapping = torch.tensor([0, 32, 16, 48], device="cuda")
        impl.do_kv_cache_update(layer, key, value, cache, slot_mapping)

        # Verify data at expected positions
        assert cache[0, 0, 0].any()   # slot 0 → block 0, offset 0
        assert cache[2, 0, 0].any()   # slot 32 → block 2, offset 0
        assert cache[1, 0, 0].any()   # slot 16 → block 1, offset 0
        assert cache[3, 0, 0].any()   # slot 48 → block 3, offset 0


# ── 2. Prefill Forward ───────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPrefillForward:
    """Prefill: causal bmm attention on raw K/V."""

    def test_prefill_causal(self):
        """Prefill output matches naive causal attention."""
        D = 128
        num_heads = 4
        num_kv_heads = 4
        seq_len = 16

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        cache = _make_cache(2, 16, num_kv_heads, packed_size)

        torch.manual_seed(42)
        query = torch.randn(seq_len, num_heads * D,
                            dtype=torch.bfloat16, device="cuda")
        key = torch.randn(seq_len, num_kv_heads * D,
                          dtype=torch.bfloat16, device="cuda")
        value = torch.randn(seq_len, num_kv_heads * D,
                            dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(seq_len, device="cuda")
        block_table = torch.tensor([[0, 1]], device="cuda")

        meta = _make_metadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=block_table,
            slot_mapping=slot_mapping,
            num_prefill=seq_len,
            num_decode=0,
            max_seq_len=seq_len,
        )

        output = impl.forward(layer, query, key, value, cache, meta)

        assert output.shape == (seq_len, num_heads * D)
        assert not output.isnan().any(), "Prefill output has NaN"
        assert output.abs().sum() > 0, "Prefill output is all zeros"

    def test_prefill_gqa(self):
        """Prefill with GQA (num_heads > num_kv_heads)."""
        D = 128
        num_heads = 8
        num_kv_heads = 2
        seq_len = 8

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        cache = _make_cache(1, 16, num_kv_heads, packed_size)

        torch.manual_seed(42)
        query = torch.randn(seq_len, num_heads * D,
                            dtype=torch.bfloat16, device="cuda")
        key = torch.randn(seq_len, num_kv_heads * D,
                          dtype=torch.bfloat16, device="cuda")
        value = torch.randn(seq_len, num_kv_heads * D,
                            dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(seq_len, device="cuda")
        block_table = torch.tensor([[0]], device="cuda")

        meta = _make_metadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=block_table,
            slot_mapping=slot_mapping,
            num_prefill=seq_len,
            num_decode=0,
            max_seq_len=seq_len,
        )

        output = impl.forward(layer, query, key, value, cache, meta)

        assert output.shape == (seq_len, num_heads * D)
        assert not output.isnan().any()


# ── 2b. Prefill GQA Mixed Batch (Warmup Crash) ──────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPrefillGQAMixed:
    """Prefill mit GQA wo num_heads != num_kv_heads — der Serving-Crash."""

    def test_prefill_gqa_warmup_pattern(self):
        """Simuliert vLLM Warmup: mixed batch mit num_prefill != query count."""
        D = 128
        num_heads = 32      # GLM-4.7 Q-Heads
        num_kv_heads = 4    # GLM-4.7 KV-Heads

        # vLLM Warmup sendet mixed prefill-decode:
        # num_decode=4, num_prefill=6, aber query hat 10 Tokens total
        num_decode = 4
        num_prefill = 6
        N = num_decode + num_prefill

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        cache = _make_cache(2, 16, num_kv_heads, packed_size)

        # Fill cache for decode tokens
        torch.manual_seed(42)
        k_ctx = torch.randn(8, num_kv_heads, D,
                            dtype=torch.bfloat16, device="cuda")
        v_ctx = torch.randn(8, num_kv_heads, D,
                            dtype=torch.bfloat16, device="cuda")
        slot_ctx = torch.arange(8, device="cuda")
        impl.do_kv_cache_update(layer, k_ctx, v_ctx, cache, slot_ctx)

        query = torch.randn(N, num_heads * D,
                            dtype=torch.bfloat16, device="cuda")
        key = torch.randn(N, num_kv_heads * D,
                          dtype=torch.bfloat16, device="cuda")
        value = torch.randn(N, num_kv_heads * D,
                            dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(N, device="cuda")
        block_table = torch.tensor([[0, 1]] * num_decode, device="cuda")

        meta = _make_metadata(
            seq_lens=torch.tensor([6] * num_decode, device="cuda"),
            block_table=block_table,
            slot_mapping=slot_mapping,
            num_prefill=num_prefill,
            num_decode=num_decode,
            max_seq_len=6,
        )

        # This crashed with: RuntimeError: shape '[6, -1]' is invalid
        # for input of size 40960 (before L vs num_prefill fix)
        output = impl.forward(layer, query, key, value, cache, meta)

        assert output.shape == (N, num_heads * D)
        assert not output.isnan().any(), "GQA mixed prefill output has NaN"

    def test_prefill_32_heads_4_kv(self):
        """Pure prefill with GLM-4.7 head config (32Q / 4KV)."""
        D = 128
        num_heads = 32
        num_kv_heads = 4
        seq_len = 8

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        cache = _make_cache(1, 16, num_kv_heads, packed_size)

        torch.manual_seed(42)
        query = torch.randn(seq_len, num_heads * D,
                            dtype=torch.bfloat16, device="cuda")
        key = torch.randn(seq_len, num_kv_heads * D,
                          dtype=torch.bfloat16, device="cuda")
        value = torch.randn(seq_len, num_kv_heads * D,
                            dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(seq_len, device="cuda")
        block_table = torch.tensor([[0]], device="cuda")

        meta = _make_metadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=block_table,
            slot_mapping=slot_mapping,
            num_prefill=seq_len,
            num_decode=0,
            max_seq_len=seq_len,
        )

        output = impl.forward(layer, query, key, value, cache, meta)

        assert output.shape == (seq_len, num_heads * D)
        assert not output.isnan().any()


# ── 3. Decode Forward (Full Pipeline) ────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDecodeForward:
    """Decode: block_table → gather → unpack → score → softmax → output."""

    @pytest.mark.parametrize("dtype", ["tq3", "rq3", "tq4", "rq4", "rq2"])
    def test_single_token_decode(self, dtype):
        """Single decode token — the bread-and-butter of serving."""
        D = 128
        num_heads = 4
        num_kv_heads = 4
        seq_len = 8  # context length
        block_size = 16

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads,
                                 kv_cache_dtype=dtype)
        packed_size = impl._packed_size
        cache = _make_cache(1, block_size, num_kv_heads, packed_size)

        # First: fill cache with KV data (simulating prefill)
        torch.manual_seed(42)
        key_ctx = torch.randn(seq_len, num_kv_heads, D,
                              dtype=torch.bfloat16, device="cuda")
        value_ctx = torch.randn(seq_len, num_kv_heads, D,
                                dtype=torch.bfloat16, device="cuda")

        slot_mapping_ctx = torch.arange(seq_len, device="cuda")
        impl.do_kv_cache_update(layer, key_ctx, value_ctx, cache, slot_mapping_ctx)

        # Now decode: 1 new token attending to seq_len cached tokens
        query = torch.randn(1, num_heads * D,
                            dtype=torch.bfloat16, device="cuda")
        # Decode doesn't use key/value args (reads from cache)
        key_new = torch.randn(1, num_kv_heads * D,
                              dtype=torch.bfloat16, device="cuda")
        value_new = torch.randn(1, num_kv_heads * D,
                                dtype=torch.bfloat16, device="cuda")

        block_table = torch.tensor([[0]], device="cuda")
        meta = _make_metadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=block_table,
            slot_mapping=torch.tensor([seq_len], device="cuda"),
            num_prefill=0,
            num_decode=1,
            max_seq_len=seq_len,
        )

        output = impl.forward(layer, query, key_new, value_new, cache, meta)

        assert output.shape == (1, num_heads * D)
        assert not output.isnan().any(), f"Decode output has NaN ({dtype})"
        assert output.abs().sum() > 0, f"Decode output is all zeros ({dtype})"

    def test_decode_multi_block(self):
        """Decode with context spanning multiple blocks."""
        D = 128
        num_heads = 2
        num_kv_heads = 2
        block_size = 16
        seq_len = 40  # spans 3 blocks

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        cache = _make_cache(3, block_size, num_kv_heads, packed_size)

        torch.manual_seed(42)
        key_ctx = torch.randn(seq_len, num_kv_heads, D,
                              dtype=torch.bfloat16, device="cuda")
        value_ctx = torch.randn(seq_len, num_kv_heads, D,
                                dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(seq_len, device="cuda")
        impl.do_kv_cache_update(layer, key_ctx, value_ctx, cache, slot_mapping)

        query = torch.randn(1, num_heads * D,
                            dtype=torch.bfloat16, device="cuda")
        key_new = torch.randn(1, num_kv_heads * D,
                              dtype=torch.bfloat16, device="cuda")
        value_new = torch.randn(1, num_kv_heads * D,
                                dtype=torch.bfloat16, device="cuda")

        block_table = torch.tensor([[0, 1, 2]], device="cuda")
        meta = _make_metadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=block_table,
            slot_mapping=torch.tensor([seq_len], device="cuda"),
            num_prefill=0,
            num_decode=1,
            max_seq_len=seq_len,
        )

        output = impl.forward(layer, query, key_new, value_new, cache, meta)

        assert output.shape == (1, num_heads * D)
        assert not output.isnan().any()

    def test_decode_gqa(self):
        """Decode with GQA: 8 query heads, 2 KV heads."""
        D = 128
        num_heads = 8
        num_kv_heads = 2
        block_size = 16
        seq_len = 12

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        cache = _make_cache(1, block_size, num_kv_heads, packed_size)

        torch.manual_seed(42)
        key_ctx = torch.randn(seq_len, num_kv_heads, D,
                              dtype=torch.bfloat16, device="cuda")
        value_ctx = torch.randn(seq_len, num_kv_heads, D,
                                dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(seq_len, device="cuda")
        impl.do_kv_cache_update(layer, key_ctx, value_ctx, cache, slot_mapping)

        query = torch.randn(1, num_heads * D,
                            dtype=torch.bfloat16, device="cuda")
        key_new = torch.randn(1, num_kv_heads * D,
                              dtype=torch.bfloat16, device="cuda")
        value_new = torch.randn(1, num_kv_heads * D,
                                dtype=torch.bfloat16, device="cuda")

        block_table = torch.tensor([[0]], device="cuda")
        meta = _make_metadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=block_table,
            slot_mapping=torch.tensor([seq_len], device="cuda"),
            num_prefill=0,
            num_decode=1,
            max_seq_len=seq_len,
        )

        output = impl.forward(layer, query, key_new, value_new, cache, meta)

        assert output.shape == (1, num_heads * D)
        assert not output.isnan().any()


# ── 4. Batch Decode (Variable Seq Lengths) ───────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestBatchDecode:
    """Batch decode with different sequence lengths per request."""

    @pytest.mark.parametrize("dtype", ["tq3", "rq3", "tq4", "rq4"])
    def test_variable_seq_lens(self, dtype):
        """3 requests with seq_lens [5, 12, 3] — the real serving pattern."""
        D = 128
        num_heads = 4
        num_kv_heads = 2
        block_size = 16
        batch_size = 3
        seq_lens_list = [5, 12, 3]
        max_seq_len = max(seq_lens_list)

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads,
                                 kv_cache_dtype=dtype)
        packed_size = impl._packed_size

        # Allocate enough blocks
        max_blocks = (max_seq_len + block_size - 1) // block_size
        total_blocks = sum((sl + block_size - 1) // block_size
                           for sl in seq_lens_list)
        cache = _make_cache(total_blocks, block_size, num_kv_heads, packed_size)

        # Fill cache for each request
        block_offset = 0
        block_table_list = []
        for req_i, sl in enumerate(seq_lens_list):
            n_blocks = (sl + block_size - 1) // block_size
            torch.manual_seed(42 + req_i)
            key_ctx = torch.randn(sl, num_kv_heads, D,
                                  dtype=torch.bfloat16, device="cuda")
            value_ctx = torch.randn(sl, num_kv_heads, D,
                                    dtype=torch.bfloat16, device="cuda")

            slot_mapping = torch.arange(sl, device="cuda") + block_offset * block_size
            impl.do_kv_cache_update(layer, key_ctx, value_ctx, cache, slot_mapping)

            blocks = list(range(block_offset, block_offset + n_blocks))
            blocks += [0] * (max_blocks - n_blocks)  # pad
            block_table_list.append(blocks)
            block_offset += n_blocks

        # Decode: 1 new token per request
        query = torch.randn(batch_size, num_heads * D,
                            dtype=torch.bfloat16, device="cuda")
        key_new = torch.randn(batch_size, num_kv_heads * D,
                              dtype=torch.bfloat16, device="cuda")
        value_new = torch.randn(batch_size, num_kv_heads * D,
                                dtype=torch.bfloat16, device="cuda")

        seq_lens = torch.tensor(seq_lens_list, device="cuda")
        block_table = torch.tensor(block_table_list, device="cuda")
        slot_mapping = torch.tensor([
            block_table_list[i][seq_lens_list[i] // block_size] * block_size
            + seq_lens_list[i] % block_size
            for i in range(batch_size)
        ], device="cuda")

        meta = _make_metadata(
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            num_prefill=0,
            num_decode=batch_size,
            max_seq_len=max_seq_len,
        )

        output = impl.forward(layer, query, key_new, value_new, cache, meta)

        assert output.shape == (batch_size, num_heads * D)
        assert not output.isnan().any(), "Batch decode output has NaN"
        # Each request should produce different output
        for i in range(batch_size):
            assert output[i].abs().sum() > 0, f"Request {i} output is zero"


# ── 5. Decode Quality (Score Correctness) ────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDecodeQuality:
    """Verify compressed decode output ≈ reference attention."""

    @pytest.mark.parametrize("dtype", ["tq3", "rq3", "tq4", "rq4"])
    def test_decode_vs_reference(self, dtype):
        """Compressed decode output cos≈ reference bmm attention."""
        D = 128
        num_heads = 4
        num_kv_heads = 4
        block_size = 16
        seq_len = 12

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads,
                                 kv_cache_dtype=dtype)
        packed_size = impl._packed_size
        cache = _make_cache(1, block_size, num_kv_heads, packed_size)

        # Fill cache
        torch.manual_seed(42)
        key_ctx = torch.randn(seq_len, num_kv_heads, D,
                              dtype=torch.bfloat16, device="cuda")
        value_ctx = torch.randn(seq_len, num_kv_heads, D,
                                dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(seq_len, device="cuda")
        impl.do_kv_cache_update(layer, key_ctx, value_ctx, cache, slot_mapping)

        # Decode
        query = torch.randn(1, num_heads * D,
                            dtype=torch.bfloat16, device="cuda")
        key_new = torch.randn(1, num_kv_heads * D,
                              dtype=torch.bfloat16, device="cuda")
        value_new = torch.randn(1, num_kv_heads * D,
                                dtype=torch.bfloat16, device="cuda")

        block_table = torch.tensor([[0]], device="cuda")
        meta = _make_metadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=block_table,
            slot_mapping=torch.tensor([seq_len], device="cuda"),
            num_prefill=0,
            num_decode=1,
            max_seq_len=seq_len,
        )

        output = impl.forward(layer, query, key_new, value_new, cache, meta)

        # Reference: naive bmm attention
        q = query.reshape(1, num_heads, D).float()
        k = key_ctx.float()
        v = value_ctx.float()

        if num_kv_heads < num_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

        scale = 1.0 / math.sqrt(D)
        scores = torch.einsum("bhd,shd->bhs", q, k) * scale
        weights = F.softmax(scores, dim=-1)
        ref_out = torch.einsum("bhs,shd->bhd", weights, v)
        ref_out = ref_out.reshape(1, num_heads * D)

        cos = F.cosine_similarity(output.float(), ref_out, dim=-1).mean()
        # Compressed attention is lossy — cos > 0.3 is reasonable
        assert cos > 0.3, f"Decode quality too low ({dtype}): cos={cos:.4f}"


# ── 6. Combined TQ KV + Archer Weights ──────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCombinedKVAndWeights:
    """Simulate combined Archer weight quant + MultiQuant KV cache."""

    def test_archer_weights_plus_tq_kv(self):
        """Archer-quantized linear + TQ3 KV cache in one forward."""
        D = 128
        num_heads = 2
        num_kv_heads = 2
        block_size = 16
        seq_len = 8

        # Create Archer-quantized weight layer
        from vllm.multiquant.weight_quant.config import ArcherConfig
        from vllm.multiquant.weight_quant.online_linear import (
            ArcherOnlineLinearMethod,
        )

        archer = ArcherOnlineLinearMethod(ArcherConfig(bits=3, method="tq"))
        weight_layer = nn.Module()
        torch.manual_seed(42)
        W = torch.randn(D, D, device="cuda")
        weight_layer.weight = nn.Parameter(W.clone(), requires_grad=False)
        archer._quantize_layer(weight_layer)

        # Simulate: x → Archer linear → use as query
        x = torch.randn(seq_len, D, dtype=torch.bfloat16, device="cuda")
        query_proj = archer.apply(weight_layer, x)  # (seq_len, D)
        query = query_proj.reshape(seq_len, num_heads, D // num_heads)
        # For test simplicity, just duplicate to get full head dim
        query_flat = query_proj  # (seq_len, D)

        # Set up KV cache
        impl, attn_layer = _make_impl(D // num_heads, num_heads=num_heads,
                                      num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        cache = _make_cache(1, block_size, num_kv_heads, packed_size)

        # Generate KV from Archer weights too
        torch.manual_seed(43)
        k_raw = torch.randn(seq_len, num_kv_heads, D // num_heads,
                            dtype=torch.bfloat16, device="cuda")
        v_raw = torch.randn(seq_len, num_kv_heads, D // num_heads,
                            dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(seq_len, device="cuda")
        impl.do_kv_cache_update(attn_layer, k_raw, v_raw, cache, slot_mapping)

        # Forward through attention (prefill)
        block_table = torch.tensor([[0]], device="cuda")
        meta = _make_metadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=block_table,
            slot_mapping=slot_mapping,
            num_prefill=seq_len,
            num_decode=0,
            max_seq_len=seq_len,
        )

        output = impl.forward(
            attn_layer, query_flat, k_raw.reshape(seq_len, -1),
            v_raw.reshape(seq_len, -1), cache, meta,
        )

        assert output.shape == (seq_len, D)
        assert not output.isnan().any(), "Combined output has NaN"
        assert output.abs().sum() > 0, "Combined output is all zeros"

    def test_archer_weights_plus_tq_kv_decode(self):
        """Archer weights + TQ3 KV cache: prefill then decode."""
        D = 128
        num_heads = 2
        num_kv_heads = 2
        block_size = 16
        seq_len = 8
        head_dim = D // num_heads

        # Create Archer-quantized weight layer
        from vllm.multiquant.weight_quant.config import ArcherConfig
        from vllm.multiquant.weight_quant.online_linear import (
            ArcherOnlineLinearMethod,
        )

        archer = ArcherOnlineLinearMethod(ArcherConfig(bits=3, method="tq"))
        weight_layer = nn.Module()
        torch.manual_seed(42)
        W = torch.randn(head_dim, head_dim, device="cuda")
        weight_layer.weight = nn.Parameter(W.clone(), requires_grad=False)
        archer._quantize_layer(weight_layer)

        # Set up KV cache attention
        impl, attn_layer = _make_impl(head_dim, num_heads=num_heads,
                                      num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        cache = _make_cache(2, block_size, num_kv_heads, packed_size)

        # Fill cache (prefill phase)
        torch.manual_seed(42)
        k_raw = torch.randn(seq_len, num_kv_heads, head_dim,
                            dtype=torch.bfloat16, device="cuda")
        v_raw = torch.randn(seq_len, num_kv_heads, head_dim,
                            dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(seq_len, device="cuda")
        impl.do_kv_cache_update(attn_layer, k_raw, v_raw, cache, slot_mapping)

        # Decode: use Archer weights to project query
        x = torch.randn(1, head_dim, dtype=torch.bfloat16, device="cuda")
        query_proj = archer.apply(weight_layer, x)  # Archer decompress + linear
        query_flat = query_proj.repeat(1, num_heads)  # (1, num_heads * head_dim)

        key_new = torch.randn(1, num_kv_heads * head_dim,
                              dtype=torch.bfloat16, device="cuda")
        value_new = torch.randn(1, num_kv_heads * head_dim,
                                dtype=torch.bfloat16, device="cuda")

        block_table = torch.tensor([[0]], device="cuda")
        meta = _make_metadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=block_table,
            slot_mapping=torch.tensor([seq_len], device="cuda"),
            num_prefill=0,
            num_decode=1,
            max_seq_len=seq_len,
        )

        output = impl.forward(
            attn_layer, query_flat, key_new, value_new, cache, meta,
        )

        assert output.shape == (1, num_heads * head_dim)
        assert not output.isnan().any(), "Decode with Archer weights has NaN"
        assert output.abs().sum() > 0, "Decode with Archer weights is zero"


# ── 7. Repeated Decode (Serving Simulation) ──────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestRepeatedDecode:
    """Simulate autoregressive generation: fill cache, then decode N tokens."""

    def test_autoregressive_10_tokens(self):
        """Generate 10 tokens autoregressively from 8-token context."""
        D = 128
        num_heads = 2
        num_kv_heads = 2
        block_size = 16
        context_len = 8
        gen_tokens = 10

        impl, layer = _make_impl(D, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
        packed_size = impl._packed_size
        max_blocks = (context_len + gen_tokens + block_size - 1) // block_size
        cache = _make_cache(max_blocks, block_size, num_kv_heads, packed_size)

        # Prefill context
        torch.manual_seed(42)
        key_ctx = torch.randn(context_len, num_kv_heads, D,
                              dtype=torch.bfloat16, device="cuda")
        value_ctx = torch.randn(context_len, num_kv_heads, D,
                                dtype=torch.bfloat16, device="cuda")

        slot_mapping = torch.arange(context_len, device="cuda")
        impl.do_kv_cache_update(layer, key_ctx, value_ctx, cache, slot_mapping)

        outputs = []
        current_len = context_len
        for step in range(gen_tokens):
            query = torch.randn(1, num_heads * D,
                                dtype=torch.bfloat16, device="cuda")
            key_new = torch.randn(1, num_kv_heads, D,
                                  dtype=torch.bfloat16, device="cuda")
            value_new = torch.randn(1, num_kv_heads, D,
                                    dtype=torch.bfloat16, device="cuda")

            # Write new KV to cache
            new_slot = torch.tensor([current_len], device="cuda")
            impl.do_kv_cache_update(layer, key_new, value_new, cache, new_slot)
            current_len += 1

            # Decode attending to all cached tokens
            n_blocks = (current_len + block_size - 1) // block_size
            bt = list(range(n_blocks))
            bt += [0] * (max_blocks - n_blocks)
            block_table = torch.tensor([bt], device="cuda")

            meta = _make_metadata(
                seq_lens=torch.tensor([current_len], device="cuda"),
                block_table=block_table,
                slot_mapping=new_slot,
                num_prefill=0,
                num_decode=1,
                max_seq_len=current_len,
            )

            output = impl.forward(layer, query,
                                  key_new.reshape(1, -1),
                                  value_new.reshape(1, -1),
                                  cache, meta)
            outputs.append(output.clone())

            assert output.shape == (1, num_heads * D)
            assert not output.isnan().any(), f"NaN at step {step}"

        # All outputs should be different (not degenerate)
        for i in range(1, gen_tokens):
            assert not torch.equal(outputs[i], outputs[0]), \
                f"Step {i} identical to step 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
