#!/usr/bin/env python3
"""Unit tests for Q3 TurboQuant custom attention backend.

8 levels — run ALL before attempting vLLM serve.
Tests run in container: python3 /opt/vllm-riy/tests/turboquant/test_tq_backend.py

Level 1: Config + Dtype registration
Level 2: Pack/Unpack round-trip
Level 3: Compressed score kernel (corr=1.0)
Level 4: Metadata builder
Level 5: do_kv_cache_update (pack into cache)
Level 6: forward decode (scores + softmax + V)
Level 7: forward prefill (naive causal)
Level 8: E2E prefill→decode simulation
"""
import math
import os
import sys
import struct

import torch
import torch.nn.functional as F


# ============================================================
# Test helpers
# ============================================================

def make_tq_matrices(D, seed=42):
    """Create Pi, S, centroids for TQ3."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    Pi = torch.randn(D, D, generator=gen, dtype=torch.float32)
    Pi, _ = torch.linalg.qr(Pi)
    Pi = Pi.contiguous()
    gen.manual_seed(seed + 1)
    S = torch.randn(D, D, generator=gen, dtype=torch.float32)
    sigma = 1.0 / math.sqrt(D)
    centroids = torch.tensor(
        [-1.51*sigma, -0.453*sigma, 0.453*sigma, 1.51*sigma],
        dtype=torch.float32)
    return Pi, S, centroids


def pack_vector_tq3(vec, Pi, S, centroids, D):
    """Pack one vector to TQ3 format. Returns uint8 tensor."""
    mse_bits = 2
    mse_bytes = (D * mse_bits + 7) // 8
    qjl_bytes = (D + 7) // 8

    x = vec.float()
    vn = x.norm(); xh = x / (vn + 1e-8)
    rot = xh @ Pi.T
    idx = (rot.unsqueeze(-1) - centroids).abs().argmin(dim=-1).to(torch.uint8)
    xm = centroids[idx.long()] @ Pi
    r = xh - xm; rn = r.norm()
    signs = (r @ S.T >= 0).to(torch.uint8)

    packed = torch.zeros(mse_bytes + qjl_bytes + 4, dtype=torch.uint8)
    # MSE indices (2-bit, 4 per byte)
    for j in range(0, D, 4):
        v = 0
        for k in range(min(4, D - j)):
            v |= (idx[j + k].item() & 0x3) << (k * 2)
        packed[j // 4] = v
    # Signs (1-bit, 8 per byte)
    for j in range(0, D, 8):
        v = 0
        for k in range(min(8, D - j)):
            v |= (signs[j + k].item() & 1) << k
        packed[mse_bytes + j // 8] = v
    # Norms
    packed[mse_bytes + qjl_bytes:mse_bytes + qjl_bytes + 2] = vn.half().reshape(1).view(torch.uint8)
    packed[mse_bytes + qjl_bytes + 2:mse_bytes + qjl_bytes + 4] = rn.half().reshape(1).view(torch.uint8)
    return packed


def unpack_vector_tq3(packed, Pi, S, centroids, D):
    """Unpack TQ3 packed vector → float32."""
    mse_bits = 2; mask = 3
    mse_bytes = (D * mse_bits + 7) // 8
    qjl_bytes = (D + 7) // 8
    correction = math.sqrt(math.pi / 2) / D

    idx = torch.zeros(D, dtype=torch.long)
    for j in range(D):
        b, k = j // 4, j % 4
        idx[j] = (packed[b].item() >> (k * 2)) & mask

    signs = torch.zeros(D, dtype=torch.float32)
    for j in range(D):
        b, k = j // 8, j % 8
        signs[j] = 1.0 if ((packed[mse_bytes + b].item() >> k) & 1) else -1.0

    no = mse_bytes + qjl_bytes
    vn = struct.unpack('e', bytes([packed[no].item(), packed[no+1].item()]))[0]
    rn = struct.unpack('e', bytes([packed[no+2].item(), packed[no+3].item()]))[0]

    c_idx = centroids[idx]
    xm = c_idx @ Pi
    xq = correction * rn * (signs @ S)
    return vn * (xm + xq)


def score_from_packed(q_rot, q_proj, packed, centroids, D, correction, attn_scale):
    """Compute attention score from packed K-vector."""
    mse_bits = 2; mask = 3
    mse_bytes = (D * mse_bits + 7) // 8
    qjl_bytes = (D + 7) // 8

    t1 = 0.0
    for b in range(mse_bytes):
        bv = packed[b].item()
        for k in range(4):
            j = b * 4 + k
            if j >= D: break
            idx = (bv >> (k * 2)) & mask
            t1 += q_rot[j].item() * centroids[idx].item()

    t2 = 0.0
    for b in range(qjl_bytes):
        bv = packed[mse_bytes + b].item()
        for k in range(8):
            j = b * 8 + k
            if j >= D: break
            sv = 1.0 if ((bv >> k) & 1) else -1.0
            t2 += q_proj[j].item() * sv

    no = mse_bytes + qjl_bytes
    vn = struct.unpack('e', bytes([packed[no].item(), packed[no+1].item()]))[0]
    rn = struct.unpack('e', bytes([packed[no+2].item(), packed[no+3].item()]))[0]
    return vn * (t1 + correction * rn * t2) * attn_scale


# ============================================================
# Level 1: Config + Dtype
# ============================================================
def test_level1():
    print("LEVEL 1: Config + Dtype")
    from vllm.config.cache import CacheDType
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.turboquant.config import TurboQuantConfig

    assert "tq3" in str(CacheDType)
    assert STR_DTYPE_TO_TORCH_DTYPE["tq3"] == torch.uint8
    assert hasattr(AttentionBackendEnum, "TURBOQUANT")

    for d, expected in [(64, 28), (128, 52), (256, 100)]:
        c = TurboQuantConfig.from_cache_dtype("tq3", d)
        assert c.key_packed_size == expected, f"D={d}: {c.key_packed_size} != {expected}"
    print("  PASSED\n")


# ============================================================
# Level 2: Pack/Unpack
# ============================================================
def test_level2():
    print("LEVEL 2: Pack/Unpack round-trip")
    D = 64
    Pi, S, centroids = make_tq_matrices(D)
    torch.manual_seed(42)

    for i in range(10):
        vec = torch.randn(D)
        packed = pack_vector_tq3(vec, Pi, S, centroids, D)
        assert packed.shape[0] == 28, f"packed size {packed.shape[0]} != 28"

        # Unpack
        recon = unpack_vector_tq3(packed, Pi, S, centroids, D)
        cos = F.cosine_similarity(vec.unsqueeze(0), recon.unsqueeze(0)).item()
        assert cos > 0.8, f"vec {i}: cos={cos}"

    # Pack size matches config
    from vllm.turboquant.config import TurboQuantConfig
    assert packed.shape[0] == TurboQuantConfig.from_cache_dtype("tq3", D).key_packed_size
    print("  PASSED\n")


# ============================================================
# Level 3: Compressed Score
# ============================================================
def test_level3():
    print("LEVEL 3: Compressed Score (Python reference)")
    D = 64
    Pi, S, centroids = make_tq_matrices(D)
    correction = math.sqrt(math.pi / 2) / D
    attn_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(42)

    for seq_len in [16, 64, 256]:
        keys = torch.randn(seq_len, D)
        query = torch.randn(D)
        q_rot = query @ Pi.T
        q_proj = query @ S.T

        ref_scores = []
        packed_scores = []
        for t in range(seq_len):
            packed = pack_vector_tq3(keys[t], Pi, S, centroids, D)
            ps = score_from_packed(q_rot, q_proj, packed, centroids, D, correction, attn_scale)
            packed_scores.append(ps)

            # Reference: from original key via TQ decompress
            recon = unpack_vector_tq3(packed, Pi, S, centroids, D)
            rs = (query @ recon).item() * attn_scale
            ref_scores.append(rs)

        ps_t = torch.tensor(packed_scores)
        rs_t = torch.tensor(ref_scores)
        corr = torch.corrcoef(torch.stack([ps_t, rs_t]))[0, 1].item()
        assert corr > 0.99, f"seq_len={seq_len}: corr={corr}"
        print(f"  seq_len={seq_len}: corr={corr:.6f}")
    print("  PASSED\n")


# ============================================================
# Level 4: Metadata Builder (mock)
# ============================================================
def test_level4():
    print("LEVEL 4: Metadata Builder")
    # Simulate what the builder needs to do
    batch_size = 2
    seq_lens = torch.tensor([32, 64])
    block_size = 16
    max_blocks = (seq_lens.max().item() + block_size - 1) // block_size

    block_table = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    for b in range(batch_size):
        n_blocks = (seq_lens[b].item() + block_size - 1) // block_size
        block_table[b, :n_blocks] = torch.arange(n_blocks) + b * 10  # fake physical blocks

    slot_mapping = torch.arange(batch_size, dtype=torch.int64)  # 1 decode token each

    # Verify block_table → slot_mapping consistency
    for b in range(batch_size):
        slot = slot_mapping[b].item()
        bi = slot // block_size
        bo = slot % block_size
        phys = block_table[b, bi].item()
        assert phys >= 0, f"Invalid block for batch {b}"

    print("  PASSED\n")


# ============================================================
# Level 5: do_kv_cache_update (pack into cache)
# ============================================================
def test_level5():
    print("LEVEL 5: do_kv_cache_update")
    D = 64
    packed_size = 28
    block_size = 16
    num_blocks = 4
    num_heads = 2
    Pi, S, centroids = make_tq_matrices(D)
    torch.manual_seed(42)

    kv_cache = torch.zeros(num_blocks, 2, block_size, num_heads, packed_size, dtype=torch.uint8)
    key = torch.randn(3, num_heads, D)
    value = torch.randn(3, num_heads, D)
    slot_mapping = torch.tensor([0, 1, 16])  # slots 0, 1 in block 0; slot 16 in block 1

    for i in range(3):
        slot = slot_mapping[i].item()
        bi, bo = slot // block_size, slot % block_size
        for h in range(num_heads):
            kp = pack_vector_tq3(key[i, h], Pi, S, centroids, D)
            kv_cache[bi, 0, bo, h, :len(kp)] = kp
            vp = pack_vector_tq3(value[i, h], Pi, S, centroids, D)
            kv_cache[bi, 1, bo, h, :len(vp)] = vp

    # Verify: unpack and compare
    for i in range(3):
        slot = slot_mapping[i].item()
        bi, bo = slot // block_size, slot % block_size
        for h in range(num_heads):
            k_recon = unpack_vector_tq3(kv_cache[bi, 0, bo, h], Pi, S, centroids, D)
            cos = F.cosine_similarity(key[i, h].unsqueeze(0), k_recon.unsqueeze(0)).item()
            assert cos > 0.8, f"token {i} head {h}: key cos={cos}"

    print("  PASSED\n")


# ============================================================
# Level 6: forward decode
# ============================================================
def test_level6():
    print("LEVEL 6: Forward decode")
    D = 64
    packed_size = 28
    block_size = 16
    num_heads = 2
    seq_len = 32
    num_blocks = (seq_len + block_size - 1) // block_size
    Pi, S, centroids = make_tq_matrices(D)
    correction = math.sqrt(math.pi / 2) / D
    attn_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(42)

    # Fill cache
    keys = torch.randn(seq_len, D)
    values = torch.randn(seq_len, D)
    kv_cache = torch.zeros(num_blocks, 2, block_size, num_heads, packed_size, dtype=torch.uint8)
    for t in range(seq_len):
        bi, bo = t // block_size, t % block_size
        for h in range(num_heads):
            kv_cache[bi, 0, bo, h] = pack_vector_tq3(keys[t], Pi, S, centroids, D)
            kv_cache[bi, 1, bo, h] = pack_vector_tq3(values[t], Pi, S, centroids, D)

    # Query
    query = torch.randn(D)
    q_rot = query @ Pi.T
    q_proj = query @ S.T

    # Decode: online softmax + V aggregation per head
    for h in range(num_heads):
        m_prev = float('-inf')
        d_prev = 0.0
        acc = torch.zeros(D)

        for t in range(seq_len):
            bi, bo = t // block_size, t % block_size
            score = score_from_packed(q_rot, q_proj, kv_cache[bi, 0, bo, h],
                                      centroids, D, correction, attn_scale)
            v_recon = unpack_vector_tq3(kv_cache[bi, 1, bo, h], Pi, S, centroids, D)

            m_new = max(m_prev, score)
            exp_prev = math.exp(m_prev - m_new)
            exp_cur = math.exp(score - m_new)
            d_new = d_prev * exp_prev + exp_cur
            acc = acc * exp_prev + exp_cur * v_recon
            m_prev, d_prev = m_new, d_new

        output_tq = acc / d_prev

        # Reference: decompress all → standard attention
        k_recon = torch.stack([unpack_vector_tq3(kv_cache[t // block_size, 0, t % block_size, h],
                                                  Pi, S, centroids, D) for t in range(seq_len)])
        v_recon = torch.stack([unpack_vector_tq3(kv_cache[t // block_size, 1, t % block_size, h],
                                                  Pi, S, centroids, D) for t in range(seq_len)])
        ref_scores = (query @ k_recon.T) * attn_scale
        ref_weights = F.softmax(ref_scores, dim=-1)
        ref_output = ref_weights @ v_recon

        cos = F.cosine_similarity(output_tq.unsqueeze(0), ref_output.unsqueeze(0)).item()
        assert cos > 0.99, f"head {h}: decode output cos={cos}"
        print(f"  head {h}: decode cos={cos:.6f}")
    print("  PASSED\n")


# ============================================================
# Level 7: forward prefill
# ============================================================
def test_level7():
    print("LEVEL 7: Forward prefill (naive causal)")
    D = 64
    num_heads = 2
    seq_len = 16
    torch.manual_seed(42)

    query = torch.randn(seq_len, num_heads, D)
    key = torch.randn(seq_len, num_heads, D)
    value = torch.randn(seq_len, num_heads, D)

    # Naive causal attention
    for h in range(num_heads):
        q = query[:, h]  # (seq, D)
        k = key[:, h]
        v = value[:, h]
        scores = (q @ k.T) / math.sqrt(D)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        out_naive = weights @ v

        # Reference: PyTorch SDPA
        out_ref = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True).squeeze(0)

        cos = F.cosine_similarity(out_naive.reshape(1, -1), out_ref.reshape(1, -1)).item()
        assert cos > 0.999, f"head {h}: prefill cos={cos}"
    print("  PASSED\n")


# ============================================================
# Level 8: E2E prefill → decode
# ============================================================
def test_level8():
    print("LEVEL 8: E2E prefill → decode")
    D = 64
    packed_size = 28
    block_size = 16
    num_heads = 2
    prefill_len = 32
    decode_steps = 8
    Pi, S, centroids = make_tq_matrices(D)
    correction = math.sqrt(math.pi / 2) / D
    attn_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(42)

    total_len = prefill_len + decode_steps
    num_blocks = (total_len + block_size - 1) // block_size
    kv_cache = torch.zeros(num_blocks, 2, block_size, num_heads, packed_size, dtype=torch.uint8)

    all_keys = torch.randn(total_len, D)
    all_values = torch.randn(total_len, D)

    # Prefill: pack first prefill_len tokens
    for t in range(prefill_len):
        bi, bo = t // block_size, t % block_size
        for h in range(num_heads):
            kv_cache[bi, 0, bo, h] = pack_vector_tq3(all_keys[t], Pi, S, centroids, D)
            kv_cache[bi, 1, bo, h] = pack_vector_tq3(all_values[t], Pi, S, centroids, D)

    # Decode: add tokens one by one
    for step in range(decode_steps):
        t = prefill_len + step
        bi, bo = t // block_size, t % block_size
        for h in range(num_heads):
            kv_cache[bi, 0, bo, h] = pack_vector_tq3(all_keys[t], Pi, S, centroids, D)
            kv_cache[bi, 1, bo, h] = pack_vector_tq3(all_values[t], Pi, S, centroids, D)

    # Final decode: query = last token, attend to all
    query = all_keys[total_len - 1]
    q_rot = query @ Pi.T
    q_proj = query @ S.T
    seq_len = total_len

    for h in range(num_heads):
        # TQ decode
        m_prev = float('-inf'); d_prev = 0.0; acc = torch.zeros(D)
        for t in range(seq_len):
            bi, bo = t // block_size, t % block_size
            score = score_from_packed(q_rot, q_proj, kv_cache[bi, 0, bo, h],
                                      centroids, D, correction, attn_scale)
            v_recon = unpack_vector_tq3(kv_cache[bi, 1, bo, h], Pi, S, centroids, D)
            m_new = max(m_prev, score)
            ep = math.exp(m_prev - m_new); ec = math.exp(score - m_new)
            d_new = d_prev * ep + ec
            acc = acc * ep + ec * v_recon
            m_prev, d_prev = m_new, d_new
        output_tq = acc / d_prev

        # BF16 reference
        ref_scores = (query @ all_keys[:seq_len].T) * attn_scale
        ref_weights = F.softmax(ref_scores, dim=-1)
        ref_output = ref_weights @ all_values[:seq_len]

        cos = F.cosine_similarity(output_tq.unsqueeze(0), ref_output.unsqueeze(0)).item()
        if step == decode_steps - 1:
            print(f"  head {h}: e2e cos={cos:.4f} (seq_len={seq_len})")
        assert cos > 0.85, f"head {h}: e2e cos={cos}"
    print("  PASSED\n")


# ============================================================
if __name__ == "__main__":
    print("\nTurboQuant Q3 Backend Unit Tests\n" + "=" * 50 + "\n")
    test_level1()
    test_level2()
    test_level3()
    test_level4()
    test_level5()
    test_level6()
    test_level7()
    test_level8()
    print("=" * 50)
    print("ALL 8 LEVELS PASSED")
    print("=" * 50)
