#!/usr/bin/env python3
"""Testgetriebene Math-Bug Analyse — Stufe 1-4.

Stufe 1: Pure Python Pack→Decode (kein CUDA Kernel)
Stufe 2: CUDA Kernel vs Python Referenz
Stufe 3: vLLM forward() Simulation (1 Layer)
Stufe 4: Multi-Layer Akkumulation (47 Layer)

Läuft im Container mit gemounteten Dateien.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ──────────────────────────────────────────────

def _make_impl(D, num_heads, num_kv_heads, kv_cache_dtype="tq3"):
    from vllm.v1.attention.backends.multiquant_attn import MultiQuantImpl
    impl = MultiQuantImpl(
        num_heads=num_heads, head_size=D,
        scale=1.0 / math.sqrt(D),
        num_kv_heads=num_kv_heads,
        kv_cache_dtype=kv_cache_dtype,
    )
    layer = nn.Module()
    is_rq = kv_cache_dtype.startswith("rq")
    if is_rq:
        from vllm.multiquant.rotorquant.quantizer import generate_rotors
        Pi = generate_rotors(D, seed=42).float()
    else:
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        Pi = generate_rotation_matrix(D, seed=42).float()
    from vllm.multiquant.shared.qjl import generate_qjl_matrix
    from vllm.multiquant.shared.centroids import get_centroids
    S = generate_qjl_matrix(D, seed=43).float()
    centroids = get_centroids(D, impl._mse_bits).float()
    layer.register_buffer("_tq_Pi", Pi)
    layer.register_buffer("_tq_S", S)
    layer.register_buffer("_tq_centroids", centroids)
    return impl, layer


def _make_metadata(seq_lens, block_table, slot_mapping,
                   num_prefill=0, num_decode=0, max_seq_len=0):
    from vllm.v1.attention.backends.multiquant_attn import TQMetadata
    return TQMetadata(
        seq_lens=seq_lens, block_table=block_table,
        slot_mapping=slot_mapping,
        num_prefill_tokens=num_prefill,
        num_decode_tokens=num_decode,
        max_seq_len=max_seq_len,
    )


def _bf16_attention(q, all_k, all_v, num_heads, num_kv_heads, D, scale):
    """BF16 reference attention — no quantization."""
    seq_len = len(all_k)
    kv_group = num_heads // num_kv_heads
    q_ref = q.reshape(1, num_heads, D).float()
    k_stack = torch.stack(all_k).float()  # [sl, Hkv, D]
    v_stack = torch.stack(all_v).float()
    if kv_group > 1:
        k_stack = k_stack.repeat_interleave(kv_group, dim=1)
        v_stack = v_stack.repeat_interleave(kv_group, dim=1)
    scores = torch.einsum("bhd,shd->bhs", q_ref, k_stack) * scale
    weights = F.softmax(scores, dim=-1)
    ref_out = torch.einsum("bhs,shd->bhd", weights, v_stack)
    return ref_out.reshape(1, num_heads * D)


# ── Stufe 1: Pure Python Round-Trip ─────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPurePythonRoundTrip:
    """Pack→Decode mit Python-only (kein CUDA Kernel).

    Testet ob der Algorithmus korrekt ist — unabhängig von Kernels.
    """

    @pytest.mark.parametrize("D", [128, 256])
    @pytest.mark.parametrize("num_heads", [4, 20])
    def test_python_pack_decode(self, D, num_heads):
        """Python _pack_batch → Python decode reference → cos vs BF16."""
        num_kv_heads = num_heads  # GLM-4.7: no GQA
        block_size = 16
        seq_len = 32
        mse_bits = 2  # TQ3

        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.shared.centroids import get_centroids
        from vllm.multiquant.shared.bitpack import pack_vectors_batched

        Pi = generate_rotation_matrix(D, seed=42).cuda().float()
        S = generate_qjl_matrix(D, seed=43).cuda().float()
        centroids = get_centroids(D, mse_bits).cuda().float()

        torch.manual_seed(42)
        scale = 1.0 / math.sqrt(D)
        correction = math.sqrt(math.pi / 2) / D

        # Fill KV cache with packed vectors
        num_blocks = (seq_len + block_size - 1) // block_size
        packed_size = (D * mse_bits + 7) // 8 + (D + 7) // 8 + 4
        cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads,
                            packed_size, dtype=torch.uint8, device="cuda")

        all_k_raw = []
        all_v_raw = []
        for t in range(seq_len):
            k = torch.randn(num_kv_heads, D, device="cuda")
            v = torch.randn(num_kv_heads, D, device="cuda")
            all_k_raw.append(k)
            all_v_raw.append(v)

            # Pack K and V
            for is_v, vec in [(0, k), (1, v)]:
                x = vec.float()
                vn = x.norm(dim=-1)
                xh = x / (vn.unsqueeze(-1) + 1e-8)
                rot = xh @ Pi.T
                idx = (rot.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
                xm = centroids[idx] @ Pi
                r = xh - xm
                rn = r.norm(dim=-1)
                signs = torch.sign(r @ S.T)
                signs[signs == 0] = 1.0
                packed = pack_vectors_batched(idx, signs, vn, rn, D, mse_bits)
                bi = t // block_size
                bo = t % block_size
                cache[bi, is_v, bo, :, :packed_size] = packed

        # Decode: use Python reference from test_fused_decode.py
        q = torch.randn(1, num_heads, D, device="cuda", dtype=torch.float32)
        block_table = torch.arange(num_blocks, device="cuda").unsqueeze(0).int()
        seq_lens_t = torch.tensor([seq_len], device="cuda", dtype=torch.int32)

        from tests.multiquant.test_fused_decode import _python_decode_reference
        mq_out = _python_decode_reference(
            q, cache, Pi, S, centroids, block_table, seq_lens_t,
            scale, block_size, num_kv_heads, mse_bits, correction,
        )

        # BF16 reference
        ref_out = _bf16_attention(
            q.reshape(1, num_heads * D),
            all_k_raw, all_v_raw,
            num_heads, num_kv_heads, D, scale,
        ).cuda()

        cos = F.cosine_similarity(
            mq_out.reshape(1, -1).float(),
            ref_out.reshape(1, -1).float(),
            dim=-1,
        ).item()
        print(f"\nStufe 1: D={D} H={num_heads}: cos={cos:.4f}")
        assert cos > 0.7, f"Python round-trip failed: cos={cos:.4f}"


# ── Stufe 3: vLLM Forward Simulation (1 Layer) ──────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSingleLayerForward:
    """Simuliert vLLM forward() mit Prefill + Decode + KV Cache.

    GLM-4.7 Dimensionen: D=256, Hq=20, Hkv=20.
    """

    @pytest.mark.parametrize("D", [128, 256])
    @pytest.mark.parametrize("num_heads", [4, 20])
    def test_forward_prefill_then_decode(self, D, num_heads):
        num_kv_heads = num_heads
        block_size = 16
        num_context = 8
        num_decode_steps = 10
        total = num_context + num_decode_steps
        num_blocks = (total + block_size - 1) // block_size

        impl, layer = _make_impl(D, num_heads, num_kv_heads, "tq3")
        # Move layer to CUDA
        layer = layer.cuda()
        packed_size = impl._packed_size
        cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads,
                            packed_size, dtype=torch.uint8, device="cuda")

        scale = 1.0 / math.sqrt(D)
        torch.manual_seed(42)

        all_k = []
        all_v = []

        # Prefill
        for t in range(num_context):
            k = torch.randn(1, num_kv_heads, D, dtype=torch.bfloat16,
                            device="cuda")
            v = torch.randn(1, num_kv_heads, D, dtype=torch.bfloat16,
                            device="cuda")
            slot = torch.tensor([t], device="cuda")
            impl.do_kv_cache_update(layer, k, v, cache, slot)
            all_k.append(k.squeeze(0))
            all_v.append(v.squeeze(0))

        # Decode steps
        cos_sims = []
        for step in range(num_decode_steps):
            pos = num_context + step
            q = torch.randn(1, num_heads * D, dtype=torch.bfloat16,
                            device="cuda")
            kn = torch.randn(1, num_kv_heads * D, dtype=torch.bfloat16,
                             device="cuda")
            vn = torch.randn(1, num_kv_heads * D, dtype=torch.bfloat16,
                             device="cuda")

            slot = torch.tensor([pos], device="cuda")
            impl.do_kv_cache_update(
                layer,
                kn.reshape(1, num_kv_heads, D),
                vn.reshape(1, num_kv_heads, D),
                cache, slot,
            )
            all_k.append(kn.reshape(num_kv_heads, D))
            all_v.append(vn.reshape(num_kv_heads, D))

            sl = pos + 1
            nbu = (sl + block_size - 1) // block_size
            bt = torch.zeros(1, num_blocks, device="cuda", dtype=torch.int32)
            bt[0, :nbu] = torch.arange(nbu, device="cuda")

            meta = _make_metadata(
                seq_lens=torch.tensor([sl], device="cuda"),
                block_table=bt,
                slot_mapping=slot,
                num_prefill=0, num_decode=1, max_seq_len=sl,
            )

            output = impl.forward(layer, q, kn, vn, cache, meta)

            ref = _bf16_attention(
                q, all_k[:sl], all_v[:sl],
                num_heads, num_kv_heads, D, scale,
            ).cuda()

            cos = F.cosine_similarity(
                output.float(), ref.float(), dim=-1
            ).mean().item()
            cos_sims.append(cos)

        avg = sum(cos_sims) / len(cos_sims)
        mn = min(cos_sims)
        print(f"\nStufe 3: D={D} H={num_heads}: avg={avg:.4f} min={mn:.4f}")
        print(f"  per-step: {['%.3f' % c for c in cos_sims]}")
        assert avg > 0.5, f"Forward simulation failed: avg={avg:.4f}"
        assert mn > 0.2, f"Forward simulation min too low: {mn:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
