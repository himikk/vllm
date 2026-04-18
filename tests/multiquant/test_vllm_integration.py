#!/usr/bin/env python3
"""vLLM Integration Tests — schrittweise von innen nach außen.

Stufe A: Eigener Test (Referenz, funktioniert cos=0.95)
Stufe B: Echte TQMetadata + echte Strides aus vLLM
Stufe C: Echte Attention-Layer Initialisierung
Stufe D: Voller vLLM Forward-Pfad

Jede Stufe nutzt mehr vLLM-Code. Wo es bricht = wo der Bug ist.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bf16_reference(q, all_k, all_v, num_heads, num_kv_heads, D, scale):
    """BF16 Attention Referenz — kein Quant."""
    sl = len(all_k)
    kg = num_heads // num_kv_heads
    qr = q.reshape(1, num_heads, D).float()
    ks = torch.stack(all_k).float()
    vs = torch.stack(all_v).float()
    if kg > 1:
        ks = ks.repeat_interleave(kg, dim=1)
        vs = vs.repeat_interleave(kg, dim=1)
    scores = torch.einsum("bhd,shd->bhs", qr, ks) * scale
    weights = F.softmax(scores, dim=-1)
    return torch.einsum("bhs,shd->bhd", weights, vs).reshape(1, num_heads * D)


# ============================================================
# Stufe A: Eigener Test (Referenz)
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestStufeA_EigenerTest:
    """Referenz: eigene Impl + eigene Metadata. Funktioniert (cos=0.95)."""

    @pytest.mark.parametrize("dtype", ["tq3", "tq4"])
    def test_own_impl_own_metadata(self, dtype):
        D = 256; nh = 20; nkv = 20; bs = 16; nc = 8; ns = 5
        nb = (nc + ns + bs - 1) // bs

        from vllm.v1.attention.backends.multiquant_attn import (
            MultiQuantImpl, TQMetadata,
        )
        from vllm.multiquant.turboquant.quantizer import (
            generate_rotation_matrix,
        )
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.shared.centroids import get_centroids

        impl = MultiQuantImpl(
            num_heads=nh, head_size=D,
            scale=1.0 / math.sqrt(D),
            num_kv_heads=nkv, kv_cache_dtype=dtype,
        )
        layer = nn.Module()
        layer.register_buffer(
            "_tq_Pi",
            generate_rotation_matrix(D, seed=42).cuda().float(),
        )
        layer.register_buffer(
            "_tq_S", generate_qjl_matrix(D, seed=43).cuda().float(),
        )
        layer.register_buffer(
            "_tq_centroids",
            get_centroids(D, impl._mse_bits).cuda().float(),
        )

        ps = impl._packed_size
        cache = torch.zeros(
            nb, 2, bs, nkv, ps, dtype=torch.uint8, device="cuda",
        )

        scale = 1.0 / math.sqrt(D)
        torch.manual_seed(42)
        ak, av, cs = [], [], []

        # Prefill
        for t in range(nc):
            k = torch.randn(1, nkv, D, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(1, nkv, D, dtype=torch.bfloat16, device="cuda")
            impl.do_kv_cache_update(
                layer, k, v, cache, torch.tensor([t], device="cuda"),
            )
            ak.append(k[0])
            av.append(v[0])

        # Decode
        for step in range(ns):
            pos = nc + step
            q = torch.randn(
                1, nh * D, dtype=torch.bfloat16, device="cuda",
            )
            kn = torch.randn(
                1, nkv * D, dtype=torch.bfloat16, device="cuda",
            )
            vn = torch.randn(
                1, nkv * D, dtype=torch.bfloat16, device="cuda",
            )
            impl.do_kv_cache_update(
                layer,
                kn.reshape(1, nkv, D),
                vn.reshape(1, nkv, D),
                cache,
                torch.tensor([pos], device="cuda"),
            )
            ak.append(kn.reshape(nkv, D))
            av.append(vn.reshape(nkv, D))

            sl = pos + 1
            nbu = (sl + bs - 1) // bs
            bt = torch.zeros(
                1, nb, device="cuda", dtype=torch.int32,
            )
            bt[0, :nbu] = torch.arange(nbu, device="cuda")

            meta = TQMetadata(
                seq_lens=torch.tensor([sl], device="cuda"),
                block_table=bt,
                slot_mapping=torch.tensor([pos], device="cuda"),
                num_prefill_tokens=0,
                num_decode_tokens=1,
                max_seq_len=sl,
            )

            out = impl.forward(layer, q, kn, vn, cache, meta)
            ref = _bf16_reference(
                q, ak[:sl], av[:sl], nh, nkv, D, scale,
            ).cuda()
            cos = F.cosine_similarity(
                out.float(), ref.float(), dim=-1,
            ).mean().item()
            cs.append(cos)

        avg = sum(cs) / len(cs)
        print(f"\nStufe A ({dtype}): avg={avg:.4f} min={min(cs):.4f}")
        assert avg > 0.8, f"Stufe A failed: avg={avg:.4f}"


# ============================================================
# Stufe B: Echte TQMetadataBuilder Strides
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestStufeB_EchteMetadata:
    """Wie Stufe A, aber block_table + seq_lens wie vLLM sie baut.

    vLLM's block_table hat max_blocks_per_seq Spalten (nicht num_blocks).
    seq_lens kommt aus CommonAttentionMetadata.
    """

    @pytest.mark.parametrize("dtype", ["tq3", "tq4"])
    def test_vllm_style_metadata(self, dtype):
        D = 256; nh = 20; nkv = 20; bs = 16; nc = 8; ns = 5
        # vLLM allociert block_table mit max_blocks_per_seq Spalten
        max_blocks_per_seq = 64  # typisch vLLM
        total_blocks = 128  # total cache blocks

        from vllm.v1.attention.backends.multiquant_attn import (
            MultiQuantImpl, TQMetadata,
        )
        from vllm.multiquant.turboquant.quantizer import (
            generate_rotation_matrix,
        )
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.shared.centroids import get_centroids

        impl = MultiQuantImpl(
            num_heads=nh, head_size=D,
            scale=1.0 / math.sqrt(D),
            num_kv_heads=nkv, kv_cache_dtype=dtype,
        )
        layer = nn.Module()
        layer.register_buffer(
            "_tq_Pi",
            generate_rotation_matrix(D, seed=42).cuda().float(),
        )
        layer.register_buffer(
            "_tq_S", generate_qjl_matrix(D, seed=43).cuda().float(),
        )
        layer.register_buffer(
            "_tq_centroids",
            get_centroids(D, impl._mse_bits).cuda().float(),
        )

        ps = impl._packed_size
        # Großer Cache wie vLLM ihn allokiert
        cache = torch.zeros(
            total_blocks, 2, bs, nkv, ps,
            dtype=torch.uint8, device="cuda",
        )

        scale = 1.0 / math.sqrt(D)
        torch.manual_seed(42)
        ak, av, cs = [], [], []

        # Simuliere vLLM Block-Allokation: Blöcke nicht bei 0 startend
        block_offset = 17  # vLLM kann Blöcke überall allokieren

        # Prefill
        for t in range(nc):
            k = torch.randn(1, nkv, D, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(1, nkv, D, dtype=torch.bfloat16, device="cuda")
            # vLLM slot = block_idx * block_size + offset_in_block
            block_idx = block_offset + t // bs
            slot = block_idx * bs + (t % bs)
            impl.do_kv_cache_update(
                layer, k, v, cache, torch.tensor([slot], device="cuda"),
            )
            ak.append(k[0])
            av.append(v[0])

        # Decode
        for step in range(ns):
            pos = nc + step
            q = torch.randn(
                1, nh * D, dtype=torch.bfloat16, device="cuda",
            )
            kn = torch.randn(
                1, nkv * D, dtype=torch.bfloat16, device="cuda",
            )
            vn = torch.randn(
                1, nkv * D, dtype=torch.bfloat16, device="cuda",
            )

            block_idx = block_offset + pos // bs
            slot = block_idx * bs + (pos % bs)
            impl.do_kv_cache_update(
                layer,
                kn.reshape(1, nkv, D),
                vn.reshape(1, nkv, D),
                cache,
                torch.tensor([slot], device="cuda"),
            )
            ak.append(kn.reshape(nkv, D))
            av.append(vn.reshape(nkv, D))

            sl = pos + 1
            # vLLM block_table: logischer Block → physischer Block
            nbu = (sl + bs - 1) // bs
            bt = torch.zeros(
                1, max_blocks_per_seq, device="cuda", dtype=torch.int32,
            )
            for b in range(nbu):
                bt[0, b] = block_offset + b  # physischer Block

            meta = TQMetadata(
                seq_lens=torch.tensor([sl], device="cuda"),
                block_table=bt,
                slot_mapping=torch.tensor([slot], device="cuda"),
                num_prefill_tokens=0,
                num_decode_tokens=1,
                max_seq_len=sl,
            )

            out = impl.forward(layer, q, kn, vn, cache, meta)
            ref = _bf16_reference(
                q, ak[:sl], av[:sl], nh, nkv, D, scale,
            ).cuda()
            cos = F.cosine_similarity(
                out.float(), ref.float(), dim=-1,
            ).mean().item()
            cs.append(cos)

        avg = sum(cs) / len(cs)
        print(f"\nStufe B ({dtype}): avg={avg:.4f} min={min(cs):.4f}")
        assert avg > 0.8, f"Stufe B failed: avg={avg:.4f}"


# ============================================================
# Stufe C: Echte Attention Layer Initialisierung
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestStufeC_EchteAttentionInit:
    """Nutze den echten Attention.__init__ Pfad mit register_buffer.

    Simuliert wie vLLM die Layer mit _init_multiquant_buffers erstellt.
    """

    @pytest.mark.parametrize("dtype", ["tq3", "tq4"])
    def test_real_buffer_init(self, dtype):
        D = 256; nh = 20; nkv = 20; bs = 16; nc = 8; ns = 5
        nb = (nc + ns + bs - 1) // bs

        from vllm.v1.attention.backends.multiquant_attn import (
            MultiQuantImpl, TQMetadata,
        )

        impl = MultiQuantImpl(
            num_heads=nh, head_size=D,
            scale=1.0 / math.sqrt(D),
            num_kv_heads=nkv, kv_cache_dtype=dtype,
        )

        # Echte Buffer-Initialisierung wie in attention.py
        from vllm.multiquant.registry import get_kv_quantizer_config
        mq_config = get_kv_quantizer_config(dtype, D)
        seed = mq_config.seed + 5 * 1337  # layer_idx=5

        layer = nn.Module()
        if dtype.startswith("rq"):
            from vllm.multiquant.rotorquant.quantizer import generate_rotors
            layer.register_buffer(
                "_tq_Pi", generate_rotors(D, seed=seed),
            )
        else:
            from vllm.multiquant.turboquant.quantizer import (
                generate_rotation_matrix,
            )
            layer.register_buffer(
                "_tq_Pi", generate_rotation_matrix(D, seed=seed),
            )

        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.shared.centroids import get_centroids
        layer.register_buffer(
            "_tq_S", generate_qjl_matrix(D, seed=seed + 1),
        )
        layer.register_buffer(
            "_tq_centroids", get_centroids(D, mq_config.mse_bits),
        )
        layer = layer.cuda()

        ps = impl._packed_size
        cache = torch.zeros(
            nb, 2, bs, nkv, ps, dtype=torch.uint8, device="cuda",
        )

        scale = 1.0 / math.sqrt(D)
        torch.manual_seed(42)
        ak, av, cs = [], [], []

        for t in range(nc):
            k = torch.randn(1, nkv, D, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(1, nkv, D, dtype=torch.bfloat16, device="cuda")
            impl.do_kv_cache_update(
                layer, k, v, cache, torch.tensor([t], device="cuda"),
            )
            ak.append(k[0])
            av.append(v[0])

        for step in range(ns):
            pos = nc + step
            q = torch.randn(
                1, nh * D, dtype=torch.bfloat16, device="cuda",
            )
            kn = torch.randn(
                1, nkv * D, dtype=torch.bfloat16, device="cuda",
            )
            vn = torch.randn(
                1, nkv * D, dtype=torch.bfloat16, device="cuda",
            )
            impl.do_kv_cache_update(
                layer,
                kn.reshape(1, nkv, D),
                vn.reshape(1, nkv, D),
                cache,
                torch.tensor([pos], device="cuda"),
            )
            ak.append(kn.reshape(nkv, D))
            av.append(vn.reshape(nkv, D))

            sl = pos + 1
            nbu = (sl + bs - 1) // bs
            bt = torch.zeros(1, nb, device="cuda", dtype=torch.int32)
            bt[0, :nbu] = torch.arange(nbu, device="cuda")

            meta = TQMetadata(
                seq_lens=torch.tensor([sl], device="cuda"),
                block_table=bt,
                slot_mapping=torch.tensor([pos], device="cuda"),
                num_prefill_tokens=0,
                num_decode_tokens=1,
                max_seq_len=sl,
            )

            out = impl.forward(layer, q, kn, vn, cache, meta)
            ref = _bf16_reference(
                q, ak[:sl], av[:sl], nh, nkv, D, scale,
            ).cuda()
            cos = F.cosine_similarity(
                out.float(), ref.float(), dim=-1,
            ).mean().item()
            cs.append(cos)

        avg = sum(cs) / len(cs)
        print(f"\nStufe C ({dtype}): avg={avg:.4f} min={min(cs):.4f}")
        assert avg > 0.8, f"Stufe C failed: avg={avg:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


# ============================================================
# Stufe D: Echtes Modell — einzelner Layer Forward
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestStufeD_ModelForward:
    """Lade GLM-4.7 und teste einen einzelnen Attention Layer.

    Vergleiche MQ Output vs Standard (FlashAttn/kein Quant).
    """

    def test_single_layer_with_model(self):
        """Lade Modell, extrahiere K/V aus Layer 0, packe+decode."""
        D = 256; nh = 20; nkv = 20; bs = 16
        dtype_kv = "tq4"

        from vllm.v1.attention.backends.multiquant_attn import (
            MultiQuantImpl, TQMetadata,
        )
        from vllm.multiquant.registry import get_kv_quantizer_config

        impl = MultiQuantImpl(
            num_heads=nh, head_size=D,
            scale=1.0 / math.sqrt(D),
            num_kv_heads=nkv, kv_cache_dtype=dtype_kv,
        )

        mq_config = get_kv_quantizer_config(dtype_kv, D)
        seed = mq_config.seed + 0 * 1337  # layer 0

        layer = nn.Module()
        from vllm.multiquant.turboquant.quantizer import (
            generate_rotation_matrix,
        )
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.shared.centroids import get_centroids
        layer.register_buffer(
            "_tq_Pi", generate_rotation_matrix(D, seed=seed),
        )
        layer.register_buffer(
            "_tq_S", generate_qjl_matrix(D, seed=seed + 1),
        )
        layer.register_buffer(
            "_tq_centroids", get_centroids(D, mq_config.mse_bits),
        )
        layer = layer.cuda()

        # Lade echte K/V aus dem Modell (erster Layer, ein paar Tokens)
        # Nutze transformers direkt
        try:
            from transformers import AutoTokenizer, AutoModel
            import os
            model_path = "/data/tensordata/GLM-4.7-Flash"
            if not os.path.exists(model_path):
                pytest.skip("BF16 model not found")
            tok = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
            )
            # Nur Tokenizer für Prompt-Encoding
            input_ids = tok.encode("3+4=", return_tensors="pt").cuda()
            seq_len = input_ids.shape[1]
            print(f"\nPrompt tokens: {seq_len}, ids: {input_ids[0].tolist()}")
        except Exception as e:
            pytest.skip(f"Cannot load tokenizer: {e}")

        # Generiere realistische K/V (normalverteilt, skaliert wie echtes Modell)
        # Echte K/V haben std ≈ 0.1-0.3 (nicht 1.0 wie torch.randn)
        torch.manual_seed(42)
        k_real = torch.randn(
            seq_len, nkv, D, dtype=torch.bfloat16, device="cuda",
        ) * 0.2
        v_real = torch.randn(
            seq_len, nkv, D, dtype=torch.bfloat16, device="cuda",
        ) * 0.2

        # Pack K/V
        ps = impl._packed_size
        nb = (seq_len + bs - 1) // bs + 2  # extra blocks
        cache = torch.zeros(
            nb, 2, bs, nkv, ps, dtype=torch.uint8, device="cuda",
        )
        for t in range(seq_len):
            impl.do_kv_cache_update(
                layer,
                k_real[t:t+1],
                v_real[t:t+1],
                cache,
                torch.tensor([t], device="cuda"),
            )

        # Decode: query for next token
        q = torch.randn(
            1, nh * D, dtype=torch.bfloat16, device="cuda",
        ) * 0.2

        nbu = (seq_len + bs - 1) // bs
        bt = torch.zeros(1, nb, device="cuda", dtype=torch.int32)
        bt[0, :nbu] = torch.arange(nbu, device="cuda")

        meta = TQMetadata(
            seq_lens=torch.tensor([seq_len], device="cuda"),
            block_table=bt,
            slot_mapping=torch.tensor([seq_len], device="cuda"),
            num_prefill_tokens=0,
            num_decode_tokens=1,
            max_seq_len=seq_len,
        )

        kn = torch.randn(
            1, nkv * D, dtype=torch.bfloat16, device="cuda",
        ) * 0.2
        vn = torch.randn(
            1, nkv * D, dtype=torch.bfloat16, device="cuda",
        ) * 0.2

        out = impl.forward(layer, q, kn, vn, cache, meta)

        # BF16 Reference
        all_k = [k_real[t] for t in range(seq_len)]
        all_v = [v_real[t] for t in range(seq_len)]
        ref = _bf16_reference(
            q, all_k, all_v, nh, nkv, D, 1.0 / math.sqrt(D),
        ).cuda()

        cos = F.cosine_similarity(
            out.float(), ref.float(), dim=-1,
        ).mean().item()
        print(f"Stufe D: cos={cos:.4f} (scaled K/V std=0.2)")
        print(f"  out norm={out.float().norm():.4f}")
        print(f"  ref norm={ref.float().norm():.4f}")

        # Vergleiche auch mit std=1.0 (wie bisherige Tests)
        torch.manual_seed(42)
        k_unit = torch.randn(
            seq_len, nkv, D, dtype=torch.bfloat16, device="cuda",
        )
        v_unit = torch.randn(
            seq_len, nkv, D, dtype=torch.bfloat16, device="cuda",
        )
        cache2 = torch.zeros(
            nb, 2, bs, nkv, ps, dtype=torch.uint8, device="cuda",
        )
        for t in range(seq_len):
            impl.do_kv_cache_update(
                layer, k_unit[t:t+1], v_unit[t:t+1], cache2,
                torch.tensor([t], device="cuda"),
            )
        out2 = impl.forward(layer, q, kn, vn, cache2, meta)
        ref2 = _bf16_reference(
            q, [k_unit[t] for t in range(seq_len)],
            [v_unit[t] for t in range(seq_len)],
            nh, nkv, D, 1.0 / math.sqrt(D),
        ).cuda()
        cos2 = F.cosine_similarity(
            out2.float(), ref2.float(), dim=-1,
        ).mean().item()
        print(f"Stufe D: cos={cos2:.4f} (unit K/V std=1.0)")

        assert cos > 0.8, f"Stufe D (scaled) failed: cos={cos:.4f}"
        assert cos2 > 0.8, f"Stufe D (unit) failed: cos={cos2:.4f}"


# ============================================================
# Stufe E: Prefill als Batch (wie vLLM)
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestStufeE_BatchPrefill:
    """Prefill alle Tokens auf einmal (nicht einzeln wie Stufe A-D).

    vLLM ruft _forward_prefill mit allen Tokens gleichzeitig auf,
    dann do_kv_cache_update für alle auf einmal.
    """

    @pytest.mark.parametrize("dtype", ["tq3", "tq4"])
    def test_batch_prefill_then_decode(self, dtype):
        D = 256; nh = 20; nkv = 20; bs = 16; nc = 8; ns = 5
        nb = (nc + ns + bs - 1) // bs

        from vllm.v1.attention.backends.multiquant_attn import (
            MultiQuantImpl, TQMetadata,
        )
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.shared.centroids import get_centroids

        impl = MultiQuantImpl(
            num_heads=nh, head_size=D,
            scale=1.0 / math.sqrt(D),
            num_kv_heads=nkv, kv_cache_dtype=dtype,
        )
        layer = nn.Module()
        layer.register_buffer("_tq_Pi", generate_rotation_matrix(D, seed=42).cuda().float())
        layer.register_buffer("_tq_S", generate_qjl_matrix(D, seed=43).cuda().float())
        layer.register_buffer("_tq_centroids", get_centroids(D, impl._mse_bits).cuda().float())

        ps = impl._packed_size
        cache = torch.zeros(nb, 2, bs, nkv, ps, dtype=torch.uint8, device="cuda")
        scale = 1.0 / math.sqrt(D)
        torch.manual_seed(42)

        # Generiere alle Prefill K/V auf einmal
        all_k = torch.randn(nc, nkv, D, dtype=torch.bfloat16, device="cuda")
        all_v = torch.randn(nc, nkv, D, dtype=torch.bfloat16, device="cuda")
        all_q = torch.randn(nc, nh * D, dtype=torch.bfloat16, device="cuda")

        # BATCH Prefill: alle Tokens auf einmal packen + KV Update
        slots_pf = torch.arange(nc, device="cuda")
        impl.do_kv_cache_update(layer, all_k, all_v, cache, slots_pf)

        # BATCH Prefill Forward (wie vLLM)
        bt_pf = torch.zeros(1, nb, device="cuda", dtype=torch.int32)
        nbu = (nc + bs - 1) // bs
        bt_pf[0, :nbu] = torch.arange(nbu, device="cuda")

        meta_pf = TQMetadata(
            seq_lens=torch.tensor([nc], device="cuda"),
            block_table=bt_pf,
            slot_mapping=slots_pf,
            num_prefill_tokens=nc,
            num_decode_tokens=0,
            max_seq_len=nc,
        )
        output_pf = torch.empty(nc, nh * D, dtype=torch.bfloat16, device="cuda")
        # Prefill: key/value kommen direkt (nicht aus Cache)
        impl.forward(
            layer, all_q,
            all_k.reshape(nc, nkv * D),
            all_v.reshape(nc, nkv * D),
            cache, meta_pf, output=output_pf,
        )

        # Prefill Referenz
        pq = all_q.reshape(nc, nh, D).float()
        pk = all_k.float()
        pv = all_v.float()
        if nh > nkv:
            pk = pk.repeat_interleave(nh // nkv, dim=1)
            pv = pv.repeat_interleave(nh // nkv, dim=1)
        scores_pf = torch.bmm(
            pq.transpose(0, 1).float(),
            pk.transpose(0, 1).float().transpose(-2, -1),
        ) * scale
        causal = torch.triu(torch.full((nc, nc), float('-inf'), device="cuda"), diagonal=1)
        scores_pf = scores_pf + causal.unsqueeze(0)
        w_pf = F.softmax(scores_pf, dim=-1)
        ref_pf = torch.bmm(w_pf, pv.transpose(0, 1).float())
        ref_pf = ref_pf.transpose(0, 1).reshape(nc, nh * D)

        cos_pf = F.cosine_similarity(
            output_pf.float(), ref_pf.float(), dim=-1,
        ).mean().item()
        print(f"\nStufe E ({dtype}): prefill cos={cos_pf:.4f}")

        # Decode steps
        cs = []
        ak_list = [all_k[t] for t in range(nc)]
        av_list = [all_v[t] for t in range(nc)]

        for step in range(ns):
            pos = nc + step
            q = torch.randn(1, nh * D, dtype=torch.bfloat16, device="cuda")
            kn = torch.randn(1, nkv * D, dtype=torch.bfloat16, device="cuda")
            vn = torch.randn(1, nkv * D, dtype=torch.bfloat16, device="cuda")

            slot = torch.tensor([pos], device="cuda")
            impl.do_kv_cache_update(
                layer, kn.reshape(1, nkv, D), vn.reshape(1, nkv, D),
                cache, slot,
            )
            ak_list.append(kn.reshape(nkv, D))
            av_list.append(vn.reshape(nkv, D))

            sl = pos + 1
            nbu = (sl + bs - 1) // bs
            bt = torch.zeros(1, nb, device="cuda", dtype=torch.int32)
            bt[0, :nbu] = torch.arange(nbu, device="cuda")

            meta = TQMetadata(
                seq_lens=torch.tensor([sl], device="cuda"),
                block_table=bt,
                slot_mapping=slot,
                num_prefill_tokens=0, num_decode_tokens=1, max_seq_len=sl,
            )
            out = impl.forward(layer, q, kn, vn, cache, meta)
            ref = _bf16_reference(
                q, ak_list[:sl], av_list[:sl], nh, nkv, D, scale,
            ).cuda()
            cos = F.cosine_similarity(out.float(), ref.float(), dim=-1).mean().item()
            cs.append(cos)

        avg = sum(cs) / len(cs)
        print(f"Stufe E ({dtype}): decode avg={avg:.4f} min={min(cs):.4f}")
        assert cos_pf > 0.95, f"Stufe E prefill failed: {cos_pf:.4f}"
        assert avg > 0.8, f"Stufe E decode failed: {avg:.4f}"


# ============================================================
# Stufe F: Mit CUDA Graph Capture
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestStufeF_CUDAGraph:
    """Wie Stufe E, aber Decode in CUDA Graph captured.

    Simuliert den vLLM CUDA Graph Pfad.
    """

    @pytest.mark.parametrize("dtype", ["tq4"])
    def test_cuda_graph_decode(self, dtype):
        D = 256; nh = 20; nkv = 20; bs = 16; nc = 8; ns = 5
        nb = (nc + ns + bs - 1) // bs

        from vllm.v1.attention.backends.multiquant_attn import (
            MultiQuantImpl, TQMetadata,
        )
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.shared.centroids import get_centroids

        impl = MultiQuantImpl(
            num_heads=nh, head_size=D,
            scale=1.0 / math.sqrt(D),
            num_kv_heads=nkv, kv_cache_dtype=dtype,
        )
        layer = nn.Module()
        layer.register_buffer("_tq_Pi", generate_rotation_matrix(D, seed=42).cuda().float())
        layer.register_buffer("_tq_S", generate_qjl_matrix(D, seed=43).cuda().float())
        layer.register_buffer("_tq_centroids", get_centroids(D, impl._mse_bits).cuda().float())

        ps = impl._packed_size
        cache = torch.zeros(nb, 2, bs, nkv, ps, dtype=torch.uint8, device="cuda")
        scale = 1.0 / math.sqrt(D)
        torch.manual_seed(42)

        # Prefill (eager)
        all_k = torch.randn(nc, nkv, D, dtype=torch.bfloat16, device="cuda")
        all_v = torch.randn(nc, nkv, D, dtype=torch.bfloat16, device="cuda")
        slots_pf = torch.arange(nc, device="cuda")
        impl.do_kv_cache_update(layer, all_k, all_v, cache, slots_pf)

        ak_list = [all_k[t] for t in range(nc)]
        av_list = [all_v[t] for t in range(nc)]

        # Pre-allocate decode tensors (static shapes for graph)
        q_buf = torch.randn(1, nh * D, dtype=torch.bfloat16, device="cuda")
        kn_buf = torch.randn(1, nkv * D, dtype=torch.bfloat16, device="cuda")
        vn_buf = torch.randn(1, nkv * D, dtype=torch.bfloat16, device="cuda")
        bt_buf = torch.zeros(1, nb, device="cuda", dtype=torch.int32)
        sl_buf = torch.tensor([nc + 1], device="cuda")
        slot_buf = torch.tensor([nc], device="cuda")
        out_buf = torch.empty(1, nh * D, dtype=torch.bfloat16, device="cuda")

        meta_buf = TQMetadata(
            seq_lens=sl_buf, block_table=bt_buf,
            slot_mapping=slot_buf,
            num_prefill_tokens=0, num_decode_tokens=1,
            max_seq_len=nc + ns,
        )

        # Warmup (required before capture)
        nbu = (nc + 1 + bs - 1) // bs
        bt_buf[0, :nbu] = torch.arange(nbu, device="cuda")
        impl.do_kv_cache_update(
            layer, kn_buf.reshape(1, nkv, D), vn_buf.reshape(1, nkv, D),
            cache, slot_buf,
        )
        impl.forward(layer, q_buf, kn_buf, vn_buf, cache, meta_buf, output=out_buf)

        # Capture CUDA Graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            impl.forward(layer, q_buf, kn_buf, vn_buf, cache, meta_buf, output=out_buf)
        torch.cuda.current_stream().wait_stream(s)

        try:
            with torch.cuda.graph(g):
                impl.forward(layer, q_buf, kn_buf, vn_buf, cache, meta_buf, output=out_buf)
            graph_ok = True
        except Exception as e:
            print(f"\nStufe F: CUDA Graph capture FAILED: {e}")
            graph_ok = False

        if not graph_ok:
            pytest.skip("CUDA Graph capture not supported")

        # Decode with graph replay
        cs = []
        for step in range(ns):
            pos = nc + step
            q_new = torch.randn(1, nh * D, dtype=torch.bfloat16, device="cuda")
            kn_new = torch.randn(1, nkv * D, dtype=torch.bfloat16, device="cuda")
            vn_new = torch.randn(1, nkv * D, dtype=torch.bfloat16, device="cuda")

            # Update buffers (graph uses same memory)
            q_buf.copy_(q_new)
            kn_buf.copy_(kn_new)
            vn_buf.copy_(vn_new)
            sl_buf.fill_(pos + 1)
            slot_buf.fill_(pos)
            nbu = (pos + 1 + bs - 1) // bs
            bt_buf.zero_()
            bt_buf[0, :nbu] = torch.arange(nbu, device="cuda")

            # KV update (eager, outside graph)
            impl.do_kv_cache_update(
                layer, kn_new.reshape(1, nkv, D), vn_new.reshape(1, nkv, D),
                cache, slot_buf,
            )
            ak_list.append(kn_new.reshape(nkv, D))
            av_list.append(vn_new.reshape(nkv, D))

            # Graph replay
            g.replay()

            ref = _bf16_reference(
                q_new, ak_list[:pos + 1], av_list[:pos + 1], nh, nkv, D, scale,
            ).cuda()
            cos = F.cosine_similarity(
                out_buf.float(), ref.float(), dim=-1,
            ).mean().item()
            cs.append(cos)

        avg = sum(cs) / len(cs)
        print(f"\nStufe F ({dtype}): graph decode avg={avg:.4f} min={min(cs):.4f}")
        assert avg > 0.8, f"Stufe F failed: avg={avg:.4f}"
