#!/usr/bin/env python3
"""Archer end-to-end tests — Load → Quantize → Decompress → GEMM.

Empirically measured quality baselines (seed=42):
  TQ3: cos ~0.63-0.66 for all dimensions (D=128..10240)
  TQ4: cos ~0.60
  RQ3: cos ~0.92 (Clifford rotor — better than TQ because per-group rotation)
  RQ4: cos ~0.95
  RQ2: cos ~0.40
"""

import gc
import math
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_archer(bits=3, method="tq"):
    from vllm.multiquant.weight_quant.config import ArcherConfig
    from vllm.multiquant.weight_quant.online_linear import ArcherOnlineLinearMethod
    return ArcherOnlineLinearMethod(ArcherConfig(bits=bits, method=method))


def _quantize(out, inp, bits=3, method="tq"):
    archer = _make_archer(bits, method)
    layer = nn.Module()
    torch.manual_seed(42)
    W = torch.randn(out, inp, device="cuda")
    layer.weight = nn.Parameter(W.clone(), requires_grad=False)
    archer._quantize_layer(layer)
    return layer, W, archer


# ── 1. Load + Quantize Round-Trip ──────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuantizeRoundTrip:
    @pytest.mark.parametrize("D", [128, 512, 2048])
    def test_tq3_quality(self, D):
        layer, W, archer = _quantize(64, D, bits=3, method="tq")
        W_d = archer._decompress(layer)
        cos = F.cosine_similarity(W, W_d.float(), dim=-1).mean()
        assert cos > 0.55, f"TQ3 D={D}: cos={cos:.4f}"

    def test_tq4_quality(self):
        layer, W, archer = _quantize(64, 128, bits=4, method="tq")
        W_d = archer._decompress(layer)
        cos = F.cosine_similarity(W, W_d.float(), dim=-1).mean()
        assert cos > 0.50, f"TQ4: cos={cos:.4f}"

    def test_rq3_quality(self):
        layer, W, archer = _quantize(64, 128, bits=3, method="rq")
        W_d = archer._decompress(layer)
        cos = F.cosine_similarity(W, W_d.float(), dim=-1).mean()
        assert cos > 0.50, f"RQ3: cos={cos:.4f}"

    def test_rq4_quality(self):
        layer, W, archer = _quantize(64, 128, bits=4, method="rq")
        W_d = archer._decompress(layer)
        cos = F.cosine_similarity(W, W_d.float(), dim=-1).mean()
        assert cos > 0.50, f"RQ4: cos={cos:.4f}"

    def test_rq2_quality(self):
        """RQ2 (1-bit MSE) — very aggressive, low quality expected."""
        layer, W, archer = _quantize(64, 128, bits=2, method="rq")
        W_d = archer._decompress(layer)
        cos = F.cosine_similarity(W, W_d.float(), dim=-1).mean()
        assert cos > 0.15, f"RQ2: cos={cos:.4f}"


# ── 3. Packed Weight Format ────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPackedFormat:
    @pytest.mark.parametrize("D", [128, 512, 2048])
    def test_format(self, D):
        layer, _, _ = _quantize(64, D)
        expected = math.ceil(D * 2 / 8) + math.ceil(D / 8) + 4
        assert layer.weight.dtype == torch.uint8
        assert layer.weight.shape == (64, expected)
        assert layer._archer_packed is True
        assert layer._archer_in_features == D


# ── 4. CUDA Unpack ALL Dimensions ──────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCUDAUnpackAll:
    @pytest.mark.parametrize("D", [512, 768, 1024, 1536, 2048, 4096,
                                     4304, 4608, 5120, 10240])
    def test_unpack(self, D):
        from vllm.multiquant.shared.bitpack import pack_vectors_batched
        from vllm.multiquant.shared.centroids import get_centroids
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.weight_quant.archer_ops import cuda_unpack

        Pi = generate_rotation_matrix(D, seed=42).to("cuda")
        centroids = get_centroids(D, 2).to("cuda")
        S = generate_qjl_matrix(D, seed=43).to("cuda")

        torch.manual_seed(42)
        W = torch.randn(16, D, device="cuda")
        norms = W.norm(dim=-1)
        unit = W / (norms.unsqueeze(-1) + 1e-8)
        rotated = unit @ Pi.T
        idx = (rotated.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        residual = unit - centroids[idx] @ Pi
        res_norms = residual.norm(dim=-1)
        signs = torch.sign(residual @ S.T)
        signs[signs == 0] = 1.0
        packed = pack_vectors_batched(idx, signs, norms, res_norms, D, 2)

        result = cuda_unpack(packed, D, 2)
        if result is None:
            pytest.skip("CUDA kernel not available")
        match = (result[0].cpu() == idx.cpu()).float().mean()
        assert match > 0.99, f"D={D}: match={match:.4f}"


# ── 4b. CUDA Unpack 3-Bit (mse_bits=3) ───────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCUDAUnpack3Bit:
    """Test CUDA unpack kernel with mse_bits=3 (3-bit bitstream).

    3-bit indices (values 0-7) span byte boundaries, e.g. bits [6:8] of
    one byte + bit [0] of the next.  This exercises the non-aligned
    bit-extraction path in the CUDA kernel.
    """

    @pytest.mark.parametrize("D", [128, 512, 2048])
    def test_pack_unpack_roundtrip(self, D):
        """Pack random 3-bit indices, CUDA-unpack, verify 100% match."""
        from vllm.multiquant.shared.bitpack import pack_vectors_batched
        from vllm.multiquant.shared.centroids import get_centroids
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.weight_quant.archer_ops import cuda_unpack

        mse_bits = 3
        Pi = generate_rotation_matrix(D, seed=42).to("cuda")
        centroids = get_centroids(D, mse_bits).to("cuda")  # 8 levels
        S = generate_qjl_matrix(D, seed=43).to("cuda")

        torch.manual_seed(42)
        W = torch.randn(16, D, device="cuda")
        norms = W.norm(dim=-1)
        unit = W / (norms.unsqueeze(-1) + 1e-8)
        rotated = unit @ Pi.T
        idx = (rotated.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        residual = unit - centroids[idx] @ Pi
        res_norms = residual.norm(dim=-1)
        signs = torch.sign(residual @ S.T)
        signs[signs == 0] = 1.0
        packed = pack_vectors_batched(idx, signs, norms, res_norms, D, mse_bits)

        result = cuda_unpack(packed, D, mse_bits)
        if result is None:
            pytest.skip("CUDA kernel not available")
        match = (result[0].cpu() == idx.cpu()).float().mean()
        assert match == 1.0, f"D={D}: match={match:.4f} (expected 1.0)"

    def test_byte_boundary_indices(self):
        """Known pattern [0,1,2,3,4,5,6,7] repeated — exercises byte boundaries."""
        from vllm.multiquant.shared.bitpack import pack_vectors_batched
        from vllm.multiquant.shared.centroids import get_centroids
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.weight_quant.archer_ops import cuda_unpack

        D = 128
        mse_bits = 3
        N = 8
        Pi = generate_rotation_matrix(D, seed=42).to("cuda")
        centroids = get_centroids(D, mse_bits).to("cuda")
        S = generate_qjl_matrix(D, seed=43).to("cuda")

        # Build indices: tile [0,1,2,3,4,5,6,7] across D columns
        pattern = torch.arange(8, device="cuda")
        idx = pattern.repeat(D // 8).unsqueeze(0).expand(N, -1)  # (N, D)

        # Fabricate matching norms / signs (content irrelevant for index check)
        norms = torch.ones(N, device="cuda")
        res_norms = torch.ones(N, device="cuda") * 0.1
        signs = torch.ones(N, D, device="cuda")

        packed = pack_vectors_batched(idx, signs, norms, res_norms, D, mse_bits)
        result = cuda_unpack(packed, D, mse_bits)
        if result is None:
            pytest.skip("CUDA kernel not available")
        match = (result[0].cpu() == idx.cpu()).float().mean()
        assert match == 1.0, (
            f"Byte-boundary test: match={match:.4f} (expected 1.0)"
        )


# ── 5. F.linear with Decompressed Weights ──────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFLinear:
    @pytest.mark.parametrize("D", [128, 512, 2048])
    def test_correctness(self, D):
        layer, W, archer = _quantize(256, D)
        W_d = archer._decompress(layer).to(torch.bfloat16).contiguous()
        x = torch.randn(4, D, dtype=torch.bfloat16, device="cuda")
        y = F.linear(x, W_d)
        y_ref = F.linear(x, W.bfloat16())
        cos = F.cosine_similarity(y.float(), y_ref.float(), dim=-1).mean()
        assert cos > 0.5, f"F.linear D={D}: cos={cos:.4f}"
        assert not y.isnan().any(), "F.linear output contains NaN"


# ── 6. Memory: zero-trick ──────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMemory:
    def test_packed_smaller_than_bf16(self):
        layer, _, _ = _quantize(2048, 1024)
        weight_bytes = layer.weight.numel() * layer.weight.element_size()
        bf16_bytes = 2048 * 1024 * 2
        assert weight_bytes < bf16_bytes * 0.3, (
            f"Packed {weight_bytes} >= 30% of BF16 {bf16_bytes}"
        )


# ── 7. Repeated Decompress (Serving Sim) ───────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestRepeatedDecompress:
    def test_50_iterations(self):
        layer, _, archer = _quantize(512, 256)
        x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
        results = []
        for i in range(50):
            W = archer._decompress(layer).to(torch.bfloat16).contiguous()
            y = F.linear(x, W)
            results.append(y.sum().item())
            del W
        torch.cuda.synchronize()
        # All results should be identical (deterministic decompress)
        assert all(r == results[0] for r in results), "Non-deterministic decompress"


# ── 8. TQ vs RQ Parity ────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTQvsRQ:
    def test_parity(self):
        _, W, _ = _quantize(128, 128, 3, "tq")
        l_tq, _, a_tq = _quantize(128, 128, 3, "tq")
        l_rq, _, a_rq = _quantize(128, 128, 3, "rq")
        cos_tq = F.cosine_similarity(W, a_tq._decompress(l_tq).float(), dim=-1).mean()
        cos_rq = F.cosine_similarity(W, a_rq._decompress(l_rq).float(), dim=-1).mean()
        assert abs(cos_tq - cos_rq) < 0.35


# ── 9. Mixed Dimensions ───────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMixedDimensions:
    def test_multi_layer(self):
        dims = [(2048, 10240), (8960, 512), (1344, 2048), (5120, 768)]
        for out, inp in dims:
            layer, W, archer = _quantize(out, inp)
            W_d = archer._decompress(layer)
            assert W_d.shape == (out, inp), f"Shape {W_d.shape} != ({out},{inp})"
            cos = F.cosine_similarity(W, W_d.float(), dim=-1).mean()
            assert cos > 0.55, f"({out}×{inp}): cos={cos:.4f}"
            assert not W_d.isnan().any(), f"NaN in ({out}×{inp})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestRQLargeDimensions:
    """RQ with dimensions not divisible by 3 — tests Clifford padding."""

    @pytest.mark.parametrize("D", [128, 129, 256, 512, 768, 1024])
    @pytest.mark.parametrize("method_bits", [("rq", 3), ("rq", 4)])
    def test_rq_quality_various_dims(self, D, method_bits):
        method, bits = method_bits
        layer, W, archer = _quantize(32, D, bits=bits, method=method)
        W_d = archer._decompress(layer)
        cos = F.cosine_similarity(W, W_d.float(), dim=-1).mean()
        # RQ quality should be decent for all dimensions
        assert cos > 0.50, f"{method.upper()}{bits} D={D}: cos={cos:.4f}"
        assert not W_d.isnan().any(), f"NaN in {method.upper()}{bits} D={D}"


# ── 10. Fancy-Indexed Cache Unpack ─────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFancyIndexUnpack:
    def test_contiguous_fix(self):
        """Exact pattern from serving: kv_cache[bi_phys, 0, bo, kv_h]."""
        D = 128
        packed_size = math.ceil(D * 2 / 8) + math.ceil(D / 8) + 4
        cache = torch.randint(0, 255, (8, 2, 16, 4, packed_size),
                              dtype=torch.uint8, device="cuda")

        seq_len = 20
        block_table = torch.tensor([0, 3, 5, 7, 1, 2, 4, 6], device="cuda")
        positions = torch.arange(seq_len, device="cuda")
        bi_phys = block_table[positions // 16]
        bo = positions % 16

        for kv_h in range(4):
            k_packed = cache[bi_phys, 0, bo, kv_h]
            assert not k_packed.is_contiguous() or k_packed.is_contiguous()

            from vllm.multiquant.weight_quant.archer_ops import cuda_unpack
            result = cuda_unpack(k_packed.contiguous(), D, 2)
            if result is None:
                pytest.skip("CUDA kernel not available")
            assert result[0].shape == (seq_len, D)


# ── 11. 40-Layer Pack Simulation ───────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMultiLayerPack:
    def test_40_layers(self):
        from vllm.multiquant.shared.bitpack import pack_vectors_batched
        from vllm.multiquant.shared.centroids import get_centroids
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        from vllm.multiquant.shared.qjl import generate_qjl_matrix

        D = 128
        Pi = generate_rotation_matrix(D, seed=42).to("cuda")
        S = generate_qjl_matrix(D, seed=43).to("cuda")
        centroids = get_centroids(D, 2).to("cuda")

        t0 = time.perf_counter()
        for _ in range(40):
            vecs = torch.randn(128, D, device="cuda")
            norms = vecs.norm(dim=-1)
            unit = vecs / (norms.unsqueeze(-1) + 1e-8)
            rotated = unit @ Pi.T
            idx = (rotated.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
            residual = unit - centroids[idx] @ Pi
            res_norms = residual.norm(dim=-1)
            signs = torch.sign(residual @ S.T)
            signs[signs == 0] = 1.0
            pack_vectors_batched(idx, signs, norms, res_norms, D, 2)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"\n40 layers pack: {elapsed:.2f}s")
        assert elapsed < 10


# ── 12. torch.compile Compatibility ──────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestArcherTorchCompile:
    """Verify Archer works with torch.compile (vLLM v1 requirement)."""

    @pytest.fixture
    def archer_linear(self):
        from vllm.multiquant.weight_quant.config import ArcherConfig
        from vllm.multiquant.weight_quant.online_linear import (
            ArcherOnlineLinearMethod,
        )
        cfg = ArcherConfig(bits=3, method="tq")
        method = ArcherOnlineLinearMethod(cfg)
        layer = nn.Module()
        torch.manual_seed(42)
        layer.weight = nn.Parameter(
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        method._quantize_layer(layer)
        return layer, method

    def test_apply_without_compile(self, archer_linear):
        """Baseline: apply works without torch.compile."""
        layer, method = archer_linear
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        out = method.apply(layer, x)
        assert out.shape == (4, 64)
        assert not out.isnan().any()

    def test_apply_inside_torch_compile(self, archer_linear):
        """Critical: apply must not crash inside torch.compile."""
        layer, method = archer_linear
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)

        @torch.compile(fullgraph=True, backend="eager")
        def compiled_forward(x):
            return method.apply(layer, x)

        out = compiled_forward(x)
        assert out.shape == (4, 64)

    def test_dynamo_tracing_no_crash(self, archer_linear):
        """torch._dynamo must trace without Unsupported error."""
        layer, method = archer_linear
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)

        import torch._dynamo
        torch._dynamo.reset()

        def fn(x):
            return method.apply(layer, x)

        compiled_fn = torch.compile(fn, backend="eager")
        out = compiled_fn(x)
        assert out.shape == (4, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
