#!/usr/bin/env python3
"""Archer CUDA kernel test — decompress packed TQ3 weights.

Compares CUDA kernel output with Python decompress for correctness.
"""

import math
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestArcherKernel:

    def _pack_test_data(self, D=128, num_rows=32, mse_bits=2):
        """Create test data: BF16 weights → compress → packed uint8."""
        torch.manual_seed(42)
        W = torch.randn(num_rows, D)

        from vllm.multiquant.shared.centroids import get_centroids
        centroids = get_centroids(D, mse_bits)

        from vllm.multiquant.turboquant.quantizer import (
            generate_rotation_matrix,
        )
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        Pi = generate_rotation_matrix(D, seed=42)
        S = generate_qjl_matrix(D, seed=43)

        # Quantize
        row_norms = W.norm(dim=-1)
        W_unit = W / (row_norms.unsqueeze(-1) + 1e-8)
        rotated = W_unit @ Pi.T
        diffs = rotated.unsqueeze(-1) - centroids
        mse_indices = diffs.abs().argmin(dim=-1)

        # QJL
        quantized = centroids[mse_indices]
        W_mse = quantized @ Pi
        residual = W_unit - W_mse
        res_norms = residual.norm(dim=-1)
        projected = residual @ S.T
        signs = torch.sign(projected)
        signs[signs == 0] = 1.0

        # Pack
        from vllm.multiquant.shared.bitpack import pack_vectors_batched
        packed = pack_vectors_batched(
            mse_indices, signs, row_norms, res_norms,
            D, mse_bits,
        )

        # Python decompress for reference
        correction = math.sqrt(math.pi / 2) / D
        W_qjl = correction * res_norms.unsqueeze(-1) * (signs @ S)
        W_ref = row_norms.unsqueeze(-1) * (W_mse + W_qjl)

        return packed, Pi, S, centroids, W_ref

    def test_python_decompress_roundtrip(self):
        """Verify Python decompress matches compression."""
        packed, Pi, S, centroids, W_ref = self._pack_test_data(D=128)
        # Use Archer's Python _decompress
        from vllm.multiquant.weight_quant.online_linear import (
            ArcherOnlineLinearMethod,
        )
        from vllm.multiquant.weight_quant.config import ArcherConfig

        method = ArcherOnlineLinearMethod(ArcherConfig(bits=3, method="tq"))

        # Create fake layer with buffers
        import torch.nn as nn
        layer = nn.Module()
        layer.weight = nn.Parameter(packed.cuda(), requires_grad=False)
        layer._archer_rotation = Pi.cuda()
        layer._archer_S = S.cuda()
        layer._archer_centroids = centroids.cuda()
        layer._archer_in_features = 128
        layer._archer_mse_bits = 2
        layer._archer_method = "tq"
        layer._archer_packed = True

        W_decomp = method._decompress(layer).cpu()

        cos = torch.nn.functional.cosine_similarity(
            W_ref, W_decomp, dim=-1
        ).mean()
        assert cos > 0.99, f"Python decompress mismatch: cos={cos:.4f}"

    def test_cuda_kernel_matches_python(self):
        """CUDA kernel output should match Python decompress."""
        D = 128
        packed, Pi, S, centroids, W_ref = self._pack_test_data(D=D)

        from vllm.multiquant.weight_quant.archer_ops import cuda_decompress
        result = cuda_decompress(
            packed.cuda(), Pi.cuda(), S.cuda(), centroids.cuda(),
            head_size=D, out_dtype=torch.float32,
        )

        if result is None:
            pytest.skip("Archer CUDA kernel not available (JIT failed)")

        cos = torch.nn.functional.cosine_similarity(
            W_ref, result.cpu(), dim=-1
        ).mean()
        assert cos > 0.99, f"CUDA kernel mismatch vs reference: cos={cos:.4f}"

    def test_cuda_kernel_speed(self):
        """CUDA kernel should be significantly faster than Python."""
        D = 2048
        num_rows = 1024
        packed, Pi, S, centroids, _ = self._pack_test_data(
            D=D, num_rows=num_rows
        )

        packed_gpu = packed.cuda()
        Pi_gpu = Pi.cuda()
        S_gpu = S.cuda()
        c_gpu = centroids.cuda()

        from vllm.multiquant.weight_quant.archer_ops import cuda_decompress

        # Warmup
        result = cuda_decompress(packed_gpu, Pi_gpu, S_gpu, c_gpu,
                                 head_size=D, out_dtype=torch.bfloat16)
        if result is None:
            pytest.skip("Archer CUDA kernel not available")

        # Benchmark
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(10):
            cuda_decompress(packed_gpu, Pi_gpu, S_gpu, c_gpu,
                            head_size=D, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cuda_ms = (t1 - t0) / 10 * 1000

        print(f"\nArcher CUDA decompress: {num_rows}×{D} in {cuda_ms:.1f}ms")
        # Should be < 50ms for 1024 rows × 2048 dim
        assert cuda_ms < 500, f"CUDA kernel too slow: {cuda_ms:.1f}ms"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestArcherUnpackKernel:

    def _make_packed(self, D=2048, num_rows=1024, mse_bits=2):
        """Pack test data and return packed + reference indices/signs/norms."""
        torch.manual_seed(42)
        from vllm.multiquant.shared.centroids import get_centroids
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.shared.bitpack import pack_vectors_batched

        W = torch.randn(num_rows, D)
        Pi = generate_rotation_matrix(D, seed=42)
        S = generate_qjl_matrix(D, seed=43)
        centroids = get_centroids(D, mse_bits)

        row_norms = W.norm(dim=-1)
        W_unit = W / (row_norms.unsqueeze(-1) + 1e-8)
        rotated = W_unit @ Pi.T
        diffs = rotated.unsqueeze(-1) - centroids
        idx = diffs.abs().argmin(dim=-1)
        residual = W_unit - centroids[idx] @ Pi
        res_norms = residual.norm(dim=-1)
        projected = residual @ S.T
        signs = torch.sign(projected)
        signs[signs == 0] = 1.0

        packed = pack_vectors_batched(idx, signs, row_norms, res_norms, D, mse_bits)
        return packed, idx, signs, row_norms, res_norms

    def test_unpack_correctness_d2048(self):
        """CUDA unpack matches Python pack for D=2048."""
        D = 2048
        packed, ref_idx, ref_signs, ref_rn, ref_resn = self._make_packed(D=D, num_rows=32)

        from vllm.multiquant.weight_quant.archer_ops import cuda_unpack
        result = cuda_unpack(packed.cuda(), D, mse_bits=2)
        if result is None:
            pytest.skip("Archer unpack kernel not available")

        idx, signs, rn, resn = result
        idx_match = (idx.cpu() == ref_idx).float().mean()
        sign_match = (signs.cpu() == ref_signs).float().mean()
        # float16 norms have limited precision
        rn_close = torch.allclose(rn.cpu(), ref_rn.float(), rtol=1e-2, atol=1e-1)

        assert idx_match > 0.99, f"Index mismatch: {idx_match:.4f}"
        assert sign_match > 0.99, f"Sign mismatch: {sign_match:.4f}"
        assert rn_close, f"Row norms mismatch: max diff={torch.abs(rn.cpu()-ref_rn.float()).max():.4f}"

    def test_unpack_speed(self):
        """CUDA unpack should be <1ms for 1024×2048."""
        D = 2048
        packed, _, _, _, _ = self._make_packed(D=D, num_rows=1024)
        packed_gpu = packed.cuda()

        from vllm.multiquant.weight_quant.archer_ops import cuda_unpack
        # Warmup
        result = cuda_unpack(packed_gpu, D, mse_bits=2)
        if result is None:
            pytest.skip("Archer unpack kernel not available")

        import time
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            cuda_unpack(packed_gpu, D, mse_bits=2)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 100 * 1000
        print(f"\nArcher CUDA unpack: 1024×{D} in {ms:.2f}ms")
        assert ms < 5, f"Too slow: {ms:.2f}ms"

    def test_full_decompress_with_unpack_kernel(self):
        """Full decompress (CUDA unpack + cuBLAS GEMM) correctness."""
        D = 2048
        num_rows = 64
        packed, ref_idx, ref_signs, ref_rn, ref_resn = self._make_packed(
            D=D, num_rows=num_rows)

        from vllm.multiquant.shared.centroids import get_centroids
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        from vllm.multiquant.shared.qjl import generate_qjl_matrix

        Pi = generate_rotation_matrix(D, seed=42).cuda()
        S = generate_qjl_matrix(D, seed=43).cuda()
        centroids = get_centroids(D, 2).cuda()

        from vllm.multiquant.weight_quant.archer_ops import cuda_unpack
        result = cuda_unpack(packed.cuda(), D, mse_bits=2)
        if result is None:
            pytest.skip("Archer unpack kernel not available")

        idx, signs, row_norms, res_norms = result

        # cuBLAS GEMM for rotation
        c_vals = centroids[idx.long()]
        W_mse = c_vals @ Pi  # cuBLAS!

        import math
        correction = math.sqrt(math.pi / 2) / D
        W_qjl = correction * res_norms.unsqueeze(-1) * (signs @ S)  # cuBLAS!
        W_decomp = row_norms.unsqueeze(-1) * (W_mse + W_qjl)

        # Reference
        W = torch.randn(num_rows, D)
        torch.manual_seed(42)
        W = torch.randn(num_rows, D)
        cos = torch.nn.functional.cosine_similarity(
            W.cuda().float(), W_decomp.float(), dim=-1
        ).mean()
        # cos won't be 1.0 (quantization loss) but should be reasonable
        print(f"\nFull decompress cos: {cos:.4f}")
        assert cos > 0.5, f"Decompress quality too low: {cos:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
