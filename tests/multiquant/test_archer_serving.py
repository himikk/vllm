#!/usr/bin/env python3
"""Reproduce Archer unpack kernel device-side assert in serving scenario.

Simulates repeated _decompress() + F.linear calls like vLLM serving does.
Tests ALL GLM-4.7 layer dimensions in a loop.
"""

import math
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestArcherServingReproduction:

    def _make_layer(self, out_features, in_features, mse_bits=2):
        """Create a packed layer like Archer _quantize_layer does."""
        from vllm.multiquant.shared.centroids import get_centroids
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        from vllm.multiquant.shared.bitpack import pack_vectors_batched

        torch.manual_seed(42)
        W = torch.randn(out_features, in_features, device="cuda")
        Pi = generate_rotation_matrix(in_features, seed=42).to("cuda")
        S = generate_qjl_matrix(in_features, seed=43).to("cuda")
        centroids = get_centroids(in_features, mse_bits).to("cuda")

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

        packed = pack_vectors_batched(idx, signs, row_norms, res_norms,
                                      in_features, mse_bits)

        # Validate packed size
        expected = math.ceil(in_features * mse_bits / 8) + math.ceil(in_features / 8) + 4
        assert packed.shape[1] == expected, (
            f"Pack mismatch: {packed.shape[1]} != {expected} "
            f"(D={in_features}, bits={mse_bits})"
        )

        layer = nn.Module()
        layer.weight = nn.Parameter(packed, requires_grad=False)
        layer._archer_rotation = Pi
        layer._archer_S = S
        layer._archer_centroids = centroids
        layer._archer_in_features = in_features
        layer._archer_mse_bits = mse_bits
        layer._archer_method = "tq"
        layer._archer_packed = True
        return layer

    def _validate_before_unpack(self, layer):
        """Log and validate tensor state before unpack."""
        packed = layer.weight.data
        D = layer._archer_in_features
        bits = layer._archer_mse_bits
        expected = math.ceil(D * bits / 8) + math.ceil(D / 8) + 4

        info = {
            "D": D,
            "rows": packed.shape[0],
            "cols": packed.shape[1],
            "expected_cols": expected,
            "stride": packed.stride(),
            "contiguous": packed.is_contiguous(),
            "dtype": str(packed.dtype),
            "device": str(packed.device),
        }

        if packed.shape[1] != expected:
            raise RuntimeError(f"PACKED SIZE MISMATCH: {info}")
        if not packed.is_contiguous():
            raise RuntimeError(f"NOT CONTIGUOUS: {info}")
        if packed.stride(0) != packed.shape[1]:
            raise RuntimeError(f"STRIDE MISMATCH: {info}")

        return info

    def test_single_dim_repeated(self):
        """Single dimension, repeated calls — basic reproduction."""
        from vllm.multiquant.weight_quant.archer_ops import cuda_unpack

        layer = self._make_layer(4096, 2048)
        packed = layer.weight.data

        for i in range(50):
            info = self._validate_before_unpack(layer)
            result = cuda_unpack(packed, 2048, 2)
            if result is None:
                pytest.skip("Unpack kernel not available")
            idx, signs, rn, resn = result
            # Use the results like serving does
            centroids = layer._archer_centroids
            c_vals = centroids[idx.long()]
            W_mse = c_vals @ layer._archer_rotation
            x = torch.randn(2, 2048, dtype=torch.bfloat16, device="cuda")
            y = F.linear(x, W_mse.bfloat16())
            if i % 10 == 0:
                torch.cuda.synchronize()
                print(f"  Step {i}: OK, y.sum={y.sum():.2f}")

    def test_all_glm_dims_interleaved(self):
        """All GLM-4.7 dimensions interleaved — like real serving."""
        from vllm.multiquant.weight_quant.archer_ops import cuda_unpack

        dims = [
            (1344, 2048), (20480, 2048), (2048, 10240),
            (2048, 1536), (2048, 5120), (3072, 2048),
            (5120, 768), (8960, 512),
        ]

        layers = {}
        for out, D in dims:
            layers[(out, D)] = self._make_layer(out, D)

        for iteration in range(10):
            for (out, D), layer in layers.items():
                info = self._validate_before_unpack(layer)
                packed = layer.weight.data

                result = cuda_unpack(packed, D, 2)
                if result is None:
                    pytest.skip("Unpack kernel not available")

                idx, signs, rn, resn = result
                centroids = layer._archer_centroids
                c_vals = centroids[idx.long()]
                W_mse = c_vals @ layer._archer_rotation

                x = torch.randn(2, D, dtype=torch.bfloat16, device="cuda")
                y = F.linear(x, W_mse.bfloat16())

            torch.cuda.synchronize()
            print(f"  Iteration {iteration}: all {len(dims)} dims OK")

    def test_qwen_dims_interleaved(self):
        """Qwen3.5 dimensions (includes 4304, 4608)."""
        from vllm.multiquant.weight_quant.archer_ops import cuda_unpack

        dims = [
            (12288, 2048), (2048, 4096), (2048, 4304),
            (2048, 4608), (4096, 1152), (2048, 512),
        ]

        layers = {}
        for out, D in dims:
            layers[(out, D)] = self._make_layer(out, D)

        for iteration in range(10):
            for (out, D), layer in layers.items():
                info = self._validate_before_unpack(layer)
                packed = layer.weight.data

                result = cuda_unpack(packed, D, 2)
                if result is None:
                    pytest.skip("Unpack kernel not available")

                idx, signs, rn, resn = result
                centroids = layer._archer_centroids
                c_vals = centroids[idx.long()]
                W_mse = c_vals @ layer._archer_rotation

                x = torch.randn(2, D, dtype=torch.bfloat16, device="cuda")
                y = F.linear(x, W_mse.bfloat16())

            torch.cuda.synchronize()
            print(f"  Iteration {iteration}: all {len(dims)} dims OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
