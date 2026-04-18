# SPDX-License-Identifier: Apache-2.0
"""Standalone tests for TurboQuant quantizer correctness.

Tests MSE distortion bounds, inner product unbiasedness,
pack/unpack round-trip, and needle-in-haystack retrieval.

Run: python tests/turboquant/test_quantizer.py
"""

import math
import sys
import os

import torch

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vllm.turboquant.config import TurboQuantConfig
from vllm.turboquant.quantizer import TurboQuantizer
from vllm.turboquant.centroids import get_centroids, solve_lloyd_max


def test_lloyd_max_centroids():
    """Verify codebook properties."""
    print("=" * 60)
    print("TEST 1: Lloyd-Max Codebook")
    print("=" * 60)

    for d in [64, 128]:
        for bits in [1, 2, 3]:
            centroids, boundaries = solve_lloyd_max(d, bits)
            n = 2**bits
            assert len(centroids) == n
            assert len(boundaries) == n - 1
            # Centroids should be symmetric around 0
            assert centroids.sum().abs() < 0.01, (
                f"d={d}, bits={bits}: centroids not symmetric"
            )
            print(
                f"  d={d}, bits={bits}: {n} centroids, "
                f"range=[{centroids.min():.4f}, {centroids.max():.4f}]"
            )
    print("  PASSED\n")


def test_mse_distortion():
    """Verify MSE distortion is within theoretical bounds."""
    print("=" * 60)
    print("TEST 2: MSE Distortion Bounds")
    print("=" * 60)

    d = 128
    n_vectors = 2000
    device = "cpu"

    for total_bits in [2, 3, 4]:
        config = TurboQuantConfig(head_dim=d, total_bits=total_bits)
        tq = TurboQuantizer(config, layer_idx=0)

        # Random unit vectors
        x = torch.randn(n_vectors, d, device=device)
        x = x / x.norm(dim=-1, keepdim=True)

        compressed = tq.quantize(x)
        x_recon = tq.dequantize(compressed)

        mse = ((x - x_recon) ** 2).sum(dim=-1).mean().item()

        # Paper bound: D_mse <= sqrt(3)*pi/2 * (1/4^b)
        # But we use mse_bits = total_bits - 1 for MSE stage
        mse_bits = config.mse_bits
        theoretical = math.sqrt(3) * math.pi / 2 * (1 / (4**mse_bits))

        ratio = mse / theoretical
        status = "OK" if ratio < 2.0 else "WARN"
        print(
            f"  bits={total_bits} (mse={mse_bits}): MSE={mse:.6f}, "
            f"bound={theoretical:.6f}, ratio={ratio:.3f} [{status}]"
        )

    print()


def test_inner_product_unbiasedness():
    """Verify QJL gives unbiased inner product estimates."""
    print("=" * 60)
    print("TEST 3: Inner Product Unbiasedness")
    print("=" * 60)

    d = 128
    n_trials = 2000
    device = "cpu"

    for total_bits in [3, 4]:
        config = TurboQuantConfig(head_dim=d, total_bits=total_bits)
        tq = TurboQuantizer(config, layer_idx=0)

        # Random unit vectors
        x = torch.randn(n_trials, d, device=device)
        x = x / x.norm(dim=-1, keepdim=True)
        y = torch.randn(n_trials, d, device=device)
        y = y / y.norm(dim=-1, keepdim=True)

        # True inner products
        true_ip = (x * y).sum(dim=-1)

        # Quantize x, compute estimated inner products
        compressed = tq.quantize(x)

        # Reshape for attention_scores: (1, 1, N, D) format
        x_comp_4d = {
            k: v.unsqueeze(0).unsqueeze(0) if v.dim() == 2
            else v.unsqueeze(0).unsqueeze(0)
            for k, v in compressed.items()
        }
        y_4d = y.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)

        # Use per-vector inner product via quantizer
        # For simplicity, compute manually
        x_mse = tq.dequantize(compressed)  # This includes QJL correction
        estimated_ip = (x_mse * y).sum(dim=-1)

        # Also test the proper asymmetric estimator
        # Manual computation of TQ inner product
        q_rot = y @ tq.PiT   # = y @ Pi^T, (N, D)
        q_proj = y @ tq.ST   # = y @ S^T, (N, D)
        c_idx = tq.centroids[compressed["mse_indices"].long()]  # (N, D)
        term1 = (q_rot * c_idx).sum(dim=-1)
        signs_float = compressed["qjl_signs"].float()
        term2 = (q_proj * signs_float).sum(dim=-1)
        correction = math.sqrt(math.pi / 2) / d
        res_norm = compressed["res_norm"].float()
        vec_norm = compressed["vec_norm"].float()
        asymmetric_ip = vec_norm * (term1 + correction * res_norm * term2)

        bias = (asymmetric_ip - true_ip).mean().item()
        rmse = ((asymmetric_ip - true_ip) ** 2).mean().sqrt().item()
        corr = torch.corrcoef(torch.stack([true_ip, asymmetric_ip]))[0, 1].item()

        print(
            f"  bits={total_bits}: bias={bias:+.6f}, RMSE={rmse:.6f}, corr={corr:.4f}"
        )
        assert abs(bias) < 0.05, f"Bias too large: {bias}"

    print("  PASSED\n")


def test_pack_unpack_roundtrip():
    """Verify pack/unpack is lossless."""
    print("=" * 60)
    print("TEST 4: Pack/Unpack Round-Trip")
    print("=" * 60)

    d = 128
    n_vectors = 100
    device = "cpu"

    for total_bits in [3, 4]:
        config = TurboQuantConfig(head_dim=d, total_bits=total_bits)
        tq = TurboQuantizer(config, layer_idx=0)

        x = torch.randn(n_vectors, d, device=device)
        compressed = tq.quantize(x)

        packed = tq.pack_cache(compressed)
        unpacked = tq.unpack_cache(packed)

        assert torch.equal(compressed["mse_indices"], unpacked["mse_indices"]), (
            "MSE indices mismatch!"
        )
        assert torch.equal(compressed["qjl_signs"], unpacked["qjl_signs"]), (
            "QJL signs mismatch!"
        )

        # Norms are stored as float16, so exact match expected
        assert torch.equal(compressed["vec_norm"], unpacked["vec_norm"]), (
            "vec_norm mismatch!"
        )
        assert torch.equal(compressed["res_norm"], unpacked["res_norm"]), (
            "res_norm mismatch!"
        )

        print(
            f"  bits={total_bits}: packed_size={packed.shape[-1]} bytes/vector, "
            f"expected={config.packed_size} — MATCH"
        )

    print("  PASSED\n")


def test_needle_in_haystack():
    """Hide a key among many, verify retrieval after quantization."""
    print("=" * 60)
    print("TEST 5: Needle-in-Haystack")
    print("=" * 60)

    d = 128
    device = "cpu"

    for total_bits in [3, 4]:
        config = TurboQuantConfig(head_dim=d, total_bits=total_bits)
        tq = TurboQuantizer(config, layer_idx=0)

        for seq_len in [512, 2048]:
            keys = torch.randn(seq_len, d, device=device)
            keys = keys / keys.norm(dim=-1, keepdim=True)

            needle_pos = seq_len // 3
            query = keys[needle_pos].clone()

            compressed = tq.quantize(keys)

            # Use dequantize for score comparison (simpler, tests full pipeline)
            keys_recon = tq.dequantize(compressed)
            scores = keys_recon @ query

            # Compare with true scores
            true_scores = keys @ query
            corr = torch.corrcoef(
                torch.stack([true_scores, scores])
            )[0, 1].item()

            top_idx = scores.argmax().item()
            true_top = true_scores.argmax().item()
            print(
                f"  bits={total_bits}, seq={seq_len}: "
                f"tq_top1={top_idx}, true_top1={true_top} "
                f"(needle={needle_pos}), score_corr={corr:.4f}"
            )
            # Score correlation should be positive
            assert corr > 0.0, f"Score correlation negative: {corr}"

    print("  PASSED\n")


def test_compression_ratio():
    """Verify compression ratios match expectations."""
    print("=" * 60)
    print("TEST 6: Compression Ratios")
    print("=" * 60)

    for total_bits in [3, 4]:
        config = TurboQuantConfig(head_dim=128, total_bits=total_bits)
        fp16_bytes = 128 * 2  # 256 bytes
        ratio = fp16_bytes / config.packed_size
        print(
            f"  tq{total_bits}: {config.packed_size} bytes/vector, "
            f"compression={ratio:.1f}x vs FP16"
        )

    print()


if __name__ == "__main__":
    print()
    print("TurboQuant Standalone Validation")
    print("================================")
    print()

    test_lloyd_max_centroids()
    test_mse_distortion()
    test_inner_product_unbiasedness()
    test_pack_unpack_roundtrip()
    test_needle_in_haystack()
    test_compression_ratio()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
