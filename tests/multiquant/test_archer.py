#!/usr/bin/env python3
"""Archer (MultiQuant weight quantization) tests.

Level 1: Config + Registration
Level 2: Weight compression quality
Level 3: Round-trip correctness
"""

import math

import pytest
import torch


class TestArcherConfig:
    def test_config_creation(self):
        from vllm.multiquant.weight_quant.config import ArcherConfig
        cfg = ArcherConfig(bits=3, method="rq")
        assert cfg.bits == 3
        assert cfg.method == "rq"
        assert cfg.get_name() == "multiquant"

    def test_config_from_config(self):
        from vllm.multiquant.weight_quant.config import ArcherConfig
        cfg = ArcherConfig.from_config({"bits": 4, "method": "tq"})
        assert cfg.bits == 4
        assert cfg.method == "tq"

    def test_supported_dtypes(self):
        from vllm.multiquant.weight_quant.config import ArcherConfig
        dtypes = ArcherConfig.get_supported_act_dtypes()
        assert torch.bfloat16 in dtypes


class TestArcherCompression:
    def _compress_decompress(self, method, bits, D=128):
        """Helper: compress weight matrix and measure quality."""
        from vllm.multiquant.registry import get_kv_quantizer_config
        from vllm.multiquant.turboquant.centroids import get_centroids

        cache_dtype = f"{method}{bits}"
        config = get_kv_quantizer_config(cache_dtype, D)
        centroids = get_centroids(D, config.mse_bits)

        torch.manual_seed(42)
        W = torch.randn(64, D)  # 64 output neurons

        # Per-row quantization
        row_norms = W.norm(dim=-1)
        W_unit = W / (row_norms.unsqueeze(-1) + 1e-8)

        if method == "rq":
            from vllm.multiquant.rotorquant.quantizer import generate_rotors
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors,
                extract_vectors_from_multivectors,
                reverse, rotor_sandwich,
            )
            rotors = generate_rotors(D, seed=42)
            mv = embed_vectors_as_multivectors(W_unit)
            mv_rot = rotor_sandwich(rotors, mv)
            for gi in [1, 2, 3, 7]:
                comp = mv_rot[..., gi]
                diffs = comp.unsqueeze(-1) - centroids
                idx = diffs.abs().argmin(dim=-1)
                mv_rot[..., gi] = centroids[idx]
            rotor_rev = reverse(rotors)
            mv_recon = rotor_sandwich(rotor_rev, mv_rot)
            W_mse = extract_vectors_from_multivectors(mv_recon, D)
        else:
            from vllm.multiquant.turboquant.quantizer import (
                generate_rotation_matrix,
            )
            Pi = generate_rotation_matrix(D, seed=42)
            rotated = W_unit @ Pi.T
            diffs = rotated.unsqueeze(-1) - centroids
            idx = diffs.abs().argmin(dim=-1)
            quantized = centroids[idx]
            W_mse = quantized @ Pi

        W_recon = row_norms.unsqueeze(-1) * W_mse

        # Cosine similarity per row
        cos = torch.nn.functional.cosine_similarity(W, W_recon, dim=-1)
        return cos.mean().item()

    def test_tq3_quality(self):
        """TQ3 = 2-bit MSE (no QJL for weights). cos ~0.79."""
        cos = self._compress_decompress("tq", 3)
        assert cos > 0.70, f"TQ3 weight quality too low: {cos:.4f}"

    def test_rq3_quality(self):
        cos = self._compress_decompress("rq", 3)
        assert cos > 0.70, f"RQ3 weight quality too low: {cos:.4f}"

    def test_rq4_quality(self):
        cos = self._compress_decompress("rq", 4)
        assert cos > 0.82, f"RQ4 weight quality too low: {cos:.4f}"

    def test_rq2_quality(self):
        """RQ2 = 1-bit MSE, very aggressive."""
        cos = self._compress_decompress("rq", 2)
        assert cos > 0.50, f"RQ2 weight quality too low: {cos:.4f}"

    def test_tq_rq_similar(self):
        cos_tq = self._compress_decompress("tq", 3)
        cos_rq = self._compress_decompress("rq", 3)
        assert abs(cos_tq - cos_rq) < 0.15, (
            f"TQ/RQ quality gap too large: TQ={cos_tq:.4f}, RQ={cos_rq:.4f}"
        )


class TestArcherLinearMethod:
    def test_uses_meta_device(self):
        from vllm.multiquant.weight_quant.online_linear import (
            ArcherOnlineLinearMethod,
        )
        from vllm.multiquant.weight_quant.config import ArcherConfig
        cfg = ArcherConfig(bits=3, method="rq")
        method = ArcherOnlineLinearMethod(cfg)
        assert method.uses_meta_device is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
