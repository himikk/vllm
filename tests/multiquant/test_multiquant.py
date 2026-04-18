#!/usr/bin/env python3
"""MultiQuant framework tests.

Level 1: Registry + Config
Level 2: RotorQuant Clifford algebra
Level 3: RotorQuant pack/unpack round-trip
Level 4: Cross-quantizer comparison (TQ vs RQ quality)
Level 5: Backend selection
"""

import math

import pytest
import torch


# ============================================================
# Level 1: Registry + Config
# ============================================================

class TestRegistry:
    def test_tq3_registered(self):
        from vllm.multiquant.registry import is_multiquant_dtype
        assert is_multiquant_dtype("tq3")
        assert is_multiquant_dtype("tq4")

    def test_rq3_registered(self):
        from vllm.multiquant.registry import is_multiquant_dtype
        assert is_multiquant_dtype("rq3")
        assert is_multiquant_dtype("rq4")

    def test_fp8_not_multiquant(self):
        from vllm.multiquant.registry import is_multiquant_dtype
        assert not is_multiquant_dtype("fp8")
        assert not is_multiquant_dtype("auto")

    def test_get_tq_config(self):
        from vllm.multiquant.registry import get_kv_quantizer_config
        cfg = get_kv_quantizer_config("tq3", head_dim=128)
        assert cfg.head_dim == 128
        assert cfg.total_bits == 3
        assert cfg.mse_bits == 2

    def test_get_rq_config(self):
        from vllm.multiquant.registry import get_kv_quantizer_config
        cfg = get_kv_quantizer_config("rq3", head_dim=128)
        assert cfg.head_dim == 128
        assert cfg.total_bits == 3
        assert cfg.mse_bits == 2

    def test_tq_rq_same_packed_size(self):
        """TQ and RQ should have identical packed sizes (same format)."""
        from vllm.multiquant.registry import get_kv_quantizer_config
        tq = get_kv_quantizer_config("tq3", head_dim=128)
        rq = get_kv_quantizer_config("rq3", head_dim=128)
        assert tq.key_packed_size == rq.key_packed_size
        assert tq.cache_head_size == rq.cache_head_size

    def test_registered_dtypes(self):
        from vllm.multiquant.registry import get_registered_dtypes
        dtypes = get_registered_dtypes()
        assert "tq3" in dtypes
        assert "rq3" in dtypes
        assert len(dtypes) >= 4

    def test_unknown_dtype_raises(self):
        from vllm.multiquant.registry import get_kv_quantizer_config
        with pytest.raises(ValueError):
            get_kv_quantizer_config("xx3", head_dim=128)


# ============================================================
# Level 2: Clifford Algebra
# ============================================================

class TestClifford:
    def test_geometric_product_identity(self):
        """e_i * e_i = +1 in Cl(3,0)."""
        from vllm.multiquant.rotorquant.clifford import geometric_product
        # e1 = [0, 1, 0, 0, 0, 0, 0, 0]
        e1 = torch.zeros(8)
        e1[1] = 1.0
        result = geometric_product(e1, e1)
        # e1*e1 = 1 (scalar)
        assert torch.allclose(result[0], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(result[1:], torch.zeros(7), atol=1e-6)

    def test_anticommutativity(self):
        """e_i * e_j = -e_j * e_i for i != j."""
        from vllm.multiquant.rotorquant.clifford import geometric_product
        e1 = torch.zeros(8); e1[1] = 1.0
        e2 = torch.zeros(8); e2[2] = 1.0
        ab = geometric_product(e1, e2)
        ba = geometric_product(e2, e1)
        assert torch.allclose(ab, -ba, atol=1e-6)

    def test_rotor_preserves_norm(self):
        """Rotor sandwich R x R̃ preserves vector norm."""
        from vllm.multiquant.rotorquant.clifford import (
            make_random_rotor, rotor_sandwich, embed_vectors_as_multivectors,
            extract_vectors_from_multivectors,
        )
        torch.manual_seed(42)
        v = torch.randn(6)  # 2 groups of 3
        mv = embed_vectors_as_multivectors(v.unsqueeze(0))

        rotor = make_random_rotor((), seed=123)
        mv_rot = rotor_sandwich(rotor, mv)
        v_rot = extract_vectors_from_multivectors(mv_rot, 6).squeeze(0)

        assert torch.allclose(
            v.norm(), v_rot.norm(), atol=1e-4
        ), f"Norm changed: {v.norm():.4f} → {v_rot.norm():.4f}"

    def test_embed_extract_roundtrip(self):
        """embed → extract is identity."""
        from vllm.multiquant.rotorquant.clifford import (
            embed_vectors_as_multivectors, extract_vectors_from_multivectors,
        )
        v = torch.randn(3, 12)  # batch=3, d=12 (4 groups)
        mv = embed_vectors_as_multivectors(v)
        v_back = extract_vectors_from_multivectors(mv, 12)
        assert torch.allclose(v, v_back, atol=1e-6)

    def test_embed_non_divisible_by_3(self):
        """Embedding works for d not divisible by 3 (padding)."""
        from vllm.multiquant.rotorquant.clifford import (
            embed_vectors_as_multivectors, extract_vectors_from_multivectors,
        )
        v = torch.randn(2, 7)  # 7 not divisible by 3
        mv = embed_vectors_as_multivectors(v)
        v_back = extract_vectors_from_multivectors(mv, 7)
        assert torch.allclose(v, v_back, atol=1e-6)


# ============================================================
# Level 3: RotorQuant Round-Trip
# ============================================================

class TestRotorQuantRoundTrip:
    def test_rotors_shape(self):
        from vllm.multiquant.rotorquant.quantizer import generate_rotors
        rotors = generate_rotors(head_dim=128, seed=42)
        # 128 / 3 = 42.67 → ceil = 43 groups
        assert rotors.shape == (43, 8)

    def test_rq_round_trip_quality(self):
        """RotorQuant round-trip should have reasonable MSE."""
        from vllm.multiquant.rotorquant.quantizer import (
            generate_rotors, generate_qjl_matrix, _rq_round_trip_pytorch,
        )
        from vllm.multiquant.turboquant.centroids import get_centroids

        D = 128
        rotors = generate_rotors(D, seed=42)
        S = generate_qjl_matrix(D, seed=43)
        centroids = get_centroids(D, bits=2)

        # Random keys
        torch.manual_seed(0)
        keys = torch.randn(4, 8, D)  # 4 tokens, 8 heads

        recon = _rq_round_trip_pytorch(keys, rotors, S, centroids)
        assert recon.shape == keys.shape

        # Check cosine similarity (should be high)
        cos = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, D), recon.reshape(-1, D), dim=-1
        )
        mean_cos = cos.mean().item()
        assert mean_cos > 0.85, f"Cosine sim too low: {mean_cos:.4f}"

    def test_rq_vs_tq_similar_quality(self):
        """RQ and TQ should have similar reconstruction quality."""
        from vllm.multiquant.rotorquant.quantizer import (
            generate_rotors, generate_qjl_matrix as rq_gen_qjl,
            _rq_round_trip_pytorch,
        )
        from vllm.multiquant.turboquant.quantizer import (
            generate_rotation_matrix,
            generate_qjl_matrix as tq_gen_qjl,
            _tq_round_trip_pytorch,
        )
        from vllm.multiquant.turboquant.centroids import get_centroids

        D = 128
        centroids = get_centroids(D, bits=2)
        torch.manual_seed(0)
        keys = torch.randn(4, 8, D)

        # TQ round-trip
        Pi = generate_rotation_matrix(D, seed=42)
        S_tq = tq_gen_qjl(D, seed=43)
        tq_recon = _tq_round_trip_pytorch(keys, Pi, S_tq, centroids)

        # RQ round-trip
        rotors = generate_rotors(D, seed=42)
        S_rq = rq_gen_qjl(D, seed=43)
        rq_recon = _rq_round_trip_pytorch(keys, rotors, S_rq, centroids)

        # Both should have similar cosine similarity
        cos_tq = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, D), tq_recon.reshape(-1, D), dim=-1
        ).mean().item()
        cos_rq = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, D), rq_recon.reshape(-1, D), dim=-1
        ).mean().item()

        # RQ should be within 5% of TQ quality
        assert abs(cos_rq - cos_tq) < 0.05, (
            f"Quality gap too large: TQ={cos_tq:.4f}, RQ={cos_rq:.4f}"
        )


# ============================================================
# Level 4: Config inheritance
# ============================================================

class TestConfigInheritance:
    def test_tq_config_is_kv_quantizer_config(self):
        from vllm.multiquant.base import KVQuantizerConfig
        from vllm.multiquant.turboquant.config import TurboQuantConfig
        cfg = TurboQuantConfig(head_dim=128, total_bits=3)
        assert isinstance(cfg, KVQuantizerConfig)

    def test_rq_config_is_kv_quantizer_config(self):
        from vllm.multiquant.base import KVQuantizerConfig
        from vllm.multiquant.rotorquant.config import RotorQuantConfig
        cfg = RotorQuantConfig(head_dim=128, total_bits=3)
        assert isinstance(cfg, KVQuantizerConfig)

    def test_rq_config_n_groups(self):
        from vllm.multiquant.rotorquant.config import RotorQuantConfig
        cfg = RotorQuantConfig(head_dim=128, total_bits=3)
        assert cfg.n_groups == 43  # ceil(128/3)

    def test_compat_shim(self):
        """Old import path vllm.turboquant still works."""
        from vllm.turboquant import TurboQuantConfig
        cfg = TurboQuantConfig.from_cache_dtype("tq3", 128)
        assert cfg.head_dim == 128


# ============================================================
# Level 5: Backend registration
# ============================================================

class TestBackendRegistration:
    def test_multiquant_in_registry(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        assert hasattr(AttentionBackendEnum, "MULTIQUANT")

    def test_turboquant_still_in_registry(self):
        """TURBOQUANT enum kept for backward compat."""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        assert hasattr(AttentionBackendEnum, "TURBOQUANT")

    def test_is_quantized_kv_cache(self):
        from vllm.v1.attention.backend import is_quantized_kv_cache
        assert is_quantized_kv_cache("tq3")
        assert is_quantized_kv_cache("rq3")
        assert is_quantized_kv_cache("fp8")
        assert not is_quantized_kv_cache("auto")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
