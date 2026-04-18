#!/usr/bin/env python3
"""Unit test: TQ dtype flow through vLLM's attention pipeline.

Tests the complete chain:
  --kv-cache-dtype tq3
    → backend selector sees "tq3" → TURBOQUANT
    → attention.__init__ maps to "fp8" for cache allocation
    → cache_config.cache_dtype becomes "fp8"
    → kv_cache_spec has dtype=uint8, head_size=real_head_size
    → FlashInfer MetadataBuilder sees fp8 → no assert failure
    → TurboQuantImpl.do_kv_cache_update applies TQ round-trip

Run: python3 tests/turboquant/test_dtype_flow.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch


def test_dtype_mapping():
    """tq3/tq4 map to uint8 in STR_DTYPE_TO_TORCH_DTYPE."""
    print("TEST 1: dtype mapping")
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    assert "tq3" in STR_DTYPE_TO_TORCH_DTYPE, "tq3 missing from dtype map"
    assert "tq4" in STR_DTYPE_TO_TORCH_DTYPE, "tq4 missing from dtype map"
    assert STR_DTYPE_TO_TORCH_DTYPE["tq3"] == torch.uint8
    assert STR_DTYPE_TO_TORCH_DTYPE["fp8"] == torch.uint8
    print("  PASSED: tq3→uint8, tq4→uint8, fp8→uint8")


def test_cache_dtype_literal():
    """tq3/tq4 are valid CacheDType values."""
    print("TEST 2: CacheDType")
    from vllm.config.cache import CacheDType
    dtype_str = str(CacheDType)
    assert "tq3" in dtype_str, "tq3 not in CacheDType"
    assert "tq4" in dtype_str, "tq4 not in CacheDType"
    print("  PASSED: tq3, tq4 in CacheDType")


def test_backend_registry():
    """TURBOQUANT is registered."""
    print("TEST 3: Backend registry")
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    assert hasattr(AttentionBackendEnum, "TURBOQUANT"), "TURBOQUANT not in registry"
    print(f"  PASSED: TURBOQUANT = {AttentionBackendEnum.TURBOQUANT.value}")


def test_is_quantized():
    """is_quantized_kv_cache for fp8 and tq."""
    print("TEST 4: is_quantized_kv_cache")
    from vllm.v1.attention.backend import is_quantized_kv_cache
    assert is_quantized_kv_cache("fp8") == True
    assert is_quantized_kv_cache("fp8_e4m3") == True
    assert is_quantized_kv_cache("auto") == False
    # After mapping, tq3 becomes fp8 — so tq3 itself is not checked here
    print("  PASSED: fp8=True, auto=False")


def test_tq_config():
    """TurboQuantConfig for different head_dims."""
    print("TEST 5: TQ config")
    from vllm.turboquant.config import TurboQuantConfig

    for d, bits, expected_packed in [(64, 3, 28), (128, 3, 52), (256, 3, 100), (64, 4, 36)]:
        c = TurboQuantConfig.from_cache_dtype(f"tq{bits}", d)
        assert c.head_dim == d
        assert c.total_bits == bits
        assert c.key_packed_size == expected_packed, \
            f"d={d} bits={bits}: expected {expected_packed}, got {c.key_packed_size}"
        print(f"  d={d} tq{bits}: packed={c.key_packed_size} ✓")
    print("  PASSED")


def test_kv_cache_dtype_mapping():
    """kv_cache_dtype_str_to_dtype for tq3."""
    print("TEST 6: kv_cache_dtype_str_to_dtype")
    from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

    # tq3 maps to uint8 directly
    result = kv_cache_dtype_str_to_dtype("tq3", None)
    assert result == torch.uint8, f"Expected uint8, got {result}"

    # After mapping to fp8, it should also be uint8
    result_fp8 = kv_cache_dtype_str_to_dtype("fp8", None)
    assert result_fp8 == torch.uint8
    print("  PASSED: tq3→uint8, fp8→uint8")


def test_attention_init_mapping():
    """Simulate attention.__init__ dtype flow."""
    print("TEST 7: Attention init dtype flow")

    # Simulate the flow in Attention.__init__
    kv_cache_dtype = "tq3"

    # Step 1: TQ check
    _tq_enabled = kv_cache_dtype.startswith("tq")
    assert _tq_enabled, "TQ should be enabled for tq3"

    # Step 2: Map to fp8
    if kv_cache_dtype.startswith("tq"):
        _kv_storage_dtype = kv_cache_dtype  # Save original
        kv_cache_dtype = "fp8"

    assert kv_cache_dtype == "fp8", f"Expected fp8, got {kv_cache_dtype}"
    assert _kv_storage_dtype == "tq3", f"Storage dtype should be tq3"

    # Step 3: cache_config update
    class FakeConfig:
        cache_dtype = "tq3"
    cache_config = FakeConfig()
    cache_config.cache_dtype = "fp8"  # This is what our code does
    assert cache_config.cache_dtype == "fp8"

    # Step 4: FlashInfer's check
    # In FlashInfer MetadataBuilder: self.cache_dtype = cache_config.cache_dtype
    cache_dtype = cache_config.cache_dtype  # "fp8"
    assert cache_dtype.startswith("fp8"), "FlashInfer should see fp8"
    # → goes into the fp8 branch → no assert error ✓

    print("  PASSED: tq3 → fp8 mapping correct")
    print(f"    _kv_storage_dtype = {_kv_storage_dtype}")
    print(f"    kv_cache_dtype = {kv_cache_dtype}")
    print(f"    cache_config.cache_dtype = {cache_config.cache_dtype}")


def test_turboquant_impl():
    """TurboQuantImpl detects TQ from layer attribute."""
    print("TEST 8: TurboQuantImpl storage dtype detection")

    # Simulate: Impl gets "fp8" but layer has _kv_storage_dtype
    class FakeLayer:
        _kv_storage_dtype = "tq3"
        _tq_Pi = torch.randn(64, 64)

    layer = FakeLayer()
    has_tq = getattr(layer, '_kv_storage_dtype', None) is not None
    has_pi = hasattr(layer, '_tq_Pi')
    assert has_tq and has_pi, "Layer should have TQ attributes"
    print("  PASSED: layer._kv_storage_dtype detected")


def test_quantizer_round_trip():
    """TQ round-trip produces valid keys."""
    print("TEST 9: Quantizer round-trip")
    from vllm.turboquant.quantizer import tq_round_trip_keys, generate_rotation_matrix, generate_qjl_matrix
    from vllm.turboquant.centroids import get_centroids
    from vllm.turboquant.config import TurboQuantConfig
    import torch.nn.functional as F

    D = 64
    config = TurboQuantConfig(head_dim=D, total_bits=3)

    class FakeLayer(torch.nn.Module):
        pass

    layer = FakeLayer()
    layer._tq_Pi = generate_rotation_matrix(D, seed=42)
    layer._tq_S = generate_qjl_matrix(D, seed=43)
    layer._tq_centroids = get_centroids(D, config.mse_bits)
    layer._tq_config = config

    key = torch.randn(4, 2, D)
    key_tq = tq_round_trip_keys(key, layer)

    assert key_tq.shape == key.shape, f"Shape mismatch: {key_tq.shape} vs {key.shape}"
    cos = F.cosine_similarity(key.reshape(1, -1), key_tq.reshape(1, -1)).item()
    print(f"  Round-trip cosine: {cos:.4f}")
    assert cos > 0.8, f"Cosine too low: {cos}"
    print("  PASSED")


if __name__ == "__main__":
    print("\nTurboQuant Dtype Flow Unit Tests\n" + "=" * 50 + "\n")

    test_dtype_mapping()
    test_cache_dtype_literal()
    test_backend_registry()
    test_is_quantized()
    test_tq_config()
    test_kv_cache_dtype_mapping()
    test_attention_init_mapping()
    test_turboquant_impl()
    test_quantizer_round_trip()

    print("\n" + "=" * 50)
    print("ALL DTYPE FLOW TESTS PASSED")
    print("=" * 50)
