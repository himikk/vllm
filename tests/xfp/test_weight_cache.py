# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MultiQuant weight cache.

Two layers of tests:
  A. Generic plumbing (``MultiQuantWeightCache.save`` / ``.load``, hash,
     manifest, read-only) — agnostic to any quant method.
  B. XFP adapter (``xfp_weight_cache.save_linear`` + ``load_linear`` +
     MoE counterparts) — validates the XFP client reconstructs the same
     layer attribute layout.

Future quant methods (Archer / TQ / RQ / AutoRound RTN) add tests here
using the same pattern: roundtrip through the generic cache + adapter.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

from vllm.multiquant.weight_cache import MultiQuantWeightCache
from vllm.multiquant.xfp import xfp_weight_cache as xfp_cache


# ─── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def cache(tmp_path):
    return MultiQuantWeightCache(
        cache_root=str(tmp_path),
        model_basename="test-model",
        cache_key="deadbeefcafef00d",
        read_only=False,
    )


def _fake_registry(dtype_routed="xfp", bits_routed=0):
    reg = MagicMock()
    reg.to_dict.return_value = {
        "weights_attn":   {"dtype": "xfp", "bits": 0, "group_size": 0},
        "weights_routed": {"dtype": dtype_routed, "bits": bits_routed,
                           "group_size": 0},
        "lm_head":        {"dtype": "fp8", "bits": 8, "group_size": 0},
    }
    return reg


def _fake_linear_layer(N=128, K=256, bits=4, with_outliers=True, device="cpu"):
    layer = nn.Module()
    K_packed = (K + 7) // 8
    layer.xfp_packed = nn.Parameter(
        torch.randint(0, 2**31 - 1, (K_packed * N,), dtype=torch.int32,
                      device=device), requires_grad=False)
    layer.xfp_codebook = nn.Parameter(
        torch.randn(N, 1 << bits, dtype=torch.float16, device=device),
        requires_grad=False)
    if with_outliers:
        nout = 32
        layer.xfp_outlier_row = nn.Parameter(
            torch.randint(0, N, (nout,), dtype=torch.int64, device=device),
            requires_grad=False)
        layer.xfp_outlier_col = nn.Parameter(
            torch.randint(0, K, (nout,), dtype=torch.int64, device=device),
            requires_grad=False)
        layer.xfp_outlier_val = nn.Parameter(
            torch.randn(nout, dtype=torch.bfloat16, device=device),
            requires_grad=False)
        layer._xfp_has_outliers = True
    else:
        layer._xfp_has_outliers = False
    layer._xfp_bits = bits
    layer._xfp_K = K
    layer._xfp_N = N
    return layer


def _fake_moe_layer(E=4, N13=64, K13=128, N2=128, K2=32, bits=4, device="cpu"):
    layer = nn.Module()
    K13_packed = (K13 + 7) // 8
    K2_packed = (K2 + 7) // 8
    fpe13 = K13_packed * N13
    fpe2 = K2_packed * N2
    lut = 1 << bits
    layer.w13_xfp_packed = nn.Parameter(
        torch.randint(0, 2**31 - 1, (E * fpe13,), dtype=torch.int32,
                      device=device), requires_grad=False)
    layer.w13_xfp_codebook = nn.Parameter(
        torch.randn(E * N13, lut, dtype=torch.float16, device=device),
        requires_grad=False)
    layer.w2_xfp_packed = nn.Parameter(
        torch.randint(0, 2**31 - 1, (E * fpe2,), dtype=torch.int32,
                      device=device), requires_grad=False)
    layer.w2_xfp_codebook = nn.Parameter(
        torch.randn(E * N2, lut, dtype=torch.float16, device=device),
        requires_grad=False)
    layer._xfp_moe_bits = bits
    layer._xfp_moe_K13 = K13
    layer._xfp_moe_N13 = N13
    layer._xfp_moe_K2 = K2
    layer._xfp_moe_N2 = N2
    layer._xfp_moe_E = E
    layer._xfp_moe_fpe13 = fpe13
    layer._xfp_moe_fpe2 = fpe2
    return layer


# ─── A. Generic cache plumbing ────────────────────────────────────────

def test_generic_save_load_roundtrip(cache):
    tensors = {
        "packed": torch.arange(64, dtype=torch.int32),
        "codebook": torch.randn(16, dtype=torch.float16),
    }
    metadata = {"bits": 4, "whatever": "string-val"}
    assert cache.save("prefix.a", "test_method", tensors, metadata) is True

    out = cache.load("prefix.a", "test_method", torch.device("cpu"))
    assert out is not None
    loaded_tensors, meta = out
    assert torch.equal(loaded_tensors["packed"], tensors["packed"])
    assert torch.equal(loaded_tensors["codebook"], tensors["codebook"])
    assert meta["bits"] == "4"
    assert meta["whatever"] == "string-val"
    assert meta["method"] == "test_method"


def test_generic_load_method_mismatch(cache):
    """A file written as method A must NOT load as method B."""
    cache.save("prefix.b", "method_a",
               {"x": torch.zeros(4)}, {"v": "1"})
    out = cache.load("prefix.b", "method_b", torch.device("cpu"))
    assert out is None


def test_generic_load_missing(cache):
    assert cache.load("nope", "m", torch.device("cpu")) is None


def test_generic_read_only_blocks_save(tmp_path):
    ro = MultiQuantWeightCache(
        cache_root=str(tmp_path), model_basename="t",
        cache_key="00" * 8, read_only=True,
    )
    ro.cache_dir.mkdir(parents=True, exist_ok=True)
    ok = ro.save("x", "m", {"t": torch.zeros(1)}, {})
    assert ok is False
    assert not any(ro.cache_dir.glob("*.safetensors"))


# ─── A2. Hash / key ────────────────────────────────────────────────────

def test_cache_key_stable(tmp_path, monkeypatch):
    md = tmp_path / "fakemodel"; md.mkdir()
    (md / "config.json").write_text('{"x":1}')
    (md / "model.safetensors.index.json").write_text('{"weight_map":{"a":"b"}}')
    monkeypatch.setenv("XFP_MOE_LLOYD_ITERS", "5")
    monkeypatch.setenv("XFP_MIN_COS", "0.98")
    reg = _fake_registry()
    hf = MagicMock()
    k1 = MultiQuantWeightCache.compute_cache_key(str(md), reg, hf)
    k2 = MultiQuantWeightCache.compute_cache_key(str(md), reg, hf)
    assert k1 == k2


def test_cache_key_changes_with_policy(tmp_path):
    md = tmp_path / "fakemodel"; md.mkdir()
    (md / "config.json").write_text('{"x":1}')
    hf = MagicMock()
    k_a = MultiQuantWeightCache.compute_cache_key(
        str(md), _fake_registry(dtype_routed="xfp", bits_routed=0), hf)
    k_b = MultiQuantWeightCache.compute_cache_key(
        str(md), _fake_registry(dtype_routed="tq4w", bits_routed=4), hf)
    assert k_a != k_b


def test_cache_key_changes_with_env(tmp_path, monkeypatch):
    md = tmp_path / "fakemodel"; md.mkdir()
    (md / "config.json").write_text('{"x":1}')
    reg = _fake_registry()
    hf = MagicMock()
    monkeypatch.setenv("XFP_MOE_LLOYD_ITERS", "5")
    k_lo = MultiQuantWeightCache.compute_cache_key(str(md), reg, hf)
    monkeypatch.setenv("XFP_MOE_LLOYD_ITERS", "20")
    k_hi = MultiQuantWeightCache.compute_cache_key(str(md), reg, hf)
    assert k_lo != k_hi


def test_cache_key_changes_with_tp(tmp_path):
    md = tmp_path / "fakemodel"; md.mkdir()
    (md / "config.json").write_text('{"x":1}')
    reg = _fake_registry()
    hf = MagicMock()
    k1 = MultiQuantWeightCache.compute_cache_key(
        str(md), reg, hf, tp_size=1)
    k2 = MultiQuantWeightCache.compute_cache_key(
        str(md), reg, hf, tp_size=2)
    assert k1 != k2


# ─── A3. Manifest ──────────────────────────────────────────────────────

def test_manifest_roundtrip(cache):
    cache.write_manifest(_fake_registry(), inventory=["a", "b"])
    assert cache.verify_manifest() is True


def test_manifest_missing(cache):
    assert cache.verify_manifest() is False


def test_manifest_key_mismatch(cache):
    cache.write_manifest(_fake_registry(), inventory=[])
    path = cache.cache_dir / "manifest.json"
    mf = json.loads(path.read_text())
    mf["cache_key"] = "00" * 8
    path.write_text(json.dumps(mf))
    assert cache.verify_manifest() is False


# ─── B. XFP adapter roundtrip ─────────────────────────────────────────

@pytest.mark.parametrize("with_outliers", [True, False])
def test_xfp_linear_roundtrip(cache, with_outliers):
    torch.manual_seed(42)
    src = _fake_linear_layer(with_outliers=with_outliers)
    assert xfp_cache.save_linear(cache, "prefix.qkv", src) is True

    dst = nn.Module()
    assert xfp_cache.load_linear(cache, "prefix.qkv", dst,
                                 torch.device("cpu")) is True
    assert torch.equal(dst.xfp_packed.data, src.xfp_packed.data)
    assert torch.equal(dst.xfp_codebook.data, src.xfp_codebook.data)
    assert dst._xfp_bits == src._xfp_bits
    assert dst._xfp_K == src._xfp_K
    assert dst._xfp_N == src._xfp_N
    assert dst._xfp_has_outliers == with_outliers
    if with_outliers:
        assert torch.equal(dst.xfp_outlier_row.data, src.xfp_outlier_row.data)
        assert torch.equal(dst.xfp_outlier_col.data, src.xfp_outlier_col.data)
        assert torch.equal(dst.xfp_outlier_val.data, src.xfp_outlier_val.data)


def test_xfp_moe_roundtrip(cache):
    torch.manual_seed(1234)
    src = _fake_moe_layer()
    assert xfp_cache.save_moe(cache, "prefix.experts", src) is True

    dst = nn.Module()
    assert xfp_cache.load_moe(cache, "prefix.experts", dst,
                              torch.device("cpu")) is True
    for attr in ("w13_xfp_packed", "w13_xfp_codebook",
                 "w2_xfp_packed", "w2_xfp_codebook"):
        assert torch.equal(getattr(dst, attr).data,
                           getattr(src, attr).data)
    for attr in ("_xfp_moe_bits", "_xfp_moe_K13", "_xfp_moe_N13",
                 "_xfp_moe_K2", "_xfp_moe_N2", "_xfp_moe_E",
                 "_xfp_moe_fpe13", "_xfp_moe_fpe2"):
        assert getattr(dst, attr) == getattr(src, attr)
    assert dst._xfp_moe_packed is True


def test_xfp_linear_miss_on_moe_file(cache):
    """An MoE-kind file must not load as Linear."""
    src = _fake_moe_layer()
    xfp_cache.save_moe(cache, "mixup", src)
    dst = nn.Module()
    assert xfp_cache.load_linear(cache, "mixup", dst,
                                 torch.device("cpu")) is False
