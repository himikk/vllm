# SPDX-License-Identifier: Apache-2.0
"""Unit tests for xfp_auto_select (CPU, no GPU needed)."""

from __future__ import annotations

import pytest
import torch

from vllm.multiquant.xfp.xfp_pack import xfp_auto_select


def test_gaussian_prefers_low_bits() -> None:
    """A well-behaved Gaussian: xfp3 (median cos ~0.982) passes the
    default 0.98 gate while xfp2 (median cos ~0.94) doesn't."""
    torch.manual_seed(10)
    W = torch.randn(256, 512, dtype=torch.float32) * 0.01
    bits = xfp_auto_select(W, candidates=(2, 3, 4), min_cos=0.98)
    assert bits == 3, f"expected xfp3 on clean Gaussian at min_cos=0.98, got xfp{bits}"


def test_gaussian_loose_threshold_picks_2() -> None:
    """With a loose threshold (0.93), even xfp2 passes on Gaussian."""
    torch.manual_seed(10)
    W = torch.randn(256, 512, dtype=torch.float32) * 0.01
    bits = xfp_auto_select(W, candidates=(2, 3, 4), min_cos=0.93)
    assert bits == 2, f"expected xfp2 at min_cos=0.93, got xfp{bits}"


def test_heavy_tail_prefers_high_bits() -> None:
    """Without outlier extraction, a strict threshold (0.995) forces xfp4
    because even xfp3's median cos (0.992) doesn't pass."""
    torch.manual_seed(11)
    W = torch.randn(128, 256, dtype=torch.float32) * 0.01
    mask = torch.rand_like(W) < 0.05
    W = torch.where(mask, W * 50.0, W)
    bits = xfp_auto_select(
        W, candidates=(2, 3, 4), min_cos=0.995, outlier_sigma=None,
    )
    assert bits == 4, f"expected xfp4 at min_cos=0.995 w/o outlier extraction, got xfp{bits}"


def test_heavy_tail_with_outlier_extraction() -> None:
    """Same heavy-tail distribution but with outlier extraction enabled —
    the bulk becomes clean and lower bits should pass."""
    torch.manual_seed(11)
    W = torch.randn(128, 256, dtype=torch.float32) * 0.01
    mask = torch.rand_like(W) < 0.05
    W = torch.where(mask, W * 50.0, W)
    bits = xfp_auto_select(
        W, candidates=(2, 3, 4), min_cos=0.97, outlier_sigma=4.0,
    )
    assert bits < 4, (
        f"expected < xfp4 with outlier extraction on heavy-tail, got xfp{bits}"
    )


def test_always_returns_valid_candidate() -> None:
    """Auto should always return one of the candidates."""
    torch.manual_seed(12)
    W = torch.randn(64, 128, dtype=torch.float32)
    for min_cos in (0.5, 0.9, 0.99, 0.999):
        bits = xfp_auto_select(W, candidates=(2, 3, 4), min_cos=min_cos)
        assert bits in (2, 3, 4), f"invalid bits {bits} for min_cos={min_cos}"


def test_strict_threshold_forces_max() -> None:
    """With an impossibly strict threshold, auto falls back to max candidate."""
    torch.manual_seed(13)
    W = torch.randn(64, 128, dtype=torch.float32)
    bits = xfp_auto_select(
        W, candidates=(2, 3, 4), min_cos=0.9999,
    )
    assert bits == 4


def test_single_candidate() -> None:
    """With only one candidate, auto returns it."""
    torch.manual_seed(14)
    W = torch.randn(32, 64, dtype=torch.float32)
    bits = xfp_auto_select(W, candidates=(3,))
    assert bits == 3
