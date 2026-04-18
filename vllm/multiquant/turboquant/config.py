# SPDX-License-Identifier: Apache-2.0
"""TurboQuant configuration."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.multiquant.base import KVQuantizerConfig


@dataclass
class TurboQuantConfig(KVQuantizerConfig):
    """Configuration for TurboQuant KV-cache quantization.

    Uses dense d×d orthogonal rotation matrix (Haar-distributed).
    """

    @property
    def packed_size(self) -> int:
        """Legacy: packed bytes per token per head (FP16 values)."""
        return self.head_dim * 2

    @classmethod
    def from_cache_dtype(cls, cache_dtype: str, head_dim: int) -> TurboQuantConfig:
        if cache_dtype == "tq3":
            return cls(head_dim=head_dim, total_bits=3)
        elif cache_dtype == "tq4":
            return cls(head_dim=head_dim, total_bits=4)
        raise ValueError(f"Unknown TurboQuant cache dtype: {cache_dtype}")
