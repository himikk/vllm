# SPDX-License-Identifier: Apache-2.0
"""RotorQuant configuration."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.multiquant.base import KVQuantizerConfig


@dataclass
class RotorQuantConfig(KVQuantizerConfig):
    """Configuration for RotorQuant KV-cache quantization.

    Uses Clifford Cl(3,0) rotors instead of dense rotation matrices.
    44× fewer parameters, 10-19× faster kernels, identical quality.
    """

    @property
    def n_groups(self) -> int:
        """Number of Cl(3,0) multivector groups (vectors split into groups of 3)."""
        return (self.head_dim + 2) // 3

    @classmethod
    def from_cache_dtype(cls, cache_dtype: str, head_dim: int) -> RotorQuantConfig:
        if cache_dtype == "rq2":
            return cls(head_dim=head_dim, total_bits=2)
        elif cache_dtype == "rq3":
            return cls(head_dim=head_dim, total_bits=3)
        elif cache_dtype == "rq4":
            return cls(head_dim=head_dim, total_bits=4)
        raise ValueError(f"Unknown RotorQuant cache dtype: {cache_dtype}")
