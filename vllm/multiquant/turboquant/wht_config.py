# SPDX-License-Identifier: Apache-2.0
"""TurboQuant v2 configuration — WHT block compression."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from vllm.multiquant.base import KVQuantizerConfig


@dataclass
class TurboQuantWHTConfig(KVQuantizerConfig):
    """Configuration for WHT/block-rotation TurboQuant KV-cache quantization.

    Uses Walsh-Hadamard Transform (tq3w/tq4w) or block-diagonal random
    orthogonal rotation (tq3r/tq4r) on fixed-size blocks instead of
    full D×D random orthogonal rotation. WHT needs no per-layer state;
    random rotation stores [n_blocks, block_size, block_size] per layer.

    Block format (per block_size values):
        qs: lower 2 bits of 3-bit indices (block_size/4 bytes)
        qr: upper 1 bit of 3-bit indices (block_size/8 bytes)
        gamma: per-block FP16 scale factor (2 bytes)
    """
    block_size: int = 32  # WHT block size (must be power of 2)
    rotation_type: str = "wht"  # "wht" (Hadamard) or "random" (block-diagonal)

    @property
    def mse_bits(self) -> int:
        """WHT uses all bits for MSE (no separate QJL correction)."""
        return self.total_bits

    @property
    def n_centroids(self) -> int:
        return 2 ** self.mse_bits

    @property
    def bytes_per_block(self) -> int:
        """Packed bytes for one WHT block."""
        bs = self.block_size
        bits = self.mse_bits
        if bits <= 2:
            # 2-bit: bs/4 bytes for indices + 2 bytes gamma
            return bs * bits // 8 + 2
        elif bits == 3:
            # 3-bit split: bs*2/8 (lower 2 bits) + bs/8 (upper 1 bit) + 2 gamma
            return bs * 2 // 8 + bs // 8 + 2
        elif bits == 4:
            # 4-bit: bs/2 bytes for indices + 2 bytes gamma
            return bs // 2 + 2
        else:
            return math.ceil(bs * bits / 8) + 2

    @property
    def n_blocks(self) -> int:
        """Number of WHT blocks per head vector."""
        return self.head_dim // self.block_size

    @property
    def key_packed_size(self) -> int:
        """Packed bytes for a single compressed key vector."""
        return self.n_blocks * self.bytes_per_block

    @property
    def cache_head_size(self) -> int:
        return self.key_packed_size

    @classmethod
    def from_cache_dtype(
        cls, cache_dtype: str, head_dim: int
    ) -> TurboQuantWHTConfig:
        """Parse dtype string like 'tq3w', 'tq3r', 'tq3w:bs64'."""
        # Determine rotation type from suffix: 'w' = WHT, 'r' = random
        base_dtype = cache_dtype.split(':')[0] if ':' in cache_dtype else cache_dtype
        if base_dtype.endswith('r'):
            rotation_type = "random"
        else:
            rotation_type = "wht"
        # Strip suffix and 'tq' prefix to get bits
        base = base_dtype.rstrip('wr').replace('tq', '')
        block_size = 32  # default
        if ':' in cache_dtype:
            parts = cache_dtype.split(':')
            for p in parts[1:]:
                if p.startswith('bs'):
                    block_size = int(p[2:])
        total_bits = int(base)
        assert block_size & (block_size - 1) == 0, \
            f"block_size must be power of 2, got {block_size}"
        assert head_dim % block_size == 0, \
            f"head_dim={head_dim} not divisible by block_size={block_size}"
        return cls(
            head_dim=head_dim,
            total_bits=total_bits,
            block_size=block_size,
            rotation_type=rotation_type,
        )
