# SPDX-License-Identifier: Apache-2.0
"""TurboQuant configuration."""

from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV-cache quantization.

    Args:
        head_dim: Attention head dimension (e.g. 64, 96, 128).
        total_bits: Total bits per coordinate (3 or 4).
            MSE stage uses (total_bits - 1) bits, QJL uses 1 bit.
        seed: Base seed for deterministic random matrix generation.
            Actual seed per layer = seed + layer_idx * 1337.
    """
    head_dim: int = 128
    total_bits: int = 3
    seed: int = 42

    @property
    def mse_bits(self) -> int:
        return max(self.total_bits - 1, 1)

    @property
    def n_centroids(self) -> int:
        return 2 ** self.mse_bits

    @property
    def key_packed_size(self) -> int:
        """Packed bytes for a single compressed KEY vector.

        Layout:
          - MSE indices: ceil(head_dim * mse_bits / 8) bytes
          - QJL signs:   ceil(head_dim / 8) bytes
          - vec_norm:     2 bytes (float16)
          - res_norm:     2 bytes (float16)
        """
        import math
        mse_bytes = math.ceil(self.head_dim * self.mse_bits / 8)
        qjl_bytes = math.ceil(self.head_dim / 8)
        norm_bytes = 4  # 2x float16
        return mse_bytes + qjl_bytes + norm_bytes

    @property
    def packed_size(self) -> int:
        """Packed bytes per token per head in the KV cache.

        Phase 1: Values stored as FP16, so packed_size = head_dim * 2.
        Keys use only key_packed_size bytes of this space.
        Net compression for K+V: ~2x vs FP16 (keys 4.9x, values 1x).
        """
        return self.head_dim * 2  # FP16 values need full space

    @property
    def cache_head_size(self) -> int:
        """Effective head_size for KV-cache allocation (uint8 bytes).

        Both K and V are TQ-compressed to the same packed size.
        This replaces head_size in get_kv_cache_shape, making the
        cache physically smaller. Symmetric layout — no arch change.

        For D=256, TQ3: 100 bytes/slot → 5.1× smaller than BF16
        For D=256, TQ4: 128 bytes/slot → 4.0× smaller than BF16
        For D=128, TQ3:  52 bytes/slot → 4.9× smaller than BF16
        """
        return self.key_packed_size

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> "TurboQuantConfig":
        if cache_dtype == "tq3":
            return TurboQuantConfig(head_dim=head_dim, total_bits=3)
        elif cache_dtype == "tq4":
            return TurboQuantConfig(head_dim=head_dim, total_bits=4)
        else:
            raise ValueError(f"Unknown TurboQuant cache dtype: {cache_dtype}")
