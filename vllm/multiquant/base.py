# SPDX-License-Identifier: Apache-2.0
"""MultiQuant: Abstract base classes for KV-cache quantizers."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor


@dataclass
class KVQuantizerConfig(ABC):
    """Base configuration for any KV-cache quantizer.

    Subclasses define algorithm-specific parameters (rotation type,
    codebook style, etc.) but must implement the packed-size properties
    so the block manager can allocate the right amount of memory.
    """
    head_dim: int
    total_bits: int
    seed: int = 42

    @property
    def mse_bits(self) -> int:
        """Bits used by MSE quantization stage (total_bits - 1 for QJL)."""
        return max(self.total_bits - 1, 1)

    @property
    def n_centroids(self) -> int:
        return 2 ** self.mse_bits

    @property
    def key_packed_size(self) -> int:
        """Packed bytes for a single compressed key vector.

        Layout: MSE indices | QJL signs | norms
        """
        mse_bytes = math.ceil(self.head_dim * self.mse_bits / 8)
        qjl_bytes = math.ceil(self.head_dim / 8)
        norm_bytes = 4  # 2× float16
        return mse_bytes + qjl_bytes + norm_bytes

    @property
    def cache_head_size(self) -> int:
        """Effective head_size for KV-cache allocation (uint8 bytes)."""
        return self.key_packed_size

    @classmethod
    @abstractmethod
    def from_cache_dtype(cls, cache_dtype: str, head_dim: int) -> Self:
        """Factory: dtype string + head_dim → config instance."""
        ...


class KVQuantizer(ABC):
    """Abstract interface for KV-cache quantization algorithms.

    Each quantizer provides:
    - Buffer initialization (rotation matrices, rotors, centroids, etc.)
    - Pack: compress a key/value vector to uint8
    - Unpack: decompress uint8 back to float
    - Attention score: compute Q·K score directly from compressed data
    """

    @abstractmethod
    def init_buffers(
        self, head_dim: int, seed: int
    ) -> dict[str, Tensor]:
        """Create quantizer-specific buffers (registered on the layer).

        Returns dict of name → tensor, e.g.:
          {"Pi": rotation_matrix, "S": qjl_matrix, "centroids": codebook}
        """
        ...

    @abstractmethod
    def pack(
        self,
        vector: Tensor,
        buffers: dict[str, Tensor],
        config: KVQuantizerConfig,
    ) -> Tensor:
        """Compress a float vector to packed uint8.

        Args:
            vector: [num_tokens, head_dim] float tensor
            buffers: quantizer buffers from init_buffers()
            config: quantizer config
        Returns:
            [num_tokens, packed_size] uint8 tensor
        """
        ...

    @abstractmethod
    def unpack(
        self,
        packed: Tensor,
        buffers: dict[str, Tensor],
        config: KVQuantizerConfig,
    ) -> Tensor:
        """Decompress packed uint8 back to float vector.

        Args:
            packed: [num_tokens, packed_size] uint8 tensor
            buffers: quantizer buffers
            config: quantizer config
        Returns:
            [num_tokens, head_dim] float tensor
        """
        ...

    @abstractmethod
    def attention_score(
        self,
        query: Tensor,
        packed_key: Tensor,
        buffers: dict[str, Tensor],
        config: KVQuantizerConfig,
    ) -> Tensor:
        """Compute attention score Q·K directly from compressed key.

        Uses two-term estimator: MSE term + QJL correction term.

        Args:
            query: [head_dim] float tensor (single query)
            packed_key: [num_keys, packed_size] uint8 tensor
            buffers: quantizer buffers
            config: quantizer config
        Returns:
            [num_keys] float tensor of attention scores
        """
        ...
