# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV-Cache Manager.

Manages a separate compressed K-cache alongside vLLM's standard V-cache.
Keys are stored as packed TQ indices + QJL signs (100-128 bytes/vector).
Values use the standard vLLM cache (BF16 or FP8).

On attention read: decompress needed K-blocks into a temporary BF16 buffer,
then pass to FlashInfer/FlashAttn as usual.

Memory layout:
  Standard vLLM cache: used for V only (kv_cache[1])
  TQ K-cache: separate uint8 tensor, same block_table/slot_mapping
"""

import math
from typing import Optional

import torch

from vllm.turboquant.config import TurboQuantConfig


class TQCacheManager:
    """Per-layer manager for compressed K-cache.

    Allocated once during model init, shares block_table with vLLM's cache.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        tq_config: TurboQuantConfig,
        device: torch.device,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.tq_config = tq_config
        self.device = device

        # Packed size per key vector
        self.k_packed_size = tq_config.key_packed_size

        # Compressed K-cache: (num_blocks, block_size, num_kv_heads, k_packed_size)
        self.k_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, self.k_packed_size,
            dtype=torch.uint8, device=device,
        )

        # Pre-allocated decompression buffer for decode
        # Size: max one sequence worth of decompressed keys
        # (num_blocks, block_size, num_kv_heads, head_size) in BF16
        self._decomp_buffer = None

        # Memory savings
        k_compressed_bytes = num_blocks * block_size * num_kv_heads * self.k_packed_size
        k_full_bytes = num_blocks * block_size * num_kv_heads * head_size * 2  # BF16
        self.memory_saved_bytes = k_full_bytes - k_compressed_bytes

    @property
    def compression_ratio(self) -> float:
        """How much smaller the K-cache is vs BF16."""
        return (self.head_size * 2) / self.k_packed_size

    def compress_and_store(
        self,
        key: torch.Tensor,      # (num_tokens, num_kv_heads, head_size)
        slot_mapping: torch.Tensor,  # (num_tokens,)
        Pi: torch.Tensor,       # (head_size, head_size) float32
        S: torch.Tensor,        # (head_size, head_size) float32
        centroids: torch.Tensor,  # (n_centroids,) float32
    ):
        """Compress keys and store directly in packed K-cache.

        PyTorch reference implementation. Will be replaced by CUDA kernel.
        """
        num_tokens, num_heads, D = key.shape
        mse_bits = self.tq_config.mse_bits

        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            block_idx = slot // self.block_size
            block_off = slot % self.block_size

            for h in range(num_heads):
                k = key[i, h].float()

                # TQ compress
                vn = k.norm()
                xh = k / (vn + 1e-8)
                rot = xh @ Pi.T
                diffs = rot.unsqueeze(-1) - centroids
                idx = diffs.abs().argmin(dim=-1).to(torch.uint8)
                yh = centroids[idx.long()]
                xm = yh @ Pi
                r = xh - xm
                rn = r.norm()
                proj = r @ S.T
                signs = (proj >= 0).to(torch.uint8)

                # Pack: MSE indices + QJL signs + norms
                packed = self._pack(idx, signs, vn, rn, D, mse_bits)
                self.k_cache[block_idx, block_off, h, :len(packed)] = packed

    def decompress_blocks(
        self,
        block_indices: torch.Tensor,  # (num_blocks_needed,) int32
        Pi: torch.Tensor,
        S: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Decompress specified blocks into BF16 key tensor.

        Returns: (num_blocks_needed, block_size, num_kv_heads, head_size) BF16
        """
        num_blocks_needed = block_indices.shape[0]
        D = self.head_size
        mse_bits = self.tq_config.mse_bits
        correction_scale = math.sqrt(math.pi / 2) / D

        out = torch.zeros(
            num_blocks_needed, self.block_size, self.num_kv_heads, D,
            dtype=torch.bfloat16, device=self.device,
        )

        for bi, block_idx in enumerate(block_indices):
            block_idx = block_idx.item()
            for t in range(self.block_size):
                for h in range(self.num_kv_heads):
                    packed = self.k_cache[block_idx, t, h]
                    idx, signs_01, vn, rn = self._unpack(
                        packed, D, mse_bits)

                    # Decompress
                    c_idx = centroids[idx.long()]
                    xm = c_idx @ Pi  # unrotate MSE
                    signs_f = signs_01.float() * 2 - 1  # 0,1 → -1,+1
                    xq = correction_scale * rn * (signs_f @ S)  # QJL correction
                    out[bi, t, h] = (vn * (xm + xq)).to(torch.bfloat16)

        return out

    def _pack(self, idx, signs, vn, rn, D, mse_bits):
        """Pack indices + signs + norms into uint8 bytes."""
        mse_bytes = math.ceil(D * mse_bits / 8)
        qjl_bytes = math.ceil(D / 8)

        parts = []

        # MSE indices
        if mse_bits == 2:
            packed_mse = torch.zeros(mse_bytes, dtype=torch.uint8, device=idx.device)
            for j in range(0, D, 4):
                val = torch.zeros(1, dtype=torch.uint8, device=idx.device)
                for k in range(min(4, D - j)):
                    val |= (idx[j + k] & 0x3) << (k * 2)
                packed_mse[j // 4] = val
            parts.append(packed_mse)
        elif mse_bits == 3:
            packed_mse = torch.zeros(mse_bytes, dtype=torch.uint8, device=idx.device)
            for j in range(D):
                bit_off = j * 3
                byte_idx = bit_off // 8
                bit_idx = bit_off % 8
                v = idx[j].to(torch.int32)
                packed_mse[byte_idx] |= ((v << bit_idx) & 0xFF).to(torch.uint8)
                if bit_idx > 5 and byte_idx + 1 < mse_bytes:
                    packed_mse[byte_idx + 1] |= ((v >> (8 - bit_idx)) & 0xFF).to(
                        torch.uint8)
            parts.append(packed_mse)
        else:
            parts.append(idx[:mse_bytes])

        # QJL signs
        packed_signs = torch.zeros(qjl_bytes, dtype=torch.uint8, device=signs.device)
        for j in range(0, D, 8):
            val = torch.zeros(1, dtype=torch.uint8, device=signs.device)
            for k in range(min(8, D - j)):
                val |= (signs[j + k] & 1) << k
            packed_signs[j // 8] = val
        parts.append(packed_signs)

        # Norms (float16 bytes)
        parts.append(vn.half().view(torch.uint8))
        parts.append(rn.half().view(torch.uint8))

        return torch.cat(parts)

    def _unpack(self, packed, D, mse_bits):
        """Unpack indices + signs + norms from uint8 bytes."""
        mse_bytes = math.ceil(D * mse_bits / 8)
        qjl_bytes = math.ceil(D / 8)
        mask = (1 << mse_bits) - 1

        # MSE indices
        packed_mse = packed[:mse_bytes]
        idx = torch.zeros(D, dtype=torch.uint8, device=packed.device)
        if mse_bits == 2:
            for j in range(D):
                byte_idx = j // 4
                bit_idx = (j % 4) * 2
                if byte_idx < len(packed_mse):
                    idx[j] = (packed_mse[byte_idx] >> bit_idx) & mask
        elif mse_bits == 3:
            for j in range(D):
                bit_off = j * 3
                byte_idx = bit_off // 8
                bit_idx = bit_off % 8
                val = packed_mse[byte_idx].to(torch.int32) >> bit_idx
                if bit_idx > 5 and byte_idx + 1 < len(packed_mse):
                    val |= packed_mse[byte_idx + 1].to(torch.int32) << (8 - bit_idx)
                idx[j] = (val & mask).to(torch.uint8)

        # Signs
        packed_signs = packed[mse_bytes:mse_bytes + qjl_bytes]
        signs = torch.zeros(D, dtype=torch.uint8, device=packed.device)
        for j in range(D):
            byte_idx = j // 8
            bit_idx = j % 8
            if byte_idx < len(packed_signs):
                signs[j] = (packed_signs[byte_idx] >> bit_idx) & 1

        # Norms
        norm_off = mse_bytes + qjl_bytes
        vn = packed[norm_off:norm_off + 2].view(torch.float16).float().item()
        rn = packed[norm_off + 2:norm_off + 4].view(torch.float16).float().item()

        return idx, signs, vn, rn

    def memory_info(self) -> dict:
        """Return memory usage info."""
        k_bytes = self.k_cache.numel()
        k_full = self.num_blocks * self.block_size * self.num_kv_heads * self.head_size * 2
        return {
            "k_compressed_mb": k_bytes / 1e6,
            "k_full_bf16_mb": k_full / 1e6,
            "saved_mb": (k_full - k_bytes) / 1e6,
            "compression_ratio": self.compression_ratio,
        }
