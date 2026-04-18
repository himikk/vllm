# SPDX-License-Identifier: Apache-2.0
"""TurboQuant v2 quantizer — WHT block compression.

Uses Walsh-Hadamard Transform (WHT) on configurable block sizes
instead of full D×D random rotation. No per-layer state needed.

Reference: github.com/animehacker/llama-turboquant
"""

from __future__ import annotations

import torch
from torch import Tensor

from vllm.multiquant.base import KVQuantizer, KVQuantizerConfig
from vllm.multiquant.shared.centroids import get_wht_centroids, get_wht_thresholds
from vllm.multiquant.shared.wht import wht_forward, wht_inverse
from vllm.multiquant.turboquant.wht_config import TurboQuantWHTConfig


class TurboQuantWHTQuantizer(KVQuantizer):
    """WHT/block-rotation TurboQuant quantizer.

    WHT mode: No per-layer state — WHT is deterministic.
    Random mode: Stores [n_blocks, block_size, block_size] rotation matrices per layer.
    Both use universal centroids (8 values for 3-bit).
    """

    def init_buffers(self, head_dim: int, seed: int) -> dict[str, Tensor]:
        """WHT: no buffers. Random: block rotation matrices."""
        # Check if we have a config with rotation_type
        # This is called from attention.py with just head_dim and seed
        return {}

    def pack(
        self,
        vector: Tensor,
        buffers: dict[str, Tensor],
        config: KVQuantizerConfig,
    ) -> Tensor:
        """Compress float vectors to packed uint8 using WHT + block quantization.

        Args:
            vector: [N, D] float tensor
            buffers: unused (WHT has no per-layer state)
            config: TurboQuantWHTConfig

        Returns:
            [N, packed_size] uint8 tensor
        """
        assert isinstance(config, TurboQuantWHTConfig)
        return pack_wht(vector.float(), config)

    def unpack(
        self,
        packed: Tensor,
        buffers: dict[str, Tensor],
        config: KVQuantizerConfig,
    ) -> Tensor:
        """Decompress packed uint8 back to float vector."""
        assert isinstance(config, TurboQuantWHTConfig)
        return unpack_wht(packed, config)

    def attention_score(
        self,
        query: Tensor,
        packed_key: Tensor,
        buffers: dict[str, Tensor],
        config: KVQuantizerConfig,
    ) -> Tensor:
        """Compute Q·K score from compressed key.

        For WHT mode: transform Q into WHT space, then dot with
        reconstructed K in WHT space (no inverse WHT needed for score).
        """
        assert isinstance(config, TurboQuantWHTConfig)
        # Decompress K and dot with Q
        k_recon = unpack_wht(packed_key, config)
        return (query.float() * k_recon).sum(dim=-1)


def pack_wht(
    x: Tensor, config: TurboQuantWHTConfig,
    Pi_blocks: Tensor | None = None,
) -> Tensor:
    """Pack float vectors using WHT or block-rotation compression.

    Args:
        x: [N, D] float32 tensor
        config: WHT config with block_size and total_bits
        Pi_blocks: [n_blocks, block_size, block_size] for random rotation mode

    Returns:
        [N, packed_size] uint8 tensor
    """
    N, D = x.shape
    bs = config.block_size
    bits = config.mse_bits
    n_blocks = D // bs

    # Step 1: Block transform (WHT or random rotation)
    if Pi_blocks is not None:
        # Random block rotation: reshape to blocks, bmm with rotation matrices
        x_blocks = x.reshape(N, n_blocks, bs)  # [N, n_blocks, bs]
        # bmm: [N*n_blocks, 1, bs] @ [N*n_blocks, bs, bs] → [N*n_blocks, 1, bs]
        # More efficient: einsum or manual broadcast
        # Pi_blocks: [n_blocks, bs, bs], x_blocks: [N, n_blocks, bs]
        # rotated[n, b, :] = x_blocks[n, b, :] @ Pi_blocks[b, :, :].T
        blocks = torch.einsum('nbi,bij->nbj', x_blocks, Pi_blocks)
    else:
        # WHT: fixed Hadamard butterfly transform
        rotated = wht_forward(x, block_size=bs)  # [N, D]
        blocks = rotated.reshape(N, n_blocks, bs)  # [N, n_blocks, bs]

    # Step 2: Per-block amax normalization
    # Cache centroids/thresholds on GPU (host→device transfer is NOT graph-safe)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)  # [N, n_blocks, 1]
    _cache_key = (bits, x.device)
    if not hasattr(pack_wht, '_gpu_cache'):
        pack_wht._gpu_cache = {}
    if _cache_key not in pack_wht._gpu_cache:
        c = get_wht_centroids(bits).to(x.device)
        t = get_wht_thresholds(bits).to(x.device)
        pack_wht._gpu_cache[_cache_key] = (c, t)
    centroids, thresholds = pack_wht._gpu_cache[_cache_key]
    outermost = centroids[-1]  # tensor, no .item()
    gamma = (amax / outermost).squeeze(-1)  # [N, n_blocks]
    normalized = blocks / (gamma.unsqueeze(-1) + 1e-10)
    # idx[i] = number of thresholds that normalized[i] exceeds
    idx = (normalized.unsqueeze(-1) > thresholds).sum(dim=-1).to(torch.uint8)
    # [N, n_blocks, bs] uint8 indices

    # Step 4: Bitpack per block — VECTORIZED (no Python loops, graph-safe)
    bpb = config.bytes_per_block
    idx_int = idx.to(torch.int32)  # [N, n_blocks, bs]

    # Cache constant shift tensors on GPU (graph-safe: no host→device during capture)
    if not hasattr(pack_wht, '_shift_cache'):
        pack_wht._shift_cache = {}
    _sk = x.device
    if _sk not in pack_wht._shift_cache:
        pack_wht._shift_cache[_sk] = {
            'qs': torch.tensor([0, 2, 4, 6], device=_sk, dtype=torch.int32),
            'qr': torch.arange(8, device=_sk, dtype=torch.int32),
            '4b': torch.tensor([0, 4], device=_sk, dtype=torch.int32),
        }
    _shifts = pack_wht._shift_cache[_sk]

    # Pre-allocate output (no torch.cat — graph-safe, less memory)
    packed = torch.empty(N, n_blocks, bpb, dtype=torch.uint8, device=x.device)
    gamma_fp16 = gamma.to(torch.float16).view(torch.uint8).reshape(N, n_blocks, 2)

    if bits == 3:
        qs_bytes = bs * 2 // 8
        qr_bytes = bs // 8
        # Write qs directly into packed[:, :, :qs_bytes]
        low2 = (idx_int & 0x3).reshape(N, n_blocks, qs_bytes, 4)
        packed[:, :, :qs_bytes] = (low2 << _shifts['qs']).sum(dim=-1).to(torch.uint8)
        # Write qr into packed[:, :, qs_bytes:qs_bytes+qr_bytes]
        hi1 = ((idx_int >> 2) & 1).reshape(N, n_blocks, qr_bytes, 8)
        packed[:, :, qs_bytes:qs_bytes+qr_bytes] = (
            hi1 << _shifts['qr']).sum(dim=-1).to(torch.uint8)
        # Write gamma
        packed[:, :, qs_bytes+qr_bytes:] = gamma_fp16
    elif bits == 4:
        n_bytes = bs // 2
        idx_pairs = idx_int.reshape(N, n_blocks, n_bytes, 2)
        packed[:, :, :n_bytes] = (
            (idx_pairs[..., 0] & 0xF) | ((idx_pairs[..., 1] & 0xF) << 4)
        ).to(torch.uint8)
        packed[:, :, n_bytes:] = gamma_fp16
    elif bits == 2:
        n_bytes = bs // 4
        idx_quads = idx_int.reshape(N, n_blocks, n_bytes, 4)
        packed[:, :, :n_bytes] = (
            (idx_quads & 0x3) << _shifts['qs']).sum(dim=-1).to(torch.uint8)
        packed[:, :, n_bytes:] = gamma_fp16
    else:
        raise ValueError(f"WHT pack: unsupported bits={bits}")

    return packed.reshape(N, -1)


def unpack_wht(
    packed: Tensor, config: TurboQuantWHTConfig,
    Pi_blocks: Tensor | None = None,
) -> Tensor:
    """Unpack compressed uint8 back to float vectors.

    Args:
        packed: [N, packed_size] uint8 tensor
        config: WHT config
        Pi_blocks: [n_blocks, block_size, block_size] for random rotation mode

    Returns:
        [N, D] float32 tensor
    """
    N = packed.shape[0]
    D = config.head_dim
    bs = config.block_size
    bits = config.mse_bits
    n_blocks = D // bs
    bpb = config.bytes_per_block

    blocks_packed = packed.reshape(N, n_blocks, bpb)

    # Cache centroids on GPU (host→device NOT graph-safe)
    _ukey = (bits, packed.device)
    if not hasattr(unpack_wht, '_gpu_cache'):
        unpack_wht._gpu_cache = {}
    if _ukey not in unpack_wht._gpu_cache:
        unpack_wht._gpu_cache[_ukey] = get_wht_centroids(bits).to(packed.device)
    centroids = unpack_wht._gpu_cache[_ukey]

    # Unpack indices — VECTORIZED (graph-safe: cached shift tensors)
    if not hasattr(unpack_wht, '_shift_cache'):
        unpack_wht._shift_cache = {}
    _sk = packed.device
    if _sk not in unpack_wht._shift_cache:
        unpack_wht._shift_cache[_sk] = {
            'qs': torch.tensor([0, 2, 4, 6], device=_sk, dtype=torch.int32),
            'qr': torch.arange(8, device=_sk, dtype=torch.int32),
            '4b': torch.tensor([0, 4], device=_sk, dtype=torch.int32),
        }
    _sh = unpack_wht._shift_cache[_sk]

    if bits == 3:
        qs_bytes = bs * 2 // 8
        qr_bytes = bs // 8
        qs_raw = blocks_packed[:, :, :qs_bytes].to(torch.int32)
        low2 = ((qs_raw.unsqueeze(-1) >> _sh['qs']) & 0x3).reshape(N, n_blocks, bs)
        qr_raw = blocks_packed[:, :, qs_bytes:qs_bytes+qr_bytes].to(torch.int32)
        hi1 = ((qr_raw.unsqueeze(-1) >> _sh['qr']) & 1).reshape(N, n_blocks, bs)
        idx = (low2 | (hi1 << 2)).to(torch.long)
        gamma_off = qs_bytes + qr_bytes
    elif bits == 4:
        n_bytes = bs // 2
        raw = blocks_packed[:, :, :n_bytes].to(torch.int32)
        idx = ((raw.unsqueeze(-1) >> _sh['4b']) & 0xF).reshape(N, n_blocks, bs).to(torch.long)
        gamma_off = n_bytes
    elif bits == 2:
        n_bytes = bs // 4
        raw = blocks_packed[:, :, :n_bytes].to(torch.int32)
        idx = ((raw.unsqueeze(-1) >> _sh['qs']) & 0x3).reshape(N, n_blocks, bs).to(torch.long)
        gamma_off = n_bytes
    else:
        raise ValueError(f"WHT unpack: unsupported bits={bits}")

    # Unpack gamma (fp16) — vectorized
    gamma_raw = blocks_packed[:, :, gamma_off:gamma_off+2].contiguous()
    gamma_fp16 = gamma_raw.view(torch.uint8).reshape(N * n_blocks, 2).view(
        torch.float16).reshape(N, n_blocks).float()

    # Reconstruct: centroid lookup + scale + inverse transform
    values_rot = centroids[idx] * gamma_fp16.unsqueeze(-1)  # [N, n_blocks, bs]

    if Pi_blocks is not None:
        # Inverse random rotation: multiply by Pi_blocks^T per block
        # values_rot: [N, n_blocks, bs], Pi_blocks: [n_blocks, bs, bs]
        # result[n, b, :] = values_rot[n, b, :] @ Pi_blocks[b, :, :]^T
        values_orig = torch.einsum('nbj,bij->nbi', values_rot, Pi_blocks)
        return values_orig.reshape(N, D)
    else:
        # Inverse WHT
        values_flat = values_rot.reshape(N, D)
        return wht_inverse(values_flat, block_size=bs)
