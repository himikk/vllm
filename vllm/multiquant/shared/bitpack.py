# SPDX-License-Identifier: Apache-2.0
"""Bit-packing utilities — shared by TQ + RQ.

Pack MSE indices + QJL signs + norms into uint8 vectors.
Unpack back to float for decompression.

Layout per key vector:
  [MSE indices: ceil(d*mse_bits/8) bytes]
  [QJL signs:   ceil(d/8) bytes]
  [vec_norm:    2 bytes (float16)]
  [res_norm:    2 bytes (float16)]
"""

import struct

import torch


def pack_vector(
    indices: torch.Tensor,
    signs: torch.Tensor,
    vec_norm: float,
    res_norm: float,
    head_dim: int,
    mse_bits: int,
) -> torch.Tensor:
    """Pack quantized vector into uint8.

    Args:
        indices: (d,) long tensor of MSE centroid indices
        signs: (d,) float tensor of QJL signs (+1/-1)
        vec_norm: scalar, vector norm
        res_norm: scalar, residual norm
        head_dim: original dimension
        mse_bits: bits per MSE index

    Returns:
        (packed_size,) uint8 tensor
    """
    import math
    mse_bytes = math.ceil(head_dim * mse_bits / 8)
    qjl_bytes = math.ceil(head_dim / 8)
    packed_size = mse_bytes + qjl_bytes + 4
    packed = torch.zeros(packed_size, dtype=torch.uint8, device=indices.device)

    mask = (1 << mse_bits) - 1

    # Pack MSE indices as bitstream (handles any mse_bits, including 3)
    for j in range(head_dim):
        bit_offset = j * mse_bits
        byte_idx = bit_offset // 8
        bit_shift = bit_offset % 8
        val = int(indices[j].item()) & mask
        packed[byte_idx] = int(packed[byte_idx].item()) | ((val << bit_shift) & 0xFF)
        spill = bit_shift + mse_bits - 8
        if spill > 0 and byte_idx + 1 < mse_bytes:
            packed[byte_idx + 1] = int(packed[byte_idx + 1].item()) | (val >> (mse_bits - spill))

    # Pack QJL signs
    for b in range(qjl_bytes):
        val = 0
        for k in range(8):
            j = b * 8 + k
            if j >= head_dim:
                break
            if signs[j] > 0:
                val |= (1 << k)
        packed[mse_bytes + b] = val

    # Pack norms as float16
    no = mse_bytes + qjl_bytes
    vn_bytes = struct.pack('e', float(vec_norm))
    rn_bytes = struct.pack('e', float(res_norm))
    packed[no] = vn_bytes[0]
    packed[no + 1] = vn_bytes[1]
    packed[no + 2] = rn_bytes[0]
    packed[no + 3] = rn_bytes[1]

    return packed


def unpack_vector(
    packed: torch.Tensor,
    head_dim: int,
    mse_bits: int,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Unpack uint8 to MSE values + signs + norms.

    Returns:
        mse_values: (d,) centroid values
        signs: (d,) float (+1/-1)
        vec_norm: scalar
        res_norm: scalar
    """
    import math
    mse_bytes = math.ceil(head_dim * mse_bits / 8)
    qjl_bytes = math.ceil(head_dim / 8)
    mask = (1 << mse_bits) - 1

    device = packed.device

    # Unpack MSE indices from bitstream (handles any mse_bits)
    indices = torch.zeros(head_dim, dtype=torch.long, device=device)
    for j in range(head_dim):
        bit_offset = j * mse_bits
        byte_idx = bit_offset // 8
        bit_shift = bit_offset % 8
        val = packed[byte_idx].item() >> bit_shift
        spill = bit_shift + mse_bits - 8
        if spill > 0 and byte_idx + 1 < mse_bytes:
            val |= packed[byte_idx + 1].item() << (mse_bits - spill)
        indices[j] = val & mask

    mse_values = centroids.to(device)[indices]

    # Unpack signs
    signs = torch.zeros(head_dim, dtype=torch.float32, device=device)
    for b in range(qjl_bytes):
        bv = packed[mse_bytes + b].item()
        for k in range(8):
            j = b * 8 + k
            if j >= head_dim:
                break
            signs[j] = 1.0 if ((bv >> k) & 1) else -1.0

    # Unpack norms
    no = mse_bytes + qjl_bytes
    vec_norm = struct.unpack('e', bytes([packed[no].item(), packed[no + 1].item()]))[0]
    res_norm = struct.unpack('e', bytes([packed[no + 2].item(), packed[no + 3].item()]))[0]

    return mse_values, signs, vec_norm, res_norm


def pack_vectors_batched(
    indices: torch.Tensor,
    signs: torch.Tensor,
    vec_norms: torch.Tensor,
    res_norms: torch.Tensor,
    head_dim: int,
    mse_bits: int,
) -> torch.Tensor:
    """Vectorized batch pack — no Python loops on batch dim.

    Args:
        indices: (batch, d) long
        signs: (batch, d) float
        vec_norms: (batch,) float
        res_norms: (batch,) float

    Returns:
        (batch, packed_size) uint8
    """
    import math
    batch = indices.shape[0]
    mse_bytes = math.ceil(head_dim * mse_bits / 8)
    qjl_bytes = math.ceil(head_dim / 8)
    packed_size = mse_bytes + qjl_bytes + 4
    device = indices.device

    packed = torch.zeros(batch, packed_size, dtype=torch.uint8, device=device)
    mask = (1 << mse_bits) - 1

    # Pack MSE indices as bitstream (handles any mse_bits, including 3)
    for j in range(head_dim):
        bit_offset = j * mse_bits
        byte_idx = bit_offset // 8
        bit_shift = bit_offset % 8
        val = (indices[:, j].int() & mask)
        # Low part fits in current byte
        packed[:, byte_idx] = packed[:, byte_idx] | (val << bit_shift).byte()
        # High part may spill into next byte
        spill = bit_shift + mse_bits - 8
        if spill > 0 and byte_idx + 1 < mse_bytes:
            packed[:, byte_idx + 1] = packed[:, byte_idx + 1] | (val >> (mse_bits - spill)).byte()

    # Pack QJL signs (vectorized over batch)
    for b in range(qjl_bytes):
        val = torch.zeros(batch, dtype=torch.uint8, device=device)
        for k in range(8):
            j = b * 8 + k
            if j >= head_dim:
                break
            val = val | ((signs[:, j] > 0).byte() << k)
        packed[:, mse_bytes + b] = val

    # Pack norms
    no = mse_bytes + qjl_bytes
    vn_u8 = vec_norms.half().view(torch.uint8).reshape(batch, 2)
    rn_u8 = res_norms.half().view(torch.uint8).reshape(batch, 2)
    packed[:, no:no + 2] = vn_u8
    packed[:, no + 2:no + 4] = rn_u8

    return packed
