# SPDX-License-Identifier: Apache-2.0
"""RTN packing to GPTQ-compatible format.

Converts BF16/FP16 weight matrices to packed int32 GPTQ format via
Round-to-Nearest symmetric quantization.  Output is directly consumable
by MQSub4LinearMethod (INT2/INT3) or Marlin (INT4).

    qweight: [K_packed, N] int32   — packed along K (input) dimension
    scales:  [n_groups, N] float16 — per-group scale factors
    qzeros:  [n_groups, N_zp] int32 — packed zero points (symmetric: midpoint)
"""

from __future__ import annotations

import torch


def rtn_pack_gptq(
    W: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack BF16 weight to GPTQ int32 format via symmetric RTN.

    Args:
        W: Weight matrix [N, K] float (PyTorch Linear: [out, in]).
        bits: Quantization bits (2, 3, or 4).
        group_size: Group size for per-group scaling.

    Returns:
        qweight: [K_packed, N] int32 (GPTQ convention: K packed in rows)
        scales:  [n_groups, N] float16
        qzeros:  [n_groups, N_zp] int32

    Packing formats:
        INT2: 16 values per int32, LSB-first along K
        INT3: bitstream, 32 bits per int32 word
        INT4: 8 values per int32, LSB-first along K (GPTQ standard)
    """
    assert bits in (2, 3, 4), f"bits must be 2, 3, or 4, got {bits}"
    N, K = W.shape
    device = W.device
    W = W.float()

    if group_size <= 0 or group_size > K:
        group_size = K
    n_groups = (K + group_size - 1) // group_size
    n_levels = 1 << bits
    zp = n_levels // 2  # symmetric zero point (midpoint)

    # --- Per-group symmetric quantization ---
    scales = torch.zeros(n_groups, N, dtype=torch.float32, device=device)
    W_int = torch.zeros(N, K, dtype=torch.int32, device=device)

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, K)
        group = W[:, start:end]
        max_val = group.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_val / (zp - 1)
        scales[g] = scale.squeeze(-1)
        q = torch.clamp(
            torch.round(group / scale), -zp, zp - 1,
        ).to(torch.int32)
        W_int[:, start:end] = q

    # Shift to unsigned: [-zp, zp-1] → [0, n_levels-1]
    W_unsigned = (W_int + zp).to(torch.int32)  # [N, K]

    # --- Pack into GPTQ format: [K_packed, N] int32 ---
    # Keep scales in float16 for now — Marlin and mq_gemm both expect fp16.
    # For bfloat16 models: Marlin handles the conversion internally,
    # but scales MUST be float16 for the permutation to work correctly.
    scales_out = scales.to(torch.float16)

    if bits in (2, 4):
        qweight, qzeros = _pack_uniform(W_unsigned, scales_out,
                                         bits, n_groups, group_size,
                                         N, K, zp, device)
    else:
        qweight, qzeros = _pack_int3(W_unsigned, scales_out,
                                      n_groups, group_size,
                                      N, K, zp, device)

    return qweight, scales_out, qzeros


def _pack_uniform(
    W_unsigned: torch.Tensor, scales: torch.Tensor,
    bits: int, n_groups: int, group_size: int,
    N: int, K: int, zp: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack INT2 or INT4 with uniform bit-width (power-of-2 values per int32).

    INT2: 16 values per int32
    INT4: 8 values per int32
    """
    pack_factor = 32 // bits  # 16 for INT2, 8 for INT4
    K_packed = (K + pack_factor - 1) // pack_factor
    mask = (1 << bits) - 1

    # W_unsigned is [N, K], GPTQ stores [K_packed, N]
    # Pack along K: every pack_factor consecutive K values into one int32
    qweight = torch.zeros(K_packed, N, dtype=torch.int32, device=device)
    for i in range(pack_factor):
        k_indices = torch.arange(i, K, pack_factor, device=device)
        if len(k_indices) > K_packed:
            k_indices = k_indices[:K_packed]
        # W_unsigned[:, k_indices] shape [N, len(k_indices)]
        # qweight[:len(k_indices), :] accumulates
        qweight[:len(k_indices)] |= (
            (W_unsigned[:, k_indices].T & mask) << (i * bits)
        )

    # Qzeros: pack zero-point value for each group
    # For symmetric quant, all zeros are the same (zp = midpoint)
    N_zp = (N + pack_factor - 1) // pack_factor
    zp_word = 0
    for i in range(pack_factor):
        zp_word |= (zp & mask) << (i * bits)
    # Handle unsigned overflow for int32 (e.g. 0xAAAAAAAA for INT2 zp=2)
    if zp_word > 0x7FFFFFFF:
        zp_word = zp_word - (1 << 32)
    qzeros = torch.full((n_groups, N_zp), zp_word,
                         dtype=torch.int32, device=device)

    return qweight, qzeros


def _pack_int3(
    W_unsigned: torch.Tensor, scales: torch.Tensor,
    n_groups: int, group_size: int,
    N: int, K: int, zp: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack INT3 as bitstream: 3 consecutive bits per value, 32 bits per int32.

    Matches mq_gemm_int3.cu expected layout.
    """
    total_bits = K * 3
    K_packed = (total_bits + 31) // 32

    # Vectorized: build all 3 bit-planes, then pack 32 bits at a time
    # W_unsigned is [N, K] with values 0..7
    qweight = torch.zeros(K_packed, N, dtype=torch.int32, device=device)

    # For each bit position (0,1,2), extract bit from all K values
    for bit_idx in range(3):
        # bit_plane[n, k] = (W_unsigned[n, k] >> bit_idx) & 1
        bit_plane = ((W_unsigned >> bit_idx) & 1).int()  # [N, K]
        # Place each bit at position k*3+bit_idx in the bitstream
        for k in range(K):
            global_bit = k * 3 + bit_idx
            word = global_bit // 32
            pos = global_bit % 32
            if word < K_packed:
                qweight[word] |= bit_plane[:, k] << pos

    # Qzeros: constant zp packed as bitstream
    N_zp_bits = N * 3
    N_zp = (N_zp_bits + 31) // 32
    zp_word = torch.zeros(N_zp, dtype=torch.int32, device=device)
    for n_idx in range(N):
        for bit_idx in range(3):
            global_bit = n_idx * 3 + bit_idx
            word = global_bit // 32
            pos = global_bit % 32
            if word < N_zp:
                zp_word[word] |= ((zp >> bit_idx) & 1) << pos
    qzeros = zp_word.unsqueeze(0).expand(n_groups, -1).contiguous()

    return qweight, qzeros
