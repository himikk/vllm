# SPDX-License-Identifier: Apache-2.0
"""Storage format abstraction for MultiQuant.

Decouples quantization (how weights are compressed) from compute
(how compressed weights are used in GEMM).

Multiple quantizers can produce the same storage format:
  RTN, GPTQ, AutoRound, AWQ  →  INT4_PACKED  →  Marlin kernel
  Archer TQ, Archer RQ       →  MQ_PACKED    →  Archer kernel
  FP8 Online, ModelOpt FP8   →  FP8          →  FP8 GEMM

And one storage format can be consumed by multiple kernels:
  INT4_PACKED  →  Marlin / Triton / naive unpack+F.linear
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch


class StorageFormat(Enum):
    """Physical format of quantized weights in VRAM."""

    # Standard formats
    BF16 = auto()        # torch.bfloat16, no compression
    FP16 = auto()        # torch.float16, no compression
    FP8 = auto()         # torch.float8_e4m3fn, per-tensor/per-channel scale

    # Integer packed formats (multiple values per int32)
    INT4_PACKED = auto()  # 8 × 4-bit values per int32, per-group scales
    INT8_PACKED = auto()  # 4 × 8-bit values per int32, per-group scales
    INT2_PACKED = auto()  # 16 × 2-bit values per int32, per-group scales

    # MultiQuant formats (rotation + centroids + QJL)
    MQ_PACKED = auto()    # MSE indices + QJL signs + norms (uint8)

    # GGUF formats
    GGUF_Q4_0 = auto()   # llama.cpp Q4_0
    GGUF_Q4_K = auto()   # llama.cpp Q4_K_M


class ComputeKernel(Enum):
    """Available GEMM kernels for decompressing+computing."""

    MARLIN = auto()       # INT4/INT8 → BF16/FP8, fused decompress+GEMM
    TRITON_DEQUANT = auto()  # Generic Triton dequant+GEMM
    TORCH_LINEAR = auto()    # Naive: unpack to BF16 → F.linear
    FP8_GEMM = auto()     # Native FP8 GEMM (cutlass/cublas)
    ARCHER = auto()       # MultiQuant fused decompress+GEMM (TODO)


@dataclass
class StorageSpec:
    """Describes how a weight tensor is stored and how to compute with it.

    This is the contract between quantizer (producer) and kernel (consumer).
    """
    format: StorageFormat
    bits: int                           # Bits per element (2, 4, 8, 16)
    group_size: int = -1                # -1 = per-channel, else per-group
    has_zeros: bool = False             # Asymmetric quantization
    has_scales: bool = True             # Per-group/per-channel scales
    symmetric: bool = True              # Symmetric quantization

    # Compatible kernels (ordered by preference)
    compatible_kernels: list[ComputeKernel] = field(default_factory=list)

    def best_kernel(self) -> ComputeKernel:
        """Return the best available kernel for this format."""
        if self.compatible_kernels:
            return self.compatible_kernels[0]
        return ComputeKernel.TORCH_LINEAR


# Pre-defined storage specs for common configurations
INT4_SYMMETRIC = StorageSpec(
    format=StorageFormat.INT4_PACKED,
    bits=4,
    group_size=128,
    symmetric=True,
    compatible_kernels=[
        ComputeKernel.MARLIN,
        ComputeKernel.TRITON_DEQUANT,
        ComputeKernel.TORCH_LINEAR,
    ],
)

INT4_ASYMMETRIC = StorageSpec(
    format=StorageFormat.INT4_PACKED,
    bits=4,
    group_size=128,
    symmetric=False,
    has_zeros=True,
    compatible_kernels=[
        ComputeKernel.MARLIN,
        ComputeKernel.TORCH_LINEAR,
    ],
)

MQ3_PACKED = StorageSpec(
    format=StorageFormat.MQ_PACKED,
    bits=3,
    group_size=-1,  # per-row (full vector)
    symmetric=True,
    compatible_kernels=[
        ComputeKernel.ARCHER,
        ComputeKernel.TORCH_LINEAR,
    ],
)

MQ4_PACKED = StorageSpec(
    format=StorageFormat.MQ_PACKED,
    bits=4,
    group_size=-1,
    symmetric=True,
    compatible_kernels=[
        ComputeKernel.ARCHER,
        ComputeKernel.TORCH_LINEAR,
    ],
)

FP8_DYNAMIC = StorageSpec(
    format=StorageFormat.FP8,
    bits=8,
    group_size=-1,
    symmetric=True,
    compatible_kernels=[
        ComputeKernel.FP8_GEMM,
        ComputeKernel.TORCH_LINEAR,
    ],
)


# Registry: which quantizers produce which format
QUANTIZER_TO_FORMAT: dict[str, StorageSpec] = {
    # Online quantizers (calibration-free)
    "multiquant_rq3": MQ3_PACKED,
    "multiquant_rq4": MQ4_PACKED,
    "multiquant_tq3": MQ3_PACKED,
    "multiquant_tq4": MQ4_PACKED,
    "autoround_rtn": INT4_SYMMETRIC,
    "fp8_online": FP8_DYNAMIC,

    # Offline quantizers (pre-quantized checkpoints)
    "gptq": INT4_SYMMETRIC,
    "awq": INT4_ASYMMETRIC,
    "auto-round": INT4_SYMMETRIC,  # calibrated AutoRound
    "marlin_mq": INT4_SYMMETRIC,
}


def get_storage_spec(quantizer_name: str) -> StorageSpec:
    """Get storage format for a quantizer."""
    if quantizer_name in QUANTIZER_TO_FORMAT:
        return QUANTIZER_TO_FORMAT[quantizer_name]
    raise ValueError(
        f"Unknown quantizer: {quantizer_name}. "
        f"Available: {list(QUANTIZER_TO_FORMAT.keys())}"
    )


def get_compatible_kernels(storage_format: StorageFormat) -> list[ComputeKernel]:
    """Get all kernels that can consume a storage format."""
    result = set()
    for spec in QUANTIZER_TO_FORMAT.values():
        if spec.format == storage_format:
            result.update(spec.compatible_kernels)
    return sorted(result, key=lambda k: k.value)
