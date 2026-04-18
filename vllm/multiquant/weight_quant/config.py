# SPDX-License-Identifier: Apache-2.0
"""Archer weight quantization configuration."""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = init_logger(__name__)


class ArcherConfig(QuantizationConfig):
    """Configuration for Archer (MultiQuant online weight quantization).

    Loads BF16/FP8 checkpoints and quantizes weights at load time using
    RotorQuant or TurboQuant compression. Decompresses on-the-fly during
    inference.

    IMPORTANT: Only accepts BF16, FP16, or FP8 source weights.
    Pre-quantized models (GPTQ, AWQ, INT4, INT8) are NOT supported —
    double quantization would introduce unacceptable error.
    """

    # Source dtypes that Archer can quantize from
    SUPPORTED_SOURCE_DTYPES = {"bfloat16", "float16", "float32", "fp8"}

    VALID_METHODS = {"tq", "rq"}

    def __init__(
        self,
        bits: int = 3,
        method: str = "tq",
        act_dtype: str = "bf16",
        seed: int = 42,
    ):
        import os
        env_method = os.environ.get("ARCHER_METHOD")
        env_bits = os.environ.get("ARCHER_BITS")
        if env_method:
            method = env_method
        if env_bits:
            bits = int(env_bits)
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Archer method must be one of {self.VALID_METHODS}, "
                f"got '{method}'"
            )
        self.bits = bits
        self.method = method
        self.act_dtype = act_dtype
        self.seed = seed

    def __repr__(self) -> str:
        return (
            f"ArcherConfig(bits={self.bits}, method={self.method}, "
            f"act_dtype={self.act_dtype})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "multiquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # SM70+

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # No config file needed — online quantization

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ArcherConfig:
        bits = config.get("bits", 3)
        method = config.get("method", "tq")
        act_dtype = config.get("act_dtype", "bf16")
        return cls(bits=bits, method=method, act_dtype=act_dtype)

    @classmethod
    def verify_source_dtype(cls, model_config) -> None:
        """Reject pre-quantized models — Archer only accepts BF16/FP16/FP8."""
        if model_config is None:
            return
        existing_quant = getattr(model_config, 'quantization', None)
        if existing_quant and existing_quant not in ('multiquant', 'fp8'):
            raise ValueError(
                f"Archer (multiquant) cannot quantize a pre-quantized model "
                f"(existing quantization: {existing_quant}). "
                f"Only BF16, FP16, or FP8 source weights are supported. "
                f"Double quantization would introduce unacceptable error."
            )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            from vllm.multiquant.weight_quant.online_linear import (
                ArcherOnlineLinearMethod,
            )
            return ArcherOnlineLinearMethod(self)
        # TODO: MoE support (ArcherOnlineMoEMethod)
        return None
