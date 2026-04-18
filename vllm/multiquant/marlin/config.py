# SPDX-License-Identifier: Apache-2.0
"""Marlin adapter config for MultiQuant registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = init_logger(__name__)


class MarlinMultiQuantConfig(QuantizationConfig):
    """Adapter: exposes Marlin (GPTQ/AWQ INT4) through MultiQuant interface.

    This does NOT do online quantization — it delegates to the existing
    GPTQMarlinLinearMethod for pre-quantized checkpoints. Used for
    benchmarking Marlin alongside TQ/RQ in the same framework.

    Usage: --quantization marlin_mq (explicit MultiQuant path)
    Normal Marlin: --quantization gptq_marlin (unchanged, preferred)
    """

    def __init__(self, bits: int = 4, group_size: int = 128,
                 act_dtype: str = "bf16"):
        self.bits = bits
        self.group_size = group_size
        self.act_dtype = act_dtype

    @classmethod
    def get_name(cls) -> str:
        return "marlin_mq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MarlinMultiQuantConfig:
        return cls(
            bits=config.get("bits", 4),
            group_size=config.get("group_size", 128),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """Delegate to existing GPTQMarlinLinearMethod."""
        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            # Import and use the existing Marlin implementation
            try:
                from vllm.model_executor.layers.quantization.gptq_marlin import (
                    GPTQMarlinConfig,
                )
                gptq_config = GPTQMarlinConfig(
                    weight_bits=self.bits,
                    group_size=self.group_size,
                    desc_act=False,
                    is_sym=True,
                    quant_method="gptq",
                    lm_head_quantized=False,
                )
                return gptq_config.get_quant_method(layer, prefix)
            except Exception as e:
                logger.warning("Marlin adapter failed: %s", e)
                return None
        return None
