# SPDX-License-Identifier: Apache-2.0
"""MultiQuant: Generic KV-cache quantization framework for vLLM.

Supports multiple quantization algorithms (TurboQuant, RotorQuant, etc.)
via a unified interface and registry.

Usage:
    from vllm.multiquant import get_kv_quantizer_config, is_multiquant_dtype

    if is_multiquant_dtype("tq3"):
        config = get_kv_quantizer_config("tq3", head_dim=128)
"""

from vllm.multiquant.registry import (
    get_kv_quantizer,
    get_kv_quantizer_config,
    get_registered_dtypes,
    is_multiquant_dtype,
    register_kv_quantizer,
)

__all__ = [
    "get_kv_quantizer",
    "get_kv_quantizer_config",
    "get_registered_dtypes",
    "is_multiquant_dtype",
    "register_kv_quantizer",
]
