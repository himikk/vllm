# SPDX-License-Identifier: Apache-2.0
"""MultiQuant quantizer registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.multiquant.base import KVQuantizer, KVQuantizerConfig

# Lazy registry — imports happen on first access to avoid circular deps
_REGISTRY: dict[
    str, tuple[str, str, str, str]  # (config_module, config_cls, quant_module, quant_cls)
] = {
    "tq3": (
        "vllm.multiquant.turboquant.config", "TurboQuantConfig",
        "vllm.multiquant.turboquant.quantizer", "TurboQuantizer",
    ),
    "tq4": (
        "vllm.multiquant.turboquant.config", "TurboQuantConfig",
        "vllm.multiquant.turboquant.quantizer", "TurboQuantizer",
    ),
    "tq2w": (
        "vllm.multiquant.turboquant.wht_config", "TurboQuantWHTConfig",
        "vllm.multiquant.turboquant.wht_quantizer", "TurboQuantWHTQuantizer",
    ),
    "tq3w": (
        "vllm.multiquant.turboquant.wht_config", "TurboQuantWHTConfig",
        "vllm.multiquant.turboquant.wht_quantizer", "TurboQuantWHTQuantizer",
    ),
    "tq4w": (
        "vllm.multiquant.turboquant.wht_config", "TurboQuantWHTConfig",
        "vllm.multiquant.turboquant.wht_quantizer", "TurboQuantWHTQuantizer",
    ),
    "tq3r": (
        "vllm.multiquant.turboquant.wht_config", "TurboQuantWHTConfig",
        "vllm.multiquant.turboquant.wht_quantizer", "TurboQuantWHTQuantizer",
    ),
    "tq4r": (
        "vllm.multiquant.turboquant.wht_config", "TurboQuantWHTConfig",
        "vllm.multiquant.turboquant.wht_quantizer", "TurboQuantWHTQuantizer",
    ),
    "rq2": (
        "vllm.multiquant.rotorquant.config", "RotorQuantConfig",
        "vllm.multiquant.rotorquant.quantizer", "RotorQuantizer",
    ),
    "rq3": (
        "vllm.multiquant.rotorquant.config", "RotorQuantConfig",
        "vllm.multiquant.rotorquant.quantizer", "RotorQuantizer",
    ),
    "rq4": (
        "vllm.multiquant.rotorquant.config", "RotorQuantConfig",
        "vllm.multiquant.rotorquant.quantizer", "RotorQuantizer",
    ),
}


def register_kv_quantizer(
    dtype_prefix: str,
    config_module: str,
    config_cls: str,
    quantizer_module: str,
    quantizer_cls: str,
    bits_range: tuple[int, ...] = (3, 4),
) -> None:
    """Register a new KV-cache quantizer for one or more bit widths."""
    for bits in bits_range:
        key = f"{dtype_prefix}{bits}"
        _REGISTRY[key] = (config_module, config_cls, quantizer_module, quantizer_cls)


def get_kv_quantizer_config(
    cache_dtype: str, head_dim: int
) -> "KVQuantizerConfig":
    """Get quantizer config for a cache dtype string.

    Supports parameterized strings like 'tq3w:bs64'.
    """
    base = cache_dtype.split(':')[0] if ':' in cache_dtype else cache_dtype
    if base not in _REGISTRY:
        raise ValueError(
            f"Unknown multiquant cache dtype: {cache_dtype}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    config_mod, config_cls_name, _, _ = _REGISTRY[base]
    import importlib
    mod = importlib.import_module(config_mod)
    config_cls = getattr(mod, config_cls_name)
    return config_cls.from_cache_dtype(cache_dtype, head_dim)


def get_kv_quantizer(cache_dtype: str) -> "KVQuantizer":
    """Get quantizer instance for a cache dtype string."""
    if cache_dtype not in _REGISTRY:
        raise ValueError(
            f"Unknown multiquant cache dtype: {cache_dtype}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    _, _, quant_mod, quant_cls_name = _REGISTRY[cache_dtype]
    import importlib
    mod = importlib.import_module(quant_mod)
    quant_cls = getattr(mod, quant_cls_name)
    return quant_cls()


def is_multiquant_dtype(cache_dtype: str) -> bool:
    """Check if a cache dtype is handled by MultiQuant.

    Supports parameterized strings like 'tq3w:bs64'.
    """
    base = cache_dtype.split(':')[0] if ':' in cache_dtype else cache_dtype
    return base in _REGISTRY


def get_registered_dtypes() -> list[str]:
    """List all registered MultiQuant cache dtypes."""
    return list(_REGISTRY.keys())
