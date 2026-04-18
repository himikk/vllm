# SPDX-License-Identifier: Apache-2.0
"""XFP adapter for the generic MultiQuant weight cache.

The actual plumbing (key, manifest, safetensors I/O, stats) lives in
``vllm.multiquant.weight_cache``. This module only knows how to pack
an XFP-quantized layer's attributes into a tensor dict and reconstruct
them on load.

Other quant methods (Archer / TurboQuant / RotorQuant / AutoRound RTN)
follow the same pattern: a per-method adapter module next to the method's
online_linear/online_moe, using the generic cache.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.multiquant.weight_cache import MultiQuantWeightCache

logger = init_logger(__name__)

# ─── XFP Linear ───────────────────────────────────────────────────────

_XFP_LINEAR_METHOD = "xfp_linear"
_XFP_MOE_METHOD = "xfp_moe"


def save_linear(
    cache: MultiQuantWeightCache, layer_prefix: str, layer: nn.Module,
) -> bool:
    tensors: dict[str, torch.Tensor] = {
        "xfp_packed": layer.xfp_packed.data,
        "xfp_codebook": layer.xfp_codebook.data,
    }
    has_outliers = bool(getattr(layer, "_xfp_has_outliers", False))
    if has_outliers:
        tensors["xfp_outlier_row"] = layer.xfp_outlier_row.data
        tensors["xfp_outlier_col"] = layer.xfp_outlier_col.data
        tensors["xfp_outlier_val"] = layer.xfp_outlier_val.data
    metadata = {
        "bits": int(layer._xfp_bits),
        "K": int(layer._xfp_K),
        "N": int(layer._xfp_N),
        "has_outliers": 1 if has_outliers else 0,
    }
    return cache.save(layer_prefix, _XFP_LINEAR_METHOD, tensors, metadata)


def load_linear(
    cache: MultiQuantWeightCache, layer_prefix: str,
    layer: nn.Module, device: torch.device,
) -> bool:
    res = cache.load(layer_prefix, _XFP_LINEAR_METHOD, device)
    if res is None:
        return False
    tensors, meta = res
    try:
        layer.xfp_packed = nn.Parameter(
            tensors["xfp_packed"], requires_grad=False)
        layer.xfp_codebook = nn.Parameter(
            tensors["xfp_codebook"], requires_grad=False)
        has_outliers = meta.get("has_outliers", "0") == "1"
        if has_outliers:
            layer.xfp_outlier_row = nn.Parameter(
                tensors["xfp_outlier_row"], requires_grad=False)
            layer.xfp_outlier_col = nn.Parameter(
                tensors["xfp_outlier_col"], requires_grad=False)
            layer.xfp_outlier_val = nn.Parameter(
                tensors["xfp_outlier_val"], requires_grad=False)
        layer._xfp_has_outliers = has_outliers
        layer._xfp_bits = int(meta["bits"])
        layer._xfp_K = int(meta["K"])
        layer._xfp_N = int(meta["N"])
        layer._xfp_stats = None
        layer._xfp_packed_done = True
        return True
    except KeyError as e:
        logger.warning(
            "XFP cache: %s is incomplete (%s) — treating as miss",
            layer_prefix, e,
        )
        return False


# ─── XFP MoE ──────────────────────────────────────────────────────────

def save_moe(
    cache: MultiQuantWeightCache, layer_prefix: str, layer: nn.Module,
) -> bool:
    tensors: dict[str, torch.Tensor] = {
        "w13_xfp_packed":   layer.w13_xfp_packed.data,
        "w13_xfp_codebook": layer.w13_xfp_codebook.data,
        "w2_xfp_packed":    layer.w2_xfp_packed.data,
        "w2_xfp_codebook":  layer.w2_xfp_codebook.data,
    }
    metadata = {
        "bits":  int(layer._xfp_moe_bits),
        "K13":   int(layer._xfp_moe_K13),
        "N13":   int(layer._xfp_moe_N13),
        "K2":    int(layer._xfp_moe_K2),
        "N2":    int(layer._xfp_moe_N2),
        "E":     int(layer._xfp_moe_E),
        "fpe13": int(layer._xfp_moe_fpe13),
        "fpe2":  int(layer._xfp_moe_fpe2),
    }
    return cache.save(layer_prefix, _XFP_MOE_METHOD, tensors, metadata)


def load_moe(
    cache: MultiQuantWeightCache, layer_prefix: str,
    layer: nn.Module, device: torch.device,
) -> bool:
    res = cache.load(layer_prefix, _XFP_MOE_METHOD, device)
    if res is None:
        return False
    tensors, meta = res
    try:
        layer.w13_xfp_packed = nn.Parameter(
            tensors["w13_xfp_packed"], requires_grad=False)
        layer.w13_xfp_codebook = nn.Parameter(
            tensors["w13_xfp_codebook"], requires_grad=False)
        layer.w2_xfp_packed = nn.Parameter(
            tensors["w2_xfp_packed"], requires_grad=False)
        layer.w2_xfp_codebook = nn.Parameter(
            tensors["w2_xfp_codebook"], requires_grad=False)
        layer._xfp_moe_bits = int(meta["bits"])
        layer._xfp_moe_K13 = int(meta["K13"])
        layer._xfp_moe_N13 = int(meta["N13"])
        layer._xfp_moe_K2 = int(meta["K2"])
        layer._xfp_moe_N2 = int(meta["N2"])
        layer._xfp_moe_E = int(meta["E"])
        layer._xfp_moe_fpe13 = int(meta["fpe13"])
        layer._xfp_moe_fpe2 = int(meta["fpe2"])
        layer._xfp_moe_packed = True
        return True
    except (KeyError, ValueError) as e:
        logger.warning(
            "XFP MoE cache: %s is incomplete (%s) — treating as miss",
            layer_prefix, e,
        )
        return False
