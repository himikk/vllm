# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the XFP fused GEMM CUDA kernel.

Compiles kernels/multiquant/xfp_gemm*.cu once per process via
torch.utils.cpp_extension.load. One .so bedient alle Bitbreiten —
der Dispatch auf BITS passiert im C++ wrapper.

v12 (default) uses a static SMEM A-row cache with a compile-time
K_SMEM_MAX = 8192 (Linear). Shapes with K > 8192 (e.g. attn kv_b 17408)
fall back to v11 via ``dispatch_linear_gemm`` at call time.

Path resolution mirrors _load_mq_gemm in
vllm/multiquant/weight_quant/mq_sub4_linear.py: try the source tree
first (for development), fall back to /opt/mq_kernels (container).
"""

from __future__ import annotations

import os
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

# Keep in sync with LinearPolicy::K_SMEM_MAX in xfp_gemm_core.cuh.
K_SMEM_MAX_LINEAR = 8192


def _resolve_kernel_dir() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.normpath(
        os.path.join(here, "..", "..", "..", "kernels", "multiquant")
    )
    force = os.environ.get("XFP_KERNEL", "")
    if force == "v8":
        preferred = ("xfp_gemm_v8.cu", "xfp_gemm.cu")
    elif force == "v9":
        preferred = ("xfp_gemm_v9.cu",)
    elif force == "v10":
        preferred = ("xfp_gemm_v10.cu",)
    elif force == "v11":
        preferred = ("xfp_gemm_v11.cu",)
    else:
        # v12 default: static SMEM A-row cache. Needs v11 as fallback for
        # K > 8192 shapes, so we try v12 first and also keep v11 around.
        preferred = ("xfp_gemm_v12.cu", "xfp_gemm_v11.cu",
                     "xfp_gemm_v10.cu", "xfp_gemm_v8.cu", "xfp_gemm.cu")
    for name in preferred:
        if os.path.exists(os.path.join(src_dir, name)):
            return src_dir
    fallback = "/opt/mq_kernels"
    for name in preferred:
        if os.path.exists(os.path.join(fallback, name)):
            return fallback
    return None


_KERNEL_SRC_DIR: Optional[str] = _resolve_kernel_dir()

# Primary kernel (v12 by default). Used for all calls unless K exceeds
# K_SMEM_MAX_LINEAR, in which case dispatch_linear_gemm falls back to v11.
_xfp_gemm_kernel = None
# Optional v11 fallback module, populated only when the primary is v12.
_xfp_gemm_v11_fallback = None
_load_attempted = False


def _find_kernel_cu(primary_only: bool = False) -> Optional[str]:
    if _KERNEL_SRC_DIR is None:
        return None
    force = os.environ.get("XFP_KERNEL", "")
    if force == "v8":
        names = ("xfp_gemm_v8.cu", "xfp_gemm.cu")
    elif force == "v9":
        names = ("xfp_gemm_v9.cu",)
    elif force == "v10":
        names = ("xfp_gemm_v10.cu",)
    elif force == "v11":
        names = ("xfp_gemm_v11.cu",)
    else:
        names = ("xfp_gemm_v12.cu", "xfp_gemm_v11.cu",
                 "xfp_gemm_v10.cu", "xfp_gemm_v8.cu", "xfp_gemm.cu")
    for name in names:
        p = os.path.join(_KERNEL_SRC_DIR, name)
        if os.path.exists(p):
            return p
    return None


_KERNEL_CU_PATH: Optional[str] = _find_kernel_cu()


def _kname_from_path(path: str) -> str:
    for tag in ("v12", "v11", "v10", "v9", "v8"):
        if tag in path:
            return f"xfp_gemm_{tag}"
    return "xfp_gemm"


def _compile_one(cu_path: str):
    from torch.utils.cpp_extension import load
    return load(
        name=_kname_from_path(cu_path),
        sources=[cu_path],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "-gencode=arch=compute_120,code=sm_120",
            "-gencode=arch=compute_121,code=sm_121",
            "-diag-suppress=177,3288",
        ],
        verbose=False,
    )


def _load_xfp_gemm(bits: int):
    """JIT compile (first call) and return the xfp_gemm kernel module.

    When the primary kernel is v12, additionally JIT-compiles v11 so
    dispatch_linear_gemm can use it as a K>8192 fallback.

    IMPORTANT: this function must not use any os.path calls after the
    cached-kernel fast path. torch.compile/Dynamo graphs through this
    on the forward pass and cannot trace posix path normalization.
    """
    global _xfp_gemm_kernel, _xfp_gemm_v11_fallback, _load_attempted

    if _xfp_gemm_kernel is not None:
        return _xfp_gemm_kernel

    if bits not in (2, 3, 4):
        raise ValueError(
            f"XFP kernel: unsupported bits={bits}, must be in {{2,3,4}}"
        )

    if _load_attempted:
        return None
    _load_attempted = True

    if _KERNEL_CU_PATH is None:
        logger.warning(
            "XFP kernel: xfp_gemm*.cu not found in source tree "
            "or /opt/mq_kernels; kernel unavailable"
        )
        return None

    try:
        _xfp_gemm_kernel = _compile_one(_KERNEL_CU_PATH)
        primary_tag = _kname_from_path(_KERNEL_CU_PATH)
        logger.info("XFP GEMM kernel compiled (%s) from %s",
                    primary_tag, _KERNEL_SRC_DIR)

        # If primary is v12, compile v11 as K>8192 fallback.
        if "v12" in _KERNEL_CU_PATH and _KERNEL_SRC_DIR is not None:
            v11_path = os.path.join(_KERNEL_SRC_DIR, "xfp_gemm_v11.cu")
            if os.path.exists(v11_path):
                try:
                    _xfp_gemm_v11_fallback = _compile_one(v11_path)
                    logger.info(
                        "XFP Linear: v12 primary (static SMEM A-row cache, "
                        "K_SMEM_MAX=%d) + v11 fallback armed for K>%d",
                        K_SMEM_MAX_LINEAR, K_SMEM_MAX_LINEAR,
                    )
                except Exception as e:
                    logger.warning(
                        "XFP Linear v11 fallback compile FAILED: %s — "
                        "layers with K>%d will RAISE at runtime, not fall back",
                        e, K_SMEM_MAX_LINEAR,
                    )
            else:
                logger.warning(
                    "XFP Linear v11 fallback source missing (%s) — "
                    "layers with K>%d will RAISE at runtime, not fall back",
                    v11_path, K_SMEM_MAX_LINEAR,
                )
        else:
            # User forced a non-v12 primary (e.g., XFP_KERNEL=v11). No SMEM
            # A-row cache is active. Surface this loudly so a missing speedup
            # is traceable to the env override.
            force = os.environ.get("XFP_KERNEL", "")
            logger.warning(
                "XFP Linear: primary=%s (XFP_KERNEL=%r). v12 SMEM A-row "
                "cache is NOT active — per-warp global A-reads in use.",
                primary_tag, force or "<auto, v12 not found>",
            )

        return _xfp_gemm_kernel
    except Exception as e:
        logger.warning("XFP kernel JIT compile FAILED: %s", e)
        return None


# Track (K, N) pairs that already hit the v11 fallback path, so we log
# each distinct shape exactly once rather than every forward pass.
_FALLBACK_WARNED_SHAPES: set = set()


def dispatch_linear_gemm(x, packed, cb, C, bits: int, K: int) -> None:
    """Call the right Linear xfp_gemm for the given K.

    K <= K_SMEM_MAX_LINEAR  → primary kernel (v12 static SMEM cache)
    K  > K_SMEM_MAX_LINEAR  → v11 fallback (direct global A-row reads)

    Both kernels have identical numerical output; the only difference is
    the A-source path. Dispatch adds one Python-level branch per call
    (negligible vs the kernel time).

    Every distinct K that triggers the v11 fallback is logged once at
    WARN level so a disappointing tok/s can be traced back to a layer
    bypassing the SMEM A-row cache.
    """
    if _xfp_gemm_kernel is None:
        raise RuntimeError(
            "XFP kernel not loaded; call _load_xfp_gemm(bits) first."
        )

    use_fallback = (
        _xfp_gemm_v11_fallback is not None
        and K > K_SMEM_MAX_LINEAR
    )

    if use_fallback:
        shape_key = (int(K), int(C.shape[-1]), int(bits))
        if shape_key not in _FALLBACK_WARNED_SHAPES:
            _FALLBACK_WARNED_SHAPES.add(shape_key)
            logger.warning(
                "XFP Linear fallback → v11 for K=%d N=%d bits=%d "
                "(> K_SMEM_MAX=%d, no SMEM A-row cache for this layer)",
                K, C.shape[-1], bits, K_SMEM_MAX_LINEAR,
            )
        _xfp_gemm_v11_fallback.xfp_gemm(x, packed, cb, C, int(bits), int(K))
    elif K > K_SMEM_MAX_LINEAR and _xfp_gemm_v11_fallback is None:
        # Primary must be v12 (since K>8192 would've been fine for v11) but
        # fallback failed to load → runtime error, don't silently miscompute.
        raise RuntimeError(
            f"XFP Linear: K={K} > K_SMEM_MAX={K_SMEM_MAX_LINEAR} and no "
            f"v11 fallback is loaded. Set XFP_KERNEL=v11 to use v11 as "
            f"primary, or rebuild image with v11 source present."
        )
    else:
        _xfp_gemm_kernel.xfp_gemm(x, packed, cb, C, int(bits), int(K))
