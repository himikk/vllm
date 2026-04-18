# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the XFP fused MoE GEMM CUDA kernel."""

from __future__ import annotations

import os
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

_xfp_moe_kernel = None
_load_attempted = False


def _find_kernel_cu() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.normpath(
        os.path.join(here, "..", "..", "..", "kernels", "multiquant")
    )
    force = os.environ.get("XFP_MOE_KERNEL", "")
    if force == "v8":
        candidates = ("xfp_moe_gemm.cu",)
    elif force == "v10":
        candidates = ("xfp_moe_gemm_v10.cu", "xfp_moe_gemm.cu")
    elif force == "v11":
        candidates = ("xfp_moe_gemm_v11.cu",)
    else:
        # v12 default: static SMEM A-row cache (K_SMEM_MAX=4096 covers all
        # Qwen/GLM MoE shapes with K=2048, 2× headroom). v11 kept as
        # fallback if v12 not present in this build.
        candidates = ("xfp_moe_gemm_v12.cu", "xfp_moe_gemm_v11.cu",
                      "xfp_moe_gemm_v10.cu", "xfp_moe_gemm.cu")
    for d in [src_dir, "/opt/mq_kernels"]:
        for name in candidates:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


_KERNEL_CU_PATH: Optional[str] = _find_kernel_cu()


def _load_xfp_moe_gemm():
    global _xfp_moe_kernel, _load_attempted

    if _xfp_moe_kernel is not None:
        return _xfp_moe_kernel

    if _load_attempted:
        return None
    _load_attempted = True

    if _KERNEL_CU_PATH is None:
        logger.warning("XFP MoE kernel: xfp_moe_gemm.cu not found")
        return None

    try:
        from torch.utils.cpp_extension import load
        # Module name must match the file so torch caches the right .so
        if "v12" in _KERNEL_CU_PATH:
            mod_name = "xfp_moe_gemm_v12"
        elif "v11" in _KERNEL_CU_PATH:
            mod_name = "xfp_moe_gemm_v11"
        elif "v10" in _KERNEL_CU_PATH:
            mod_name = "xfp_moe_gemm_v10"
        else:
            mod_name = "xfp_moe_gemm"
        _xfp_moe_kernel = load(
            name=mod_name,
            sources=[_KERNEL_CU_PATH],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("XFP MoE GEMM kernel compiled (%s) from %s",
                     mod_name, os.path.dirname(_KERNEL_CU_PATH))
        # Be loud about whether the SMEM A-row cache is active for MoE —
        # a silently-selected v10/v11 would explain any missing speedup.
        force = os.environ.get("XFP_MOE_KERNEL", "")
        if mod_name == "xfp_moe_gemm_v12":
            logger.info(
                "XFP MoE: v12 primary (static SMEM A-row cache, K_SMEM_MAX=4096)"
            )
        else:
            logger.warning(
                "XFP MoE: primary=%s (XFP_MOE_KERNEL=%r). v12 SMEM A-row "
                "cache is NOT active — per-warp global A-reads in use.",
                mod_name, force or "<auto, v12 not found>",
            )
        return _xfp_moe_kernel
    except Exception as e:
        logger.warning("XFP MoE kernel JIT compile FAILED: %s", e)
        return None
