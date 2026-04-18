# SPDX-License-Identifier: Apache-2.0
"""MultiQuant sub-4-bit LinearMethod — wraps GPTQLinearMethod with
our own fused GEMM kernels (mq_gemm_int2/mq_gemm_int3).

Inherits create_weights from GPTQLinearMethod (correct shard-fusing,
PackedvLLMParameter, Fraction pack_factor). Overrides apply() to call
our verified CUDA kernels instead of vLLM's broken gptq_gemm.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.gptq import (
    GPTQConfig,
    GPTQLinearMethod,
)

logger = init_logger(__name__)

_mq_gemm_int2 = None
_mq_gemm_int3 = None


def _load_mq_gemm(bits: int):
    """JIT compile the fused GEMM kernel."""
    global _mq_gemm_int2, _mq_gemm_int3

    if bits == 2 and _mq_gemm_int2 is not None:
        return _mq_gemm_int2
    if bits == 3 and _mq_gemm_int3 is not None:
        return _mq_gemm_int3

    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "kernels", "multiquant"
    )
    if not os.path.exists(src_dir):
        # Container path
        src_dir = "/opt/mq_kernels"

    from torch.utils.cpp_extension import load

    if bits == 2:
        src = os.path.join(src_dir, "mq_gemm_int2.cu")
        if not os.path.exists(src):
            logger.warning("mq_gemm_int2.cu not found at %s", src)
            return None
        _mq_gemm_int2 = load(
            name="mq_gemm_int2", sources=[src],
            extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288"],
            verbose=False)
        logger.info("mq_gemm_int2 compiled and loaded")
        return _mq_gemm_int2

    elif bits == 3:
        src = os.path.join(src_dir, "mq_gemm_int3.cu")
        if not os.path.exists(src):
            logger.warning("mq_gemm_int3.cu not found at %s", src)
            return None
        _mq_gemm_int3 = load(
            name="mq_gemm_int3", sources=[src],
            extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288"],
            verbose=False)
        logger.info("mq_gemm_int3 compiled and loaded")
        return _mq_gemm_int3

    return None


class MQSub4LinearMethod(GPTQLinearMethod):
    """Sub-4-bit Linear method using MultiQuant fused GEMM kernels.

    Inherits create_weights() from GPTQLinearMethod (shard-fusing works).
    Overrides apply() to use mq_gemm_int2/mq_gemm_int3 instead of gptq_gemm.
    Overrides process_weights_after_loading() to skip ExLlama shuffle.
    """

    def __init__(self, quant_config: GPTQConfig):
        super().__init__(quant_config)
        self._bits = quant_config.weight_bits
        self._group_size = quant_config.group_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Convert to plain Parameters (for torch.compile)
        layer.qweight = torch.nn.Parameter(
            layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(
            layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(
            layer.scales.data, requires_grad=False)
        layer.g_idx = torch.nn.Parameter(
            layer.g_idx.data, requires_grad=False)
        # No ExLlama shuffle — our kernels read raw GPTQ format

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kernel = _load_mq_gemm(self._bits)
        if kernel is None:
            raise RuntimeError(
                f"mq_gemm_int{self._bits} kernel not available")

        out_shape = x.shape[:-1] + (layer.qweight.shape[-1],)
        reshaped_x = x.reshape(-1, x.shape[-1]).to(torch.float16)
        M = reshaped_x.shape[0]
        N = layer.qweight.shape[1]

        # Output must be zeroed (kernel uses atomicAdd for K-split)
        C = torch.zeros(M, N, dtype=torch.float16, device=x.device)

        if self._bits == 2:
            kernel.mq_gemm_int2(
                reshaped_x, layer.qweight, layer.scales,
                layer.qzeros, C, self._group_size)
        elif self._bits == 3:
            K = reshaped_x.shape[1]
            kernel.mq_gemm_int3(
                reshaped_x, layer.qweight, layer.scales,
                layer.qzeros, C, K, self._group_size)

        if bias is not None:
            C.add_(bias)
        return C.reshape(out_shape)
