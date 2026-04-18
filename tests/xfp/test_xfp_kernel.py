# SPDX-License-Identifier: Apache-2.0
"""GPU unit tests for xfp_gemm CUDA kernel.

Runs only when CUDA is available. Compares the kernel output against a
pure-torch reference `x @ dequant(packed, codebook).T` for bits in {2, 3, 4}
on small shapes. Toleranz is cos similarity > 0.999 — this is a functional
correctness check, not a performance benchmark.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm.multiquant.xfp.xfp_pack import xfp_pack, dequant_xfp, xfp_repack

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xfp_gemm requires CUDA"
)


def _reference(x, packed_int32, codebook, K, bits):
    """Reference matmul: x @ dequant(packed, codebook).T via torch."""
    W_rec = dequant_xfp(packed_int32, codebook, K=K, bits=bits)
    return (x.to(torch.float32) @ W_rec.to(torch.float32).T).to(x.dtype)


@cuda_only
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("M", [1, 4, 16])
@pytest.mark.parametrize("N_out, K", [(64, 128), (128, 256), (256, 512)])
def test_xfp_gemm_matches_reference(bits: int, M: int, N_out: int, K: int) -> None:
    torch.manual_seed(bits * 1000 + M * 100 + N_out + K)
    device = "cuda"

    # Construct a weight matrix, pack it, repack, move to GPU
    W = torch.randn(N_out, K, dtype=torch.float32) * 0.1
    packed_cpu, codebook_cpu, _, _, _ = xfp_pack(W, bits=bits)
    repacked_cpu = xfp_repack(packed_cpu)
    repacked = repacked_cpu.to(device)
    packed_orig = packed_cpu.to(device)
    codebook = codebook_cpu.to(device)

    # Build an input activation
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    # Reference: fp32 matmul through the torch dequant path (uses ORIGINAL packed)
    expected = _reference(x, packed_orig, codebook, K=K, bits=bits)

    # Kernel (uses REPACKED)
    from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm
    kernel = _load_xfp_gemm(bits)
    assert kernel is not None, "xfp_gemm kernel did not load"
    C = torch.zeros(M, N_out, dtype=torch.bfloat16, device=device)
    kernel.xfp_gemm(x, repacked, codebook, C, int(bits), int(K))

    # Cosine similarity (fp16 rounding makes exact match infeasible).
    cos = F.cosine_similarity(
        expected.to(torch.float32).flatten().unsqueeze(0),
        C.to(torch.float32).flatten().unsqueeze(0),
        dim=1,
    ).item()
    assert cos > 0.999, (
        f"xfp{bits} M={M} N={N_out} K={K} cos={cos:.5f}; "
        f"expected[0,:4]={expected[0, :4].tolist()}, "
        f"got[0,:4]={C[0, :4].tolist()}"
    )


@cuda_only
def test_xfp_gemm_rejects_bad_bits() -> None:
    device = "cuda"
    W = torch.randn(64, 128, dtype=torch.float32)
    packed_cpu, codebook_cpu, _, _, _ = xfp_pack(W, bits=4)
    repacked = xfp_repack(packed_cpu).to(device)
    codebook = codebook_cpu.to(device)
    x = torch.randn(2, 128, dtype=torch.bfloat16, device=device)
    C = torch.zeros(2, 64, dtype=torch.bfloat16, device=device)

    from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm
    kernel = _load_xfp_gemm(4)
    with pytest.raises(RuntimeError):
        kernel.xfp_gemm(x, repacked, codebook, C, 5, 128)
