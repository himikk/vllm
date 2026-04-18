# SPDX-License-Identifier: Apache-2.0
"""Isolated test for MMA primitive: mma.m16n8k16.bf16.bf16.f32.

Before integrating Tensor-Core MMA into the XFP v11 MoE kernel, we verify
the instruction and our lane-fragment layout against torch matmul.
"""
from __future__ import annotations
import sys, pytest, torch
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA needed")


def _load():
    from torch.utils.cpp_extension import load
    return load(
        name="mma_skeleton",
        sources=["/work/kernels/multiquant/mma_skeleton.cu"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math",
            "-gencode=arch=compute_121,code=sm_121",
            "-diag-suppress=177,3288",
        ],
        verbose=False,
    )


@cuda_only
@pytest.mark.parametrize("K", [16, 32, 64, 128, 512])
def test_mma_matches_torch(K):
    """MMA output must match torch.matmul within bf16 precision."""
    torch.manual_seed(42)
    A = torch.randn(16, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, 8, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(16, 8, dtype=torch.float32, device="cuda")

    k = _load()
    k.mma_skeleton(A, B, C, K)
    torch.cuda.synchronize()

    C_ref = (A.float() @ B.float())  # fp32 reference
    cos = torch.nn.functional.cosine_similarity(
        C.reshape(-1), C_ref.reshape(-1), dim=0).item()
    maxdiff = (C - C_ref).abs().max().item()
    relerr = (C - C_ref).abs().max().item() / (C_ref.abs().max().item() + 1e-9)

    assert cos > 0.999, f"K={K}: cos={cos:.6f} maxdiff={maxdiff:.4f}"
    # bf16 precision ~0.8%, reference is fp32
    assert relerr < 0.02, f"K={K}: relerr={relerr:.4f}"
