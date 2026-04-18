# SPDX-License-Identifier: Apache-2.0
"""Test MMA kernel with XFP-decoded B against v8 reference."""
from __future__ import annotations
import sys, pytest, torch
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA needed")


def _load(name, path):
    from torch.utils.cpp_extension import load
    return load(
        name=name, sources=[path],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math",
            "-gencode=arch=compute_121,code=sm_121",
            "-diag-suppress=177,3288",
        ],
        verbose=False,
    )


@cuda_only
@pytest.mark.parametrize("K", [16, 32, 64, 256, 2048])
def test_xfp_mma_single_matches_v8(K):
    """MMA w/ XFP-decoded B must match xfp_gemm_v8 (N=8, M=16)."""
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack

    bits, N, M = 4, 8, 16
    torch.manual_seed(42)
    W = torch.randn(N, K, device="cuda").float()
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
    repacked = xfp_repack(packed).cuda().reshape(-1)
    cb = codebook.to(torch.float16).cuda()

    k_v8 = _load("xfp_gemm_v8", "/work/kernels/multiquant/xfp_gemm_v8.cu")
    k_mma = _load("xfp_mma_single",
                  "/work/kernels/multiquant/xfp_gemm_mma_single.cu")

    C_v8 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    k_v8.xfp_gemm(x, repacked, cb, C_v8, bits, K)

    C_mma = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    k_mma.xfp_mma_single(x, repacked, cb, C_mma, bits, K)
    torch.cuda.synchronize()

    cos = torch.nn.functional.cosine_similarity(
        C_mma.reshape(-1), C_v8.float().reshape(-1), dim=0).item()
    maxdiff = (C_mma - C_v8.float()).abs().max().item()
    refmag = C_v8.float().abs().max().item()

    # bf16 precision in v8 vs fp32 accumulator in MMA → some diff expected
    assert cos > 0.9999, \
        f"K={K}: cos={cos:.6f} maxdiff={maxdiff:.4f} refmag={refmag:.2f}"
