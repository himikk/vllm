# SPDX-License-Identifier: Apache-2.0
"""Bitwise equivalence tests for XFP v12 (static SMEM A-row cache).

v12 caches the block-uniform A-row in __shared__ bytes before the K-loop.
Since every warp of a block reads the same A-row, the values read from
SMEM are bit-identical to the per-warp global reads of v11. This test
enforces that invariant: v12 output must match v11 output exactly
(cos=1.0, maxdiff=0) across Linear and MoE shapes.

It also checks that K > K_SMEM_MAX triggers the host-side TORCH_CHECK
(rigid implementation — no kernel fallback, caller must use v11).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA needed"
)

# Keep in sync with Policies in xfp_gemm_core.cuh
K_SMEM_MAX_LINEAR = 8192
K_SMEM_MAX_MOE = 4096


def _kernel_src(name: str) -> str:
    for base in (
        os.path.expanduser("~/vllm-riy/kernels/multiquant"),
        "/opt/mq_kernels",
    ):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def _load(mod_name: str, cu_name: str):
    from torch.utils.cpp_extension import load
    return load(
        name=mod_name,
        sources=[_kernel_src(cu_name)],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math",
            "-gencode=arch=compute_120,code=sm_120",
            "-gencode=arch=compute_121,code=sm_121",
            "-diag-suppress=177,3288",
        ],
        verbose=False,
    )


@pytest.fixture(scope="module")
def k_linear_v11():
    return _load("xfp_gemm_v11", "xfp_gemm_v11.cu")


@pytest.fixture(scope="module")
def k_linear_v12():
    return _load("xfp_gemm_v12", "xfp_gemm_v12.cu")


@pytest.fixture(scope="module")
def k_moe_v11():
    return _load("xfp_moe_gemm_v11", "xfp_moe_gemm_v11.cu")


@pytest.fixture(scope="module")
def k_moe_v12():
    return _load("xfp_moe_gemm_v12", "xfp_moe_gemm_v12.cu")


# ─── Linear v12 ≡ v11 ──────────────────────────────────────────────────

@cuda_only
@pytest.mark.parametrize("bits,M,N,K", [
    (4,  1,   64,   256),   # tiny decode
    (4,  1,  256,   512),   # multi-block
    (4,  1, 3072,  2048),   # Qwen MoE K (reused by linear)
    (4,  4, 1024,   768),   # small prefill
    (4,  8, 2048,  4096),   # medium prefill
    (4, 16, 3072,  8192),   # boundary: K == K_SMEM_MAX_LINEAR
    (3,  1,  256,   512),
    (3,  4, 2048,  3072),
    (2,  1, 1024,   768),
    (2,  8,  512,  2048),
])
def test_linear_v12_matches_v11(
    k_linear_v11, k_linear_v12, bits, M, N, K
):
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    torch.manual_seed(42)
    W = torch.randn(N, K, device="cuda").float()
    x = torch.randn(M, K, device="cuda").bfloat16()

    packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
    repacked = xfp_repack(packed).cuda().reshape(-1)
    cb = codebook.to(torch.float16).cuda()

    C11 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    C12 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    k_linear_v11.xfp_gemm(x, repacked, cb, C11, bits, K)
    k_linear_v12.xfp_gemm(x, repacked, cb, C12, bits, K)
    torch.cuda.synchronize()

    # Bitwise equivalence: SMEM load just moves bits around, no conversion.
    maxdiff = (C12.float() - C11.float()).abs().max().item()
    assert maxdiff == 0.0, (
        f"v12 vs v11 not bitwise identical: bits={bits} "
        f"M={M} N={N} K={K} maxdiff={maxdiff}"
    )


@cuda_only
def test_linear_v12_rejects_oversize_K(k_linear_v12):
    """K > K_SMEM_MAX_LINEAR must trigger TORCH_CHECK (rigid — no fallback)."""
    bits, M, N, K = 4, 1, 64, K_SMEM_MAX_LINEAR + 256
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda").bfloat16()
    # Fabricate valid-but-unused tensors; check must fire before kernel launch.
    packed = torch.zeros(N * (K // 8), dtype=torch.int32, device="cuda")
    cb = torch.zeros(N, 16, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="K_SMEM_MAX"):
        k_linear_v12.xfp_gemm(x, packed, cb, C, bits, K)


# ─── MoE v12 ≡ v11 ─────────────────────────────────────────────────────

def _make_moe_routing(B, topk, E, device):
    """Build sorted_token_ids / expert_ids analogous to online_moe.py."""
    # Random topk expert choice, uniform over E.
    topk_ids = torch.randint(0, E, (B, topk), dtype=torch.int32, device=device)
    flat = topk_ids.reshape(-1)
    sort_idx = flat.argsort(stable=True)
    sorted_token_ids = sort_idx.to(torch.int32)
    sorted_expert_ids = flat[sort_idx].to(torch.int32)
    num_valid = sorted_token_ids.shape[0]
    return sorted_token_ids, sorted_expert_ids, num_valid


@cuda_only
@pytest.mark.parametrize("bits,B,topk,E,N,K", [
    (4, 1, 8,   8,  256,  512),    # tiny decode
    (4, 1, 8,  16, 3072, 2048),    # Qwen MoE decode shape
    (4, 2, 8,  64, 3072, 2048),    # small prefill
    (4, 4, 4,  64, 1536, 2048),    # GLM MoE-ish
    (4, 1, 8,   8, 1024, 4096),    # boundary: K == K_SMEM_MAX_MOE
    (3, 1, 8,  16,  512, 2048),
    (3, 2, 4,   8,  768, 1536),
    (2, 1, 4,   4,  256,  768),
])
def test_moe_v12_matches_v11(
    k_moe_v11, k_moe_v12, bits, B, topk, E, N, K
):
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    torch.manual_seed(1234)

    # Pack per-expert weights. flat_per_expert = N * K_packed (int32 words).
    packed_list = []
    cb_list = []
    for _ in range(E):
        W = torch.randn(N, K, device="cuda").float()
        packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
        repacked = xfp_repack(packed).cuda().reshape(-1)
        packed_list.append(repacked)
        cb_list.append(codebook.to(torch.float16).cuda())

    fpe = packed_list[0].numel()
    for p in packed_list:
        assert p.numel() == fpe, "Inconsistent flat_per_expert across experts"
    B_packed = torch.cat(packed_list, dim=0)
    CB = torch.cat(cb_list, dim=0)  # [E*N, 2^bits]

    x = torch.randn(B, K, device="cuda").bfloat16()
    sorted_tok, sorted_exp, num_valid = _make_moe_routing(
        B, topk, E, device="cuda")

    BT = B * topk
    no_w = torch.empty(0, dtype=torch.float32, device="cuda")
    C11 = torch.zeros(BT, N, dtype=torch.bfloat16, device="cuda")
    C12 = torch.zeros(BT, N, dtype=torch.bfloat16, device="cuda")

    k_moe_v11.xfp_moe_gemm(
        x, B_packed, CB, C11, sorted_tok, sorted_exp,
        no_w, int(bits), int(K), int(N), int(topk),
        int(fpe), int(num_valid))
    k_moe_v12.xfp_moe_gemm(
        x, B_packed, CB, C12, sorted_tok, sorted_exp,
        no_w, int(bits), int(K), int(N), int(topk),
        int(fpe), int(num_valid))
    torch.cuda.synchronize()

    maxdiff = (C12.float() - C11.float()).abs().max().item()
    assert maxdiff == 0.0, (
        f"MoE v12 vs v11 not bitwise identical: bits={bits} "
        f"B={B} topk={topk} E={E} N={N} K={K} maxdiff={maxdiff}"
    )


@cuda_only
def test_moe_v12_rejects_oversize_K(k_moe_v12):
    """MoE K > K_SMEM_MAX_MOE must TORCH_CHECK."""
    bits, B, topk, E, N, K = 4, 1, 4, 4, 64, K_SMEM_MAX_MOE + 256
    x = torch.randn(B, K, device="cuda").bfloat16()
    fpe = N * (K // 8)
    B_packed = torch.zeros(E * fpe, dtype=torch.int32, device="cuda")
    CB = torch.zeros(E * N, 16, dtype=torch.float16, device="cuda")
    sorted_tok = torch.arange(B * topk, dtype=torch.int32, device="cuda")
    sorted_exp = torch.zeros(B * topk, dtype=torch.int32, device="cuda")
    no_w = torch.empty(0, dtype=torch.float32, device="cuda")
    C = torch.zeros(B * topk, N, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="K_SMEM_MAX"):
        k_moe_v12.xfp_moe_gemm(
            x, B_packed, CB, C, sorted_tok, sorted_exp,
            no_w, int(bits), int(K), int(N), int(topk),
            int(fpe), int(B * topk))
