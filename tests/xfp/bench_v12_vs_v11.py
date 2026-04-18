# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: XFP v12 (SMEM A-row cache) vs v11.

Isolates the kernel-time delta from the A-row caching. Runs on the exact
shapes that appear in Qwen 122B / Qwen 35B / GLM-4.7 decode:

  Linear  — FFN gate_up + down, attn q/k/v (K ≤ K_SMEM_MAX=8192)
  MoE     — gate_up + down per expert (K=2048, K_SMEM_MAX=4096)

Warm-up 50×, measure 500×, take median.
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")


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


def _time_median(fn, warmup=50, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[iters // 2]  # median (ms)


def bench_linear():
    print("=" * 72)
    print("LINEAR — M=1 decode")
    print("=" * 72)
    print(f"{'bits':>4} {'N':>6} {'K':>5}   {'v11 µs':>9} {'v12 µs':>9} "
          f"{'delta %':>9}")
    print("-" * 72)

    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack

    k11 = _load("xfp_gemm_v11", "xfp_gemm_v11.cu")
    k12 = _load("xfp_gemm_v12", "xfp_gemm_v12.cu")

    shapes = [
        # Qwen/GLM realistic Linear decode shapes
        (4,  3072, 2048),   # earlier bench shape
        (4,  9216, 2048),   # attn qkv-ish
        (3,  2048, 3072),
        (4,  4608, 4608),   # Qwen attn projection
        (4,  2048, 2048),   # FFN down
        (4,  8192, 4096),   # K=4096 decode
        (4,  4096, 8192),   # boundary K=K_SMEM_MAX
    ]
    torch.manual_seed(42)
    for bits, N, K in shapes:
        W = torch.randn(N, K, device="cuda").float()
        x = torch.randn(1, K, device="cuda").bfloat16()
        packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
        repacked = xfp_repack(packed).cuda().reshape(-1)
        cb = codebook.to(torch.float16).cuda()
        C = torch.zeros(1, N, dtype=torch.bfloat16, device="cuda")

        def f11():
            k11.xfp_gemm(x, repacked, cb, C, bits, K)

        def f12():
            k12.xfp_gemm(x, repacked, cb, C, bits, K)

        t11 = _time_median(f11) * 1000.0  # µs
        t12 = _time_median(f12) * 1000.0
        delta = (t12 - t11) / t11 * 100.0
        print(f"{bits:>4} {N:>6} {K:>5}   {t11:>9.2f} {t12:>9.2f} "
              f"{delta:>+8.1f}%")


def bench_moe():
    print("=" * 72)
    print("MoE — Qwen 122B decode (B=1, topk=8)")
    print("=" * 72)
    print(f"{'bits':>4} {'E':>3} {'N':>5} {'K':>5}   "
          f"{'v11 µs':>9} {'v12 µs':>9} {'delta %':>9}")
    print("-" * 72)

    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack

    k11 = _load("xfp_moe_gemm_v11", "xfp_moe_gemm_v11.cu")
    k12 = _load("xfp_moe_gemm_v12", "xfp_moe_gemm_v12.cu")

    shapes = [
        # (bits, B, topk, E, N, K)
        (4, 1, 8,   8, 3072, 2048),    # Qwen 35B gate+up decode
        (4, 1, 8,  16, 3072, 2048),    # Qwen 35B more experts
        (4, 1, 8,   8, 1536, 2048),    # GLM gate+up
        (4, 1, 8,   8, 2048, 1536),    # GLM down
        (4, 1, 8,  64, 3072, 2048),    # Qwen 122B decode-like
        (4, 2, 8,  64, 3072, 2048),    # small prefill
    ]
    torch.manual_seed(1234)
    for bits, B, topk, E, N, K in shapes:
        packed_list = []
        cb_list = []
        for _ in range(E):
            W = torch.randn(N, K, device="cuda").float()
            packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
            repacked = xfp_repack(packed).cuda().reshape(-1)
            packed_list.append(repacked)
            cb_list.append(codebook.to(torch.float16).cuda())
        fpe = packed_list[0].numel()
        B_packed = torch.cat(packed_list, dim=0)
        CB = torch.cat(cb_list, dim=0)

        x = torch.randn(B, K, device="cuda").bfloat16()
        topk_ids = torch.randint(0, E, (B, topk), dtype=torch.int32,
                                 device="cuda")
        flat = topk_ids.reshape(-1)
        sort_idx = flat.argsort(stable=True)
        sorted_tok = sort_idx.to(torch.int32)
        sorted_exp = flat[sort_idx].to(torch.int32)
        num_valid = sorted_tok.shape[0]

        BT = B * topk
        no_w = torch.empty(0, dtype=torch.float32, device="cuda")
        C = torch.zeros(BT, N, dtype=torch.bfloat16, device="cuda")

        def f11():
            k11.xfp_moe_gemm(
                x, B_packed, CB, C, sorted_tok, sorted_exp,
                no_w, int(bits), int(K), int(N), int(topk),
                int(fpe), int(num_valid))

        def f12():
            k12.xfp_moe_gemm(
                x, B_packed, CB, C, sorted_tok, sorted_exp,
                no_w, int(bits), int(K), int(N), int(topk),
                int(fpe), int(num_valid))

        t11 = _time_median(f11) * 1000.0
        t12 = _time_median(f12) * 1000.0
        delta = (t12 - t11) / t11 * 100.0
        print(f"{bits:>4} {E:>3} {N:>5} {K:>5}   {t11:>9.2f} {t12:>9.2f} "
              f"{delta:>+8.1f}%")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    bench_linear()
    print()
    bench_moe()
