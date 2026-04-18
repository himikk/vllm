#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile where time goes during XFP inference.

Measures:
1. Isolated kernel time (xfp_gemm only)
2. Custom op overhead (xfp_apply wrapper)
3. Outlier scatter-add time
4. bf16→fp16→bf16 dtype conversion time
5. Full forward pass via vLLM API (wall clock per token)
6. Breakdown: kernel fraction vs overhead fraction

Usage: run inside container with vLLM server already running, OR
standalone with model weights loaded.
"""

import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")


def profile_isolated_kernel():
    """Pure kernel time — no Python wrapper."""
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm

    print("=== 1. Isolated Kernel Timing ===")
    kernel = _load_xfp_gemm(4)

    sizes = [
        (768, 2048, "attn q_a_proj"),
        (2048, 5120, "attn o_proj"),
        (5120, 768, "attn q_b_proj"),
        (3072, 2048, "routed gate_up"),
        (2048, 1536, "routed down"),
        (8960, 512, "kv_b_proj"),
    ]

    for N_out, K, name in sizes:
        W = torch.randn(N_out, K) * 0.01
        packed, cb, _, _, _ = xfp_pack(W, bits=4)
        repacked = xfp_repack(packed).cuda()
        cb_gpu = cb.cuda()
        x = torch.randn(1, K, dtype=torch.float16, device="cuda")
        C = torch.zeros(1, N_out, dtype=torch.float16, device="cuda")

        # Warmup
        for _ in range(20):
            kernel.xfp_gemm(x, repacked, cb_gpu, C, 4, K)
        torch.cuda.synchronize()

        # Measure
        n_iter = 500
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            kernel.xfp_gemm(x, repacked, cb_gpu, C, 4, K)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        us = (t1 - t0) / n_iter * 1e6
        gb = (N_out * K * 0.5 + K * 2) / 1e9  # packed weights + A
        bw = gb / (us * 1e-6)
        print(f"  {name:20s} [{N_out:5d}x{K:4d}]: {us:7.1f} us  "
              f"({bw:.1f} GB/s, {bw/273*100:.0f}% of 273 GB/s peak)")


def profile_custom_op_overhead():
    """Custom op wrapper overhead: xfp_apply includes bf16→fp16→bf16."""
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm
    from vllm.multiquant.xfp.online_linear import _xfp_apply_impl

    print("\n=== 2. Custom Op Overhead (includes dtype cast) ===")
    kernel = _load_xfp_gemm(4)

    N_out, K = 2048, 2048
    W = torch.randn(N_out, K) * 0.01
    packed, cb, _, _, _ = xfp_pack(W, bits=4)
    repacked = xfp_repack(packed).cuda()
    cb_gpu = cb.cuda()

    # bf16 input (what vLLM actually sends)
    x_bf16 = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")

    # a) Raw kernel (fp16 in, fp16 out)
    x_fp16 = x_bf16.to(torch.float16)
    C = torch.zeros(1, N_out, dtype=torch.float16, device="cuda")
    for _ in range(20):
        kernel.xfp_gemm(x_fp16, repacked, cb_gpu, C, 4, K)
    torch.cuda.synchronize()
    n = 500
    t0 = time.perf_counter()
    for _ in range(n):
        kernel.xfp_gemm(x_fp16, repacked, cb_gpu, C, 4, K)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    raw_us = (t1 - t0) / n * 1e6

    # b) Custom op (bf16 in → fp16 cast → kernel → bf16 cast out)
    for _ in range(20):
        _xfp_apply_impl(x_bf16, repacked, cb_gpu, 4, K, N_out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        _xfp_apply_impl(x_bf16, repacked, cb_gpu, 4, K, N_out)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    op_us = (t1 - t0) / n * 1e6

    print(f"  Raw kernel (fp16→fp16):     {raw_us:7.1f} us")
    print(f"  Custom op (bf16→fp16→bf16): {op_us:7.1f} us")
    print(f"  Overhead:                   {op_us - raw_us:7.1f} us "
          f"({(op_us/raw_us - 1)*100:.0f}%)")


def profile_dtype_cast():
    """Isolated bf16↔fp16 conversion cost."""
    print("\n=== 3. Dtype Conversion Cost ===")
    for size in [2048, 5120, 8960]:
        x = torch.randn(1, size, dtype=torch.bfloat16, device="cuda")
        # Warmup
        for _ in range(100):
            y = x.to(torch.float16)
        torch.cuda.synchronize()
        n = 5000
        t0 = time.perf_counter()
        for _ in range(n):
            y = x.to(torch.float16)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        us = (t1 - t0) / n * 1e6
        print(f"  bf16→fp16 [1×{size}]: {us:.2f} us")


def profile_outlier_scatter():
    """Outlier scatter-add cost."""
    from vllm.multiquant.xfp.online_linear import _xfp_outlier_scatter_impl
    print("\n=== 4. Outlier Scatter-Add Cost ===")

    N_out, K = 2048, 2048
    C = torch.randn(1, N_out, dtype=torch.float16, device="cuda")
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")

    for n_outliers in [0, 50, 200, 1000]:
        if n_outliers == 0:
            print(f"  {n_outliers:5d} outliers: 0.0 us (skip)")
            continue
        rows = torch.randint(0, N_out, (n_outliers,), device="cuda")
        cols = torch.randint(0, K, (n_outliers,), device="cuda")
        vals = torch.randn(n_outliers, dtype=torch.float16, device="cuda") * 0.01

        for _ in range(100):
            _xfp_outlier_scatter_impl(C, x, rows, cols, vals)
        torch.cuda.synchronize()
        n = 2000
        t0 = time.perf_counter()
        for _ in range(n):
            _xfp_outlier_scatter_impl(C, x, rows, cols, vals)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        us = (t1 - t0) / n * 1e6
        print(f"  {n_outliers:5d} outliers: {us:.1f} us")


def profile_full_forward_api():
    """Wall-clock per token via the running vLLM server."""
    import urllib.request
    import json
    print("\n=== 5. Full vLLM API Token Timing ===")

    url = "http://localhost:8011/v1/chat/completions"
    data = json.dumps({
        "model": "glm-4.7-flash",
        "messages": [{"role": "user", "content": "Count from 1 to 50."}],
        "max_tokens": 60,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    # Warmup
    for _ in range(2):
        try:
            urllib.request.urlopen(req, timeout=30)
        except Exception:
            print("  Server not reachable — skip API timing")
            return

    # Measure
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=60)
    t1 = time.perf_counter()
    result = json.loads(resp.read())
    tokens = result["usage"]["completion_tokens"]
    elapsed = t1 - t0
    print(f"  {tokens} tokens in {elapsed:.2f}s = {tokens/elapsed:.1f} tok/s")
    print(f"  Per-token latency: {elapsed/tokens*1000:.1f} ms")


if __name__ == "__main__":
    profile_isolated_kernel()
    profile_custom_op_overhead()
    profile_dtype_cast()
    profile_outlier_scatter()
    profile_full_forward_api()

    print("\n=== Summary ===")
    print("Compare isolated kernel μs to per-token API latency.")
    print("The gap is Python/vLLM overhead + other layers (attention, MoE routing, norms, etc.)")
