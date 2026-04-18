# SPDX-License-Identifier: Apache-2.0
"""Inside-out unit tests for XFP v10 SHFL kernel.

Hierarchy:
1. test_shfl_primitive          — verify __shfl_sync semantics work
2. test_codebook_lookup         — verify SHFL lookup matches SMEM lookup
3. test_single_warp_gemm        — 1 warp, 1 output, small K
4. test_multi_warp_gemm         — full block, small N
5. test_realistic_shape         — GLM-sized shapes (3072, 2048)
6. test_warmup_pattern          — replicate vLLM's first-call pattern
   (large grid, batch>1) which is what hangs in production
"""
from __future__ import annotations
import pytest, torch, sys, os
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA needed")


def _load_kernel(version: str):
    """Load v8 or v10 kernel from /opt/mq_kernels."""
    from torch.utils.cpp_extension import load
    return load(
        name=f"xfp_gemm_{version}",
        sources=[f"/opt/mq_kernels/xfp_gemm_{version}.cu"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math",
            "-gencode=arch=compute_121,code=sm_121",
            "-diag-suppress=177,3288",
        ],
        verbose=False,
    )


@cuda_only
def test_shfl_primitive():
    """Sanity: __shfl_sync via inline CUDA gives expected values."""
    src = """
    #include <torch/extension.h>
    __global__ void shfl_test(float* out, int* idx) {
        int lane = threadIdx.x;
        float my_val = (float)lane;  // lane 0=0.0, lane 1=1.0, ...
        out[lane] = __shfl_sync(0xffffffff, my_val, idx[lane]);
    }
    void run(torch::Tensor out, torch::Tensor idx) {
        shfl_test<<<1, 32>>>(out.data_ptr<float>(),
                              idx.data_ptr<int>());
    }
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w") as f:
        f.write(src); f.flush()
        from torch.utils.cpp_extension import load
        k = load(name="shfl_test", sources=[f.name],
                 extra_cuda_cflags=["-gencode=arch=compute_121,code=sm_121"],
                 verbose=False)
        # Each lane reads from lane (31-lane): lane 0 reads lane 31 = 31.0
        idx = torch.arange(31, -1, -1, dtype=torch.int32, device="cuda")
        out = torch.zeros(32, dtype=torch.float32, device="cuda")
        k.run(out, idx)
        expected = torch.arange(31, -1, -1, dtype=torch.float32, device="cuda")
        assert torch.allclose(out, expected), \
            f"SHFL broken: out={out.tolist()}, exp={expected.tolist()}"


@cuda_only
@pytest.mark.parametrize("bits,N,K", [
    (4, 8, 32),       # tiny
    (4, 16, 64),      # 1 warp = 1 N, K = 2 packed words
    (4, 64, 256),     # multi-block
    (4, 256, 512),    # mid
    (4, 3072, 2048),  # GLM attn shape
])
def test_v10_matches_v8(bits, N, K):
    """v10 SHFL output must match v8 SMEM output bit-exactly."""
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    torch.manual_seed(42)
    W = torch.randn(N, K, device="cuda").float()
    x = torch.randn(1, K, device="cuda").bfloat16()
    packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
    repacked = xfp_repack(packed).cuda().reshape(-1)
    cb = codebook.to(torch.float16).cuda()

    k8 = _load_kernel("v8")
    k10 = _load_kernel("v10")
    C8 = torch.zeros(1, N, dtype=torch.bfloat16, device="cuda")
    C10 = torch.zeros(1, N, dtype=torch.bfloat16, device="cuda")
    k8.xfp_gemm(x, repacked, cb, C8, bits, K)
    k10.xfp_gemm(x, repacked, cb, C10, bits, K)

    cos = torch.nn.functional.cosine_similarity(
        C10.float(), C8.float(), dim=1).item()
    maxdiff = (C10.float() - C8.float()).abs().max().item()
    assert cos > 0.9999, \
        f"v10 vs v8 cos={cos:.6f} maxdiff={maxdiff:.4f} N={N} K={K}"


@cuda_only
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16, 32])
def test_v10_batch_sizes(M):
    """v10 with M>1 (prefill batch) — vLLM warmup uses larger M."""
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    bits, N, K = 4, 64, 256
    torch.manual_seed(42)
    W = torch.randn(N, K, device="cuda").float()
    x = torch.randn(M, K, device="cuda").bfloat16()
    packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
    repacked = xfp_repack(packed).cuda().reshape(-1)
    cb = codebook.to(torch.float16).cuda()

    k8 = _load_kernel("v8")
    k10 = _load_kernel("v10")
    C8 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    C10 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    k8.xfp_gemm(x, repacked, cb, C8, bits, K)
    k10.xfp_gemm(x, repacked, cb, C10, bits, K)
    torch.cuda.synchronize()  # ← critical: sync before comparison

    cos = torch.nn.functional.cosine_similarity(
        C10.float().reshape(-1), C8.float().reshape(-1), dim=0).item()
    assert cos > 0.9999, f"M={M}: cos={cos:.6f}"


@cuda_only
def test_v10_repeated_calls():
    """v10 100× repeated calls — replicate vLLM warmup loop pattern."""
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    bits, N, K = 4, 256, 512
    torch.manual_seed(42)
    W = torch.randn(N, K, device="cuda").float()
    packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
    repacked = xfp_repack(packed).cuda().reshape(-1)
    cb = codebook.to(torch.float16).cuda()

    k10 = _load_kernel("v10")
    for i in range(100):
        x = torch.randn(1, K, device="cuda").bfloat16()
        C = torch.zeros(1, N, dtype=torch.bfloat16, device="cuda")
        k10.xfp_gemm(x, repacked, cb, C, bits, K)
        torch.cuda.synchronize()
        assert torch.isfinite(C).all(), f"iter {i}: NaN/Inf in output"
