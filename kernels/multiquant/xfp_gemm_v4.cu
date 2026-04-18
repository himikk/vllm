// SPDX-License-Identifier: Apache-2.0
// XFP v4 fused decode kernel — warp-per-output-element, register-cached LUT.
//
// Design principles:
//   - 1 warp (32 threads) computes 1 output element C[m, n]
//   - 32 threads split K: each thread handles K/32 packed words
//   - A[m, k] broadcast via __shfl_sync (1 load, 31 receives)
//   - Codebook[n, 2^BITS] cached in thread-local registers (~32 bytes)
//   - Warp-level reduction via __shfl_down_sync (no atomicAdd)
//   - Grid: (N, M) — no Z-split, no inter-block synchronization
//
// Expected: bandwidth-bound on packed weight reads. At XFP4 on GB10
// (273 GB/s), a 2048×2048 weight matrix at 0.5 bytes/val = 2 MB,
// theoretical peak ~130K mat-vec/s. For decode (M=1) the bottleneck
// is latency not bandwidth, so real throughput depends on occupancy.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

#define WARP_SIZE 32

template <int BITS>
__global__ void xfp_gemm_v4_kernel(
    const half* __restrict__ A,             // [M, K]
    const uint32_t* __restrict__ B_packed,  // [K_packed, N] uint32
    const half* __restrict__ codebook,      // [N, 2^BITS] fp16
    half* __restrict__ C,                   // [M, N] fp16
    int M, int N, int K, int K_packed)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    // Grid mapping: blockIdx.x → N columns, blockIdx.y → M rows
    // Each block has exactly 1 warp (32 threads) computing 1 output.
    int n = blockIdx.x;   // output column
    int m = blockIdx.y;   // output row
    int lane = threadIdx.x;

    if (n >= N || m >= M) return;

    // Load codebook for column n into registers.
    // LUT_SIZE is 4/8/16 — fits comfortably in registers.
    float cb[LUT_SIZE];
    const half* cb_ptr = codebook + n * LUT_SIZE;
    #pragma unroll
    for (int i = 0; i < LUT_SIZE; i++) {
        cb[i] = __half2float(cb_ptr[i]);
    }

    // Each thread processes words [lane, lane+32, lane+64, ...] of column n.
    // Partial accumulator.
    float acc = 0.0f;

    for (int kw = lane; kw < K_packed; kw += WARP_SIZE) {
        uint32_t packed_word = B_packed[kw * N + n];
        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot++) {
            int k = k_base + slot;
            if (k >= K) break;

            int idx = (int)((packed_word >> (slot * BITS)) & MASK);
            float w = cb[idx];

            // A[m, k] — all threads in the warp need this same value.
            // Thread (k % 32) owns it; broadcast via __shfl_sync.
            // But k varies per slot and per kw, so we can't pre-load.
            // Instead: direct global load — L1 coalesces across warp.
            float a = __half2float(A[m * K + k]);

            acc += w * a;
        }
    }

    // Warp-level reduction — sum across all 32 lanes.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 writes the result.
    if (lane == 0) {
        C[m * N + n] = __float2half(acc);
    }
}

template <int BITS>
static void launch_v4(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C, int K)
{
    int M = A.size(0);
    int N = B_packed.size(1);
    int K_packed = B_packed.size(0);

    // 1 warp per output element, grid = (N, M)
    dim3 block(WARP_SIZE);
    dim3 grid(N, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_v4_kernel<BITS><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(A.data_ptr()),
        reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(codebook.data_ptr()),
        reinterpret_cast<half*>(C.data_ptr()),
        M, N, K, K_packed);
}

void xfp_gemm(
    torch::Tensor A,
    torch::Tensor B_packed,
    torch::Tensor codebook,
    torch::Tensor C,
    int64_t bits,
    int64_t K)
{
    TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() &&
                codebook.is_cuda() && C.is_cuda(),
                "xfp_gemm: all tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "xfp_gemm: A must be float16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32,
                "xfp_gemm: B_packed must be int32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16,
                "xfp_gemm: codebook must be float16");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "xfp_gemm: C must be float16");
    TORCH_CHECK(A.dim() == 2 && B_packed.dim() == 2 &&
                codebook.dim() == 2 && C.dim() == 2,
                "xfp_gemm: all tensors must be 2D");
    TORCH_CHECK(A.size(1) == K, "xfp_gemm: A.size(1) must equal K");
    TORCH_CHECK(A.size(0) == C.size(0) && B_packed.size(1) == C.size(1),
                "xfp_gemm: M/N shape mismatch");
    TORCH_CHECK(codebook.size(0) == B_packed.size(1),
                "xfp_gemm: codebook rows must equal N");
    TORCH_CHECK(codebook.size(1) == (1LL << bits),
                "xfp_gemm: codebook columns must equal 2^bits");

    if (bits == 2) launch_v4<2>(A, B_packed, codebook, C, static_cast<int>(K));
    else if (bits == 3) launch_v4<3>(A, B_packed, codebook, C, static_cast<int>(K));
    else if (bits == 4) launch_v4<4>(A, B_packed, codebook, C, static_cast<int>(K));
    else TORCH_CHECK(false, "xfp_gemm: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP v4 fused decode: warp-per-element, register LUT, shfl reduction");
}
