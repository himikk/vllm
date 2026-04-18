// SPDX-License-Identifier: Apache-2.0
// XFP v5 — multi-output warp kernel: 1 warp computes N_PER_WARP output elements.
//
// Amortizes A loads across multiple N columns: each thread caches
// N_PER_WARP codebooks in registers and processes them with the same
// A value. This reduces global A reads by N_PER_WARP × vs v4.
//
// Grid: (ceil(N / N_PER_WARP), M), block = 32 (1 warp)

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

#define WARP_SIZE 32
#define N_PER_WARP 4

template <int BITS>
__global__ void xfp_gemm_v5_kernel(
    const half* __restrict__ A,
    const uint32_t* __restrict__ B_packed,  // [K_packed, N]
    const half* __restrict__ codebook,      // [N, 2^BITS]
    half* __restrict__ C,                   // [M, N]
    int M, int N, int K, int K_packed)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int n_base = blockIdx.x * N_PER_WARP;
    int m = blockIdx.y;
    int lane = threadIdx.x;

    if (m >= M) return;

    // Load N_PER_WARP codebooks into registers.
    // Total: N_PER_WARP * LUT_SIZE floats. For BITS=4: 4 * 16 = 64 regs.
    float cb[N_PER_WARP][LUT_SIZE];
    #pragma unroll
    for (int j = 0; j < N_PER_WARP; j++) {
        int n = n_base + j;
        if (n < N) {
            const half* p = codebook + n * LUT_SIZE;
            #pragma unroll
            for (int i = 0; i < LUT_SIZE; i++)
                cb[j][i] = __half2float(p[i]);
        }
    }

    // Accumulators
    float acc[N_PER_WARP];
    #pragma unroll
    for (int j = 0; j < N_PER_WARP; j++) acc[j] = 0.0f;

    // Each thread strides over K_packed at lane offset
    for (int kw = lane; kw < K_packed; kw += WARP_SIZE) {
        // Load packed words for each of the N_PER_WARP columns
        uint32_t w[N_PER_WARP];
        #pragma unroll
        for (int j = 0; j < N_PER_WARP; j++) {
            int n = n_base + j;
            w[j] = (n < N) ? B_packed[kw * N + n] : 0u;
        }

        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot++) {
            int k = k_base + slot;
            if (k >= K) break;

            // ONE A load, shared across all N_PER_WARP columns
            float a = __half2float(A[m * K + k]);

            #pragma unroll
            for (int j = 0; j < N_PER_WARP; j++) {
                int idx = (int)((w[j] >> (slot * BITS)) & MASK);
                acc[j] += cb[j][idx] * a;
            }
        }
    }

    // Warp-level reduction for each of the N_PER_WARP accumulators
    #pragma unroll
    for (int j = 0; j < N_PER_WARP; j++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[j] += __shfl_down_sync(0xffffffff, acc[j], offset);
        }
    }

    // Lane 0 writes N_PER_WARP results
    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < N_PER_WARP; j++) {
            int n = n_base + j;
            if (n < N) {
                C[m * N + n] = __float2half(acc[j]);
            }
        }
    }
}

template <int BITS>
static void launch_v5(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C, int K)
{
    int M = A.size(0);
    int N = B_packed.size(1);
    int K_packed = B_packed.size(0);

    dim3 block(WARP_SIZE);
    dim3 grid((N + N_PER_WARP - 1) / N_PER_WARP, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_v5_kernel<BITS><<<grid, block, 0, stream>>>(
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

    if (bits == 2) launch_v5<2>(A, B_packed, codebook, C, static_cast<int>(K));
    else if (bits == 3) launch_v5<3>(A, B_packed, codebook, C, static_cast<int>(K));
    else if (bits == 4) launch_v5<4>(A, B_packed, codebook, C, static_cast<int>(K));
    else TORCH_CHECK(false, "xfp_gemm: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP v5 fused decode: multi-output warp, register LUT, shfl reduction");
}
