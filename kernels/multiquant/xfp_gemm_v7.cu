// SPDX-License-Identifier: Apache-2.0
// XFP v7 — coalesced-N warp kernel.
//
// Key insight: v4 splits K across 32 lanes → non-coalesced weight reads
// (stride N between consecutive kw values for the same lane). v7 splits
// N across 32 lanes → perfectly coalesced reads (consecutive N addresses
// at the same kw form a 128-byte cache line).
//
// Design:
//   Block = WARPS_PER_BLOCK warps × 32 lanes = 256 threads
//   Each warp → 32 consecutive N columns (lane i → n_base + i)
//   Each lane → full K reduction for its column (no K-split, no atomicAdd)
//   A broadcast: lane 0 loads A[m,k], __shfl_sync to all 31 others
//   Codebook: 16 regs per lane (same as v4)
//   Grid: (ceil(N/32/WARPS_PER_BLOCK), M)
//
// Memory pattern: all 32 lanes read B_packed[kw * N + n_base .. n_base+31]
// = 32 consecutive uint32 = 128 bytes = exactly one L2 cache line.
// This is 32× more bandwidth-efficient than v4's strided access.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

template <int BITS>
__global__ void xfp_gemm_v7_kernel(
    const half* __restrict__ A,
    const uint32_t* __restrict__ B_packed,  // [K_packed, N]
    const half* __restrict__ codebook,      // [N, 2^BITS]
    half* __restrict__ C,                   // [M, N]
    int M, int N, int K, int K_packed)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Each warp handles 32 consecutive N columns
    int n_base = (blockIdx.x * WARPS_PER_BLOCK + warp_id) * WARP_SIZE;
    int n = n_base + lane;
    int m = blockIdx.y;

    if (m >= M) return;

    // Load this lane's codebook into registers (16 floats for BITS=4)
    float cb[LUT_SIZE];
    if (n < N) {
        const half* p = codebook + n * LUT_SIZE;
        #pragma unroll
        for (int i = 0; i < LUT_SIZE; i++)
            cb[i] = __half2float(p[i]);
    }

    float acc = 0.0f;

    // Main K loop — all lanes in the warp process the SAME kw
    // → coalesced reads across the N dimension
    for (int kw = 0; kw < K_packed; kw++) {
        // COALESCED: lanes read consecutive N addresses
        uint32_t packed = (n < N) ? B_packed[kw * N + n] : 0u;

        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot++) {
            int k = k_base + slot;
            if (k >= K) break;

            // A broadcast: lane 0 loads, all receive via shfl
            float a = __half2float(A[m * K + k]);
            // All lanes need the same A value — on SM121, the L1 cache
            // serves this efficiently (all lanes hit the same address).
            // We keep the direct load instead of shfl because:
            // 1. L1 coalesces the broadcast naturally
            // 2. shfl adds instruction latency
            // 3. v6 showed explicit broadcast doesn't help

            int idx = (int)((packed >> (slot * BITS)) & MASK);
            float w = cb[idx];
            acc += w * a;
        }
    }

    // Each lane writes its own output — no reduction needed
    if (n < N) {
        C[m * N + n] = __float2half(acc);
    }
}

template <int BITS>
static void launch_v7(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C, int K)
{
    int M = A.size(0);
    int N = B_packed.size(1);
    int K_packed = B_packed.size(0);

    int warps_n = (N + WARP_SIZE - 1) / WARP_SIZE;  // warps needed for N
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (warps_n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
        M
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_v7_kernel<BITS><<<grid, block, 0, stream>>>(
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

    if (bits == 2) launch_v7<2>(A, B_packed, codebook, C, static_cast<int>(K));
    else if (bits == 3) launch_v7<3>(A, B_packed, codebook, C, static_cast<int>(K));
    else if (bits == 4) launch_v7<4>(A, B_packed, codebook, C, static_cast<int>(K));
    else TORCH_CHECK(false, "xfp_gemm: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP v7: coalesced-N warp, register LUT, no reduction");
}
