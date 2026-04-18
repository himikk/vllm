// SPDX-License-Identifier: Apache-2.0
// XFP v4opt — micro-optimized warp-per-element kernel with weight repack.
//
// Base: v4 design (1 warp = 1 output element, register codebook, shfl reduce).
// Optimizations:
//   A: Unrolled main K-loop without bounds check (tail handled separately)
//   B: half2 A-loads (2 K values per load instruction)
//   C: Vectorized codebook init via uint32 + half2 cast
//   D: Software pipelining (prefetch next packed word while processing current)
//   E: FMA instead of separate mul+add (fmaf = 1 instruction vs 2)
//   F: Weight repack for coalesced warp reads (xfp_repack in xfp_pack.py)
//
// B_packed layout (repacked): [K_groups * N * 32] int32 flattened from
// [K_groups, N, 32] where the K dimension is interleaved over WARP_SIZE.
// Lane i reads B_packed[kw_group * N * 32 + n * 32 + i] — all 32 lanes
// hit consecutive addresses = 1 cache line = 100% L2 utilization.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

#define WARP_SIZE 32

template <int BITS>
__global__ void xfp_gemm_v4opt_kernel(
    const half* __restrict__ A,
    const uint32_t* __restrict__ B_packed,  // repacked flat [K_groups*N*32]
    const half* __restrict__ codebook,      // [N, 2^BITS]
    half* __restrict__ C,                   // [M, N]
    int M, int N, int K, int K_packed)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int n = blockIdx.x;
    int m = blockIdx.y;
    int lane = threadIdx.x;

    if (n >= N || m >= M) return;

    // === Patch C: Vectorized codebook init ===
    float cb[LUT_SIZE];
    {
        const uint32_t* cb_u32 = reinterpret_cast<const uint32_t*>(
            codebook + n * LUT_SIZE);
        #pragma unroll
        for (int i = 0; i < LUT_SIZE / 2; i++) {
            uint32_t packed_fp16 = cb_u32[i];
            half2 h2 = *reinterpret_cast<const half2*>(&packed_fp16);
            cb[2 * i]     = __low2float(h2);
            cb[2 * i + 1] = __high2float(h2);
        }
    }

    const half* A_row = A + m * K;
    float acc = 0.0f;

    // === Repack addressing (Patch F) ===
    // B_packed is [K_groups, N, WARP_SIZE] flattened.
    // Lane i at group g reads: B_packed[g * N * WARP_SIZE + n * WARP_SIZE + lane]
    int n_offset = n * WARP_SIZE + lane;  // fixed per thread

    int K_packed_safe = K / VALS_PER_WORD;

    // === Patch D: Prefetch first packed word ===
    int kw = lane;
    int g = 0;
    uint32_t buf_cur = (kw < K_packed)
        ? B_packed[g * N * WARP_SIZE + n_offset] : 0u;

    // === Patch A: Unrolled main loop ===
    for (; kw < K_packed_safe; kw += WARP_SIZE, g++) {
        // Patch D: prefetch next group
        uint32_t buf_next = 0u;
        int kw_next = kw + WARP_SIZE;
        if (kw_next < K_packed)
            buf_next = B_packed[(g + 1) * N * WARP_SIZE + n_offset];

        int k_base = kw * VALS_PER_WORD;

        // === Patch B: half2 A-loads + Patch E: FMA ===
        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot += 2) {
            half2 a2 = *reinterpret_cast<const half2*>(A_row + k_base + slot);
            float a0 = __low2float(a2);
            float a1 = __high2float(a2);

            int idx0 = (int)((buf_cur >> (slot * BITS)) & MASK);
            int idx1 = (int)((buf_cur >> ((slot + 1) * BITS)) & MASK);

            acc = fmaf(cb[idx0], a0, acc);
            acc = fmaf(cb[idx1], a1, acc);
        }

        buf_cur = buf_next;
    }

    // === Tail: remaining words with bounds check ===
    for (; kw < K_packed; kw += WARP_SIZE, g++) {
        uint32_t packed = B_packed[g * N * WARP_SIZE + n_offset];
        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot++) {
            int k = k_base + slot;
            if (k >= K) break;
            float a = __half2float(A_row[k]);
            int idx = (int)((packed >> (slot * BITS)) & MASK);
            acc = fmaf(cb[idx], a, acc);
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        C[m * N + n] = __float2half(acc);
    }
}

template <int BITS>
static void launch_v4opt(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C, int K, int N, int K_packed)
{
    int M = A.size(0);

    dim3 block(WARP_SIZE);
    dim3 grid(N, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_v4opt_kernel<BITS><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(A.data_ptr()),
        reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(codebook.data_ptr()),
        reinterpret_cast<half*>(C.data_ptr()),
        M, N, K, K_packed);
}

void xfp_gemm(
    torch::Tensor A,         // [M, K] fp16
    torch::Tensor B_packed,  // [K_groups * N * 32] int32 (repacked flat)
    torch::Tensor codebook,  // [N, 2^bits] fp16
    torch::Tensor C,         // [M, N] fp16
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
    TORCH_CHECK(A.dim() == 2 && C.dim() == 2,
                "xfp_gemm: A and C must be 2D");
    TORCH_CHECK(B_packed.dim() == 1,
                "xfp_gemm: B_packed must be 1D (repacked flat)");
    TORCH_CHECK(codebook.dim() == 2,
                "xfp_gemm: codebook must be 2D");
    TORCH_CHECK(A.size(1) == K, "xfp_gemm: A.size(1) must equal K");
    TORCH_CHECK(codebook.size(1) == (1LL << bits),
                "xfp_gemm: codebook columns must equal 2^bits");

    int N = static_cast<int>(codebook.size(0));
    int vals_per_word = (bits == 2) ? 16 : (bits == 3) ? 10 : 8;
    int K_packed = (static_cast<int>(K) + vals_per_word - 1) / vals_per_word;

    TORCH_CHECK(A.size(0) == C.size(0) && C.size(1) == N,
                "xfp_gemm: M/N shape mismatch");

    if (bits == 2) launch_v4opt<2>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 3) launch_v4opt<3>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 4) launch_v4opt<4>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else TORCH_CHECK(false, "xfp_gemm: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP v4opt+repack: FMA, unrolled, half2, prefetch, coalesced warp reads");
}
