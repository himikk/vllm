// SPDX-License-Identifier: Apache-2.0
// XFP v6 — multi-warp block with SMEM A-broadcast.
//
// Block = WARPS_PER_BLOCK warps (e.g. 8). Each warp computes 1 output
// element C[m, n_base + warp_id]. All warps in the block share the
// same m row and the same A values via shared memory.
//
// Key insight from v5 regression: register pressure kills occupancy.
// v6 keeps each warp at 16 codebook regs (same as v4) but amortizes
// A loads across WARPS_PER_BLOCK warps via a SMEM tile.
//
// A-tile in SMEM: we load a chunk of A[m, k_chunk..k_chunk+CHUNK_K]
// cooperatively, then all warps process that chunk. This reduces
// global A bandwidth by WARPS_PER_BLOCK × compared to v4.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)  // 256 threads
// Process this many K values per A-tile load. Must be a multiple of
// VALS_PER_WORD for all supported BITS. LCM(16,10,8) = 80, but that's
// too large for SMEM. Use 160 (divisible by 16, 10, 8, and close to
// 128 which is a good SMEM tile width).
#define CHUNK_K 160

template <int BITS>
__global__ void xfp_gemm_v6_kernel(
    const half* __restrict__ A,
    const uint32_t* __restrict__ B_packed,
    const half* __restrict__ codebook,
    half* __restrict__ C,
    int M, int N, int K, int K_packed)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int warp_id = threadIdx.x / WARP_SIZE;   // 0..WARPS_PER_BLOCK-1
    int lane = threadIdx.x % WARP_SIZE;

    int n = blockIdx.x * WARPS_PER_BLOCK + warp_id;  // output column
    int m = blockIdx.y;                                // output row

    if (m >= M) return;

    // Load codebook for this warp's N column — 16 regs, same as v4
    float cb[LUT_SIZE];
    if (n < N) {
        const half* p = codebook + n * LUT_SIZE;
        #pragma unroll
        for (int i = 0; i < LUT_SIZE; i++)
            cb[i] = __half2float(p[i]);
    }

    // Shared memory for A tile — all warps in the block share this
    __shared__ float s_A[CHUNK_K];

    float acc = 0.0f;

    // Process K in chunks of CHUNK_K
    for (int k_start = 0; k_start < K; k_start += CHUNK_K) {
        int chunk_len = min(CHUNK_K, K - k_start);

        // Cooperative A load: all 256 threads load A into SMEM
        for (int i = threadIdx.x; i < chunk_len; i += BLOCK_SIZE) {
            s_A[i] = __half2float(A[m * K + k_start + i]);
        }
        __syncthreads();

        if (n < N) {
            // Determine which packed words cover [k_start, k_start+chunk_len)
            int kw_start = k_start / VALS_PER_WORD;
            int kw_end = (k_start + chunk_len + VALS_PER_WORD - 1) / VALS_PER_WORD;

            // Each lane strides over the packed words
            for (int kw = kw_start + lane; kw < kw_end; kw += WARP_SIZE) {
                uint32_t packed_word = B_packed[kw * N + n];
                int k_base = kw * VALS_PER_WORD;

                #pragma unroll
                for (int slot = 0; slot < VALS_PER_WORD; slot++) {
                    int k = k_base + slot;
                    if (k < k_start || k >= k_start + chunk_len || k >= K)
                        continue;

                    int idx = (int)((packed_word >> (slot * BITS)) & MASK);
                    float w = cb[idx];
                    float a = s_A[k - k_start];
                    acc += w * a;
                }
            }
        }
        __syncthreads();
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0 && n < N) {
        C[m * N + n] = __float2half(acc);
    }
}

template <int BITS>
static void launch_v6(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C, int K)
{
    int M = A.size(0);
    int N = B_packed.size(1);
    int K_packed = B_packed.size(0);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_v6_kernel<BITS><<<grid, block, 0, stream>>>(
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

    if (bits == 2) launch_v6<2>(A, B_packed, codebook, C, static_cast<int>(K));
    else if (bits == 3) launch_v6<3>(A, B_packed, codebook, C, static_cast<int>(K));
    else if (bits == 4) launch_v6<4>(A, B_packed, codebook, C, static_cast<int>(K));
    else TORCH_CHECK(false, "xfp_gemm: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP v6: multi-warp block, SMEM A-broadcast, register LUT");
}
