// SPDX-License-Identifier: Apache-2.0
// XFP v8 — SMEM codebook pool for max occupancy / latency hiding.
//
// ncu on v4opt showed: 83% of cycles idle (0.21 eligible warps/scheduler).
// Root cause: ~32 regs/thread for cb[16] limits active warps to ~5/SM.
// Fix: move codebook to shared memory → ~8 regs/thread → 48 warps/SM
// → 12 warps/scheduler → full latency hiding.
//
// Block = WARPS_PER_BLOCK warps × 32 lanes
// Each warp → 1 output C[m, n_warp]
// Codebooks for all warps in the block → shared memory pool
// s_cb[WARPS_PER_BLOCK][LUT_SIZE] (e.g. 8 × 16 = 512 bytes)
//
// B_packed is repacked [K_groups, N, 32] flattened (same as v4opt).

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

template <int BITS>
__global__ void xfp_gemm_v8_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint32_t* __restrict__ B_packed,  // repacked flat
    const half* __restrict__ codebook,  // [N, 2^BITS] fp16 for precision
    __nv_bfloat16* __restrict__ C,               // [M, N]
    int M, int N, int K, int K_packed)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int m = blockIdx.y;

    if (m >= M) return;

    __shared__ float s_cb[WARPS_PER_BLOCK * LUT_SIZE];

    if (n < N && lane < LUT_SIZE) {
        s_cb[warp_id * LUT_SIZE + lane] =
            __half2float(codebook[n * LUT_SIZE + lane]);
    }
    __syncthreads();

    if (n >= N) return;

    const float* my_cb = s_cb + warp_id * LUT_SIZE;

    const __nv_bfloat16* A_row = A + m * K;
    float acc = 0.0f;

    // Repack addressing (same as v4opt)
    int n_offset = n * WARP_SIZE + lane;
    int K_packed_safe = K / VALS_PER_WORD;

    // Prefetch first packed word
    int kw = lane;
    int g = 0;
    uint32_t buf_cur = (kw < K_packed)
        ? B_packed[g * N * WARP_SIZE + n_offset] : 0u;

    // Main loop — no bounds check on slots
    for (; kw < K_packed_safe; kw += WARP_SIZE, g++) {
        uint32_t buf_next = 0u;
        int kw_next = kw + WARP_SIZE;
        if (kw_next < K_packed)
            buf_next = B_packed[(g + 1) * N * WARP_SIZE + n_offset];

        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot += 2) {
            __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(A_row + k_base + slot);
            float a0 = __bfloat162float(__low2bfloat16(a2));
            float a1 = __bfloat162float(__high2bfloat16(a2));

            int idx0 = (int)((buf_cur >> (slot * BITS)) & MASK);
            int idx1 = (int)((buf_cur >> ((slot + 1) * BITS)) & MASK);

            // SMEM codebook read — ~28 cycle latency, hidden by 12 warps/scheduler
            acc = fmaf(my_cb[idx0], a0, acc);
            acc = fmaf(my_cb[idx1], a1, acc);
        }

        buf_cur = buf_next;
    }

    // Tail
    for (; kw < K_packed; kw += WARP_SIZE, g++) {
        uint32_t packed = B_packed[g * N * WARP_SIZE + n_offset];
        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot++) {
            int k = k_base + slot;
            if (k >= K) break;
            float a = __bfloat162float(A_row[k]);
            int idx = (int)((packed >> (slot * BITS)) & MASK);
            acc = fmaf(my_cb[idx], a, acc);
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        C[m * N + n] = __float2bfloat16(acc);
    }
}

template <int BITS>
static void launch_v8(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C, int K, int N, int K_packed)
{
    int M = A.size(0);

    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
        M
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_v8_kernel<BITS><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(codebook.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
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
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "xfp_gemm: A must be bfloat16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32,
                "xfp_gemm: B_packed must be int32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16,
                "xfp_gemm: codebook must be float16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "xfp_gemm: C must be bfloat16");
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

    if (bits == 2) launch_v8<2>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 3) launch_v8<3>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 4) launch_v8<4>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else TORCH_CHECK(false, "xfp_gemm: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP v8: SMEM codebook pool, multi-warp block, max occupancy");
}
