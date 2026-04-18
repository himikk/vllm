// SPDX-License-Identifier: Apache-2.0
// XFP v10 — SHFL.IDX codebook lookup (register shuffle, 0 SMEM).
//
// Game-changer vs v8: codebook lookup via __shfl_sync instead of SMEM.
//   v8:  my_cb[idx] from shared memory → ~28 cycle latency per lookup
//   v10: __shfl_sync(mask, my_cb_val, idx) → 1 cycle, register-to-register
//
// Each lane in the warp holds ONE codebook entry as a float register.
// XFP4: 16 entries → lanes 0-15 hold codebook[n][0..15].
// XFP3: 8 entries → lanes 0-7.  XFP2: 4 entries → lanes 0-3.
// Lookup = warp shuffle with the packed index as source lane.
//
// No __shared__ memory needed for codebook → no __syncthreads,
// no bank conflicts, no SMEM pressure. Pure register path.
//
// Block = WARPS_PER_BLOCK warps × 32 lanes (same as v8)
// Each warp → 1 output C[m, n_warp]
// B_packed is repacked [K_groups, N, 32] flattened (same as v8).

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

template <int BITS>
__global__ void xfp_gemm_v10_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint32_t* __restrict__ B_packed,
    const half* __restrict__ codebook,  // [N, 2^BITS]
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, int K_packed)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int m = blockIdx.y;

    if (m >= M || n >= N) return;

    // ── v10: codebook in register, lookup via warp shuffle ──
    // Each lane < LUT_SIZE loads one codebook entry for this warp's
    // output column n. Lanes >= LUT_SIZE hold 0 (unused but harmless).
    float my_cb_val = (lane < LUT_SIZE)
        ? __half2float(codebook[n * LUT_SIZE + lane])
        : 0.0f;
    // No __syncthreads needed — purely warp-local.

    const __nv_bfloat16* A_row = A + m * K;
    float acc = 0.0f;

    int n_offset = n * WARP_SIZE + lane;

    // Single unified K-loop. Every lane runs the SAME number of iterations
    // (n_groups), so __shfl_sync(0xffffffff) always has all 32 participants.
    // Lanes whose kw >= K_packed (or k >= K) load zeros and skip the FMA.
    // This avoids the warp-synchronous SHFL hazard that broke the old
    // safe/tail split (lanes diverged in loop entry → SHFL undefined).
    int n_groups = (K_packed + WARP_SIZE - 1) / WARP_SIZE;

    for (int gi = 0; gi < n_groups; gi++) {
        int kw = lane + gi * WARP_SIZE;
        uint32_t packed = (kw < K_packed)
            ? B_packed[gi * N * WARP_SIZE + n_offset] : 0u;
        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot++) {
            int idx = (int)((packed >> (slot * BITS)) & MASK);
            // Warp-synchronous: all 32 lanes MUST be at this instruction.
            float w = __shfl_sync(0xffffffff, my_cb_val, idx);
            int k = k_base + slot;
            if (k < K && kw < K_packed) {
                float a = __bfloat162float(A_row[k]);
                acc = fmaf(w, a, acc);
            }
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
static void launch_v10(
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

    xfp_gemm_v10_kernel<BITS><<<grid, block, 0, stream>>>(
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
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32, "B_packed must be int32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16, "codebook must be float16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");

    int N = static_cast<int>(codebook.size(0));
    int vals_per_word = (bits == 2) ? 16 : (bits == 3) ? 10 : 8;
    int K_packed = (static_cast<int>(K) + vals_per_word - 1) / vals_per_word;

    if (bits == 2) launch_v10<2>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 3) launch_v10<3>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 4) launch_v10<4>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else TORCH_CHECK(false, "xfp_gemm: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP v10: SHFL.IDX codebook lookup, zero SMEM");
}
