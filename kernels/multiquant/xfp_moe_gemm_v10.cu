// SPDX-License-Identifier: Apache-2.0
// XFP Fused MoE GEMM v10 — SHFL.IDX codebook lookup.
//
// Same game-changer as xfp_gemm_v10: codebook lookup via __shfl_sync
// instead of SMEM. Each lane holds one codebook entry as a register;
// lookup is a 1-cycle warp shuffle instead of a ~28-cycle SMEM load.
//
// Unified K-loop: ALL 32 lanes run the same number of iterations, so
// __shfl_sync(0xffffffff) always has all 32 participants. Lanes whose
// kw >= K_packed load zeros and skip the FMA. This avoids the warp-
// synchronous SHFL hazard (diverged lanes → undefined shuffle).
//
// Same grid/block layout as xfp_moe_gemm:
//   Grid.x = ceil(N / WARPS_PER_BLOCK)  (output columns)
//   Grid.y = num_token_blocks           (sorted tokens, grouped by expert)
// Each block computes one output element C[token_id, n_warp] for one
// token-expert pair. Expert weights and codebook selected via
// expert_ids[blockIdx.y].

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

#define WARP_SIZE 32
#define MOE_WARPS_PER_BLOCK 8
#define MOE_BLOCK_SIZE (WARP_SIZE * MOE_WARPS_PER_BLOCK)

template <int BITS>
__global__ void xfp_moe_gemm_v10_kernel(
    const __nv_bfloat16* __restrict__ A,          // [M, K]
    const uint32_t* __restrict__ B_packed,         // [E * flat_per_expert]
    const half* __restrict__ codebook,             // [E * N * LUT_SIZE]
    __nv_bfloat16* __restrict__ C,                 // [num_tokens_padded, N]
    const int32_t* __restrict__ sorted_token_ids,  // [num_tokens_padded]
    const int32_t* __restrict__ expert_ids,        // [num_token_blocks]
    const float* __restrict__ topk_weights,        // [M * top_k] or nullptr
    int M, int N, int K, int K_packed,
    int top_k,
    int flat_per_expert,
    int num_valid_tokens)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int n = blockIdx.x * MOE_WARPS_PER_BLOCK + warp_id;
    int token_block = blockIdx.y;

    int expert_id = expert_ids[token_block];
    if (expert_id < 0) return;  // padding block

    int token_id = sorted_token_ids[token_block];
    if (token_id >= num_valid_tokens) return;  // padding token

    int orig_token = token_id / top_k;
    if (orig_token >= M || n >= N) return;

    // ── v10: codebook in register, lookup via warp shuffle ──
    // Per-expert codebook slice: codebook[expert_id * N * LUT_SIZE + n * LUT_SIZE + lane]
    const half* expert_cb = codebook + expert_id * N * LUT_SIZE;
    float my_cb_val = (lane < LUT_SIZE)
        ? __half2float(expert_cb[n * LUT_SIZE + lane])
        : 0.0f;
    // No __syncthreads — purely warp-local.

    const __nv_bfloat16* A_row = A + orig_token * K;
    const uint32_t* expert_B = B_packed + expert_id * flat_per_expert;

    float acc = 0.0f;
    int n_offset = n * WARP_SIZE + lane;

    // Unified K-loop: every lane iterates the same number of times
    // so __shfl_sync always has all 32 participants.
    int n_groups = (K_packed + WARP_SIZE - 1) / WARP_SIZE;

    for (int gi = 0; gi < n_groups; gi++) {
        int kw = lane + gi * WARP_SIZE;
        uint32_t packed = (kw < K_packed)
            ? expert_B[gi * N * WARP_SIZE + n_offset] : 0u;
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

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Apply topk weight and write
    if (lane == 0) {
        if (topk_weights != nullptr) {
            acc *= topk_weights[token_id];
        }
        C[token_id * N + n] = __float2bfloat16(acc);
    }
}

template <int BITS>
static void launch_moe_v10(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor topk_weights,
    int M, int N, int K, int K_packed,
    int top_k, int flat_per_expert, int num_valid_tokens)
{
    int num_token_blocks = sorted_token_ids.size(0);

    dim3 block(MOE_BLOCK_SIZE);
    dim3 grid(
        (N + MOE_WARPS_PER_BLOCK - 1) / MOE_WARPS_PER_BLOCK,
        num_token_blocks
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const float* tw_ptr = topk_weights.defined() && topk_weights.numel() > 0
        ? topk_weights.data_ptr<float>() : nullptr;

    xfp_moe_gemm_v10_kernel<BITS><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(codebook.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        tw_ptr,
        M, N, K, K_packed,
        top_k, flat_per_expert, num_valid_tokens);
}

void xfp_moe_gemm(
    torch::Tensor A,
    torch::Tensor B_packed,
    torch::Tensor codebook,
    torch::Tensor C,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor topk_weights,
    int64_t bits,
    int64_t K,
    int64_t N,
    int64_t top_k,
    int64_t flat_per_expert,
    int64_t num_valid_tokens)
{
    TORCH_CHECK(A.is_cuda(), "xfp_moe_gemm: A must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32, "B_packed must be int32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16, "codebook must be fp16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bf16");
    TORCH_CHECK(bits >= 2 && bits <= 4, "bits must be 2,3,4");

    int M = static_cast<int>(A.size(0));
    int vals_per_word = (bits == 2) ? 16 : (bits == 3) ? 10 : 8;
    int K_packed = (static_cast<int>(K) + vals_per_word - 1) / vals_per_word;

    if (bits == 2) launch_moe_v10<2>(A, B_packed, codebook, C, sorted_token_ids, expert_ids, topk_weights, M, static_cast<int>(N), static_cast<int>(K), K_packed, static_cast<int>(top_k), static_cast<int>(flat_per_expert), static_cast<int>(num_valid_tokens));
    else if (bits == 3) launch_moe_v10<3>(A, B_packed, codebook, C, sorted_token_ids, expert_ids, topk_weights, M, static_cast<int>(N), static_cast<int>(K), K_packed, static_cast<int>(top_k), static_cast<int>(flat_per_expert), static_cast<int>(num_valid_tokens));
    else if (bits == 4) launch_moe_v10<4>(A, B_packed, codebook, C, sorted_token_ids, expert_ids, topk_weights, M, static_cast<int>(N), static_cast<int>(K), K_packed, static_cast<int>(top_k), static_cast<int>(flat_per_expert), static_cast<int>(num_valid_tokens));
    else TORCH_CHECK(false, "xfp_moe_gemm: unsupported bits");
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_moe_gemm", &multiquant::xfp_moe_gemm,
          "XFP MoE v10: SHFL.IDX codebook lookup, zero SMEM, unified K-loop");
}
