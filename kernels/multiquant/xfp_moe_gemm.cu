// SPDX-License-Identifier: Apache-2.0
// XFP Fused MoE GEMM — single kernel launch for all experts.
//
// Based on xfp_gemm_v8 (SMEM codebook pool), extended with:
// - Expert routing via sorted_token_ids / expert_ids (Marlin pattern)
// - Per-expert weight and codebook indexing
// - topk_weights multiplication
//
// Grid.x = ceil(N / WARPS_PER_BLOCK)  (output columns)
// Grid.y = num_token_blocks           (sorted tokens, grouped by expert)
//
// Each block computes one output element C[token_id, n_warp] for one
// token-expert pair. Expert weights and codebook are selected via
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
__global__ void xfp_moe_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,          // [M, K]
    const uint32_t* __restrict__ B_packed,         // [E * flat_per_expert]
    const half* __restrict__ codebook,             // [E * N * LUT_SIZE]
    __nv_bfloat16* __restrict__ C,                 // [num_tokens_padded, N]
    const int32_t* __restrict__ sorted_token_ids,  // [num_tokens_padded]
    const int32_t* __restrict__ expert_ids,        // [num_token_blocks]
    const float* __restrict__ topk_weights,        // [M * top_k] or nullptr
    int M, int N, int K, int K_packed,
    int top_k,
    int flat_per_expert,    // repacked size per expert
    int num_valid_tokens)   // from num_tokens_post_padded
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int n = blockIdx.x * MOE_WARPS_PER_BLOCK + warp_id;
    int token_block = blockIdx.y;

    // Get expert and token for this block
    int expert_id = expert_ids[token_block];
    if (expert_id < 0) return;  // padding block

    int token_id = sorted_token_ids[token_block];
    if (token_id >= num_valid_tokens) return;  // padding token

    // Original token index (before topk expansion)
    int orig_token = token_id / top_k;
    if (orig_token >= M) return;

    // Load this expert's codebook slice into SMEM
    __shared__ float s_cb[MOE_WARPS_PER_BLOCK * LUT_SIZE];

    // Codebook offset: expert_id * N * LUT_SIZE + n * LUT_SIZE
    const half* expert_cb = codebook + expert_id * N * LUT_SIZE;

    if (n < N && lane < LUT_SIZE) {
        s_cb[warp_id * LUT_SIZE + lane] =
            __half2float(expert_cb[n * LUT_SIZE + lane]);
    }
    __syncthreads();

    if (n >= N) return;

    const float* my_cb = s_cb + warp_id * LUT_SIZE;

    // Input activation row
    const __nv_bfloat16* A_row = A + orig_token * K;

    // Weight offset for this expert
    const uint32_t* expert_B = B_packed + expert_id * flat_per_expert;

    float acc = 0.0f;

    // Repack addressing (same as v8)
    int n_offset = n * WARP_SIZE + lane;
    int K_packed_safe = K / VALS_PER_WORD;

    // Prefetch
    int kw = lane;
    int g = 0;
    uint32_t buf_cur = (kw < K_packed)
        ? expert_B[g * N * WARP_SIZE + n_offset] : 0u;

    // Main loop
    for (; kw < K_packed_safe; kw += WARP_SIZE, g++) {
        uint32_t buf_next = 0u;
        int kw_next = kw + WARP_SIZE;
        if (kw_next < K_packed)
            buf_next = expert_B[(g + 1) * N * WARP_SIZE + n_offset];

        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot += 2) {
            __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(
                A_row + k_base + slot);
            float a0 = __bfloat162float(__low2bfloat16(a2));
            float a1 = __bfloat162float(__high2bfloat16(a2));

            int idx0 = (int)((buf_cur >> (slot * BITS)) & MASK);
            int idx1 = (int)((buf_cur >> ((slot + 1) * BITS)) & MASK);

            acc = fmaf(my_cb[idx0], a0, acc);
            acc = fmaf(my_cb[idx1], a1, acc);
        }

        buf_cur = buf_next;
    }

    // Tail
    for (; kw < K_packed; kw += WARP_SIZE, g++) {
        uint32_t packed = expert_B[g * N * WARP_SIZE + n_offset];
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
static void launch_moe(
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

    xfp_moe_gemm_kernel<BITS><<<grid, block, 0, stream>>>(
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
    torch::Tensor A,              // [M, K] bf16
    torch::Tensor B_packed,       // [E * flat_per_expert] int32
    torch::Tensor codebook,       // [E * N * lut_size] fp16 (flattened)
    torch::Tensor C,              // [num_tokens_padded, N] bf16
    torch::Tensor sorted_token_ids,  // [num_tokens_padded] int32
    torch::Tensor expert_ids,     // [num_token_blocks] int32
    torch::Tensor topk_weights,   // [M * top_k] float32
    int64_t bits,
    int64_t K,
    int64_t N,
    int64_t top_k,
    int64_t flat_per_expert,
    int64_t num_valid_tokens)
{
    TORCH_CHECK(A.is_cuda(), "xfp_moe_gemm: A must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "xfp_moe_gemm: A must be bf16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32, "xfp_moe_gemm: B_packed must be int32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16, "xfp_moe_gemm: codebook must be fp16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "xfp_moe_gemm: C must be bf16");
    TORCH_CHECK(bits >= 2 && bits <= 4, "xfp_moe_gemm: bits must be 2,3,4");

    int M = static_cast<int>(A.size(0));
    int vals_per_word = (bits == 2) ? 16 : (bits == 3) ? 10 : 8;
    int K_packed = (static_cast<int>(K) + vals_per_word - 1) / vals_per_word;

    if (bits == 2) launch_moe<2>(A, B_packed, codebook, C, sorted_token_ids, expert_ids, topk_weights, M, static_cast<int>(N), static_cast<int>(K), K_packed, static_cast<int>(top_k), static_cast<int>(flat_per_expert), static_cast<int>(num_valid_tokens));
    else if (bits == 3) launch_moe<3>(A, B_packed, codebook, C, sorted_token_ids, expert_ids, topk_weights, M, static_cast<int>(N), static_cast<int>(K), K_packed, static_cast<int>(top_k), static_cast<int>(flat_per_expert), static_cast<int>(num_valid_tokens));
    else if (bits == 4) launch_moe<4>(A, B_packed, codebook, C, sorted_token_ids, expert_ids, topk_weights, M, static_cast<int>(N), static_cast<int>(K), K_packed, static_cast<int>(top_k), static_cast<int>(flat_per_expert), static_cast<int>(num_valid_tokens));
    else TORCH_CHECK(false, "xfp_moe_gemm: unsupported bits");
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_moe_gemm", &multiquant::xfp_moe_gemm,
          "XFP fused MoE GEMM — single launch, all experts");
}
