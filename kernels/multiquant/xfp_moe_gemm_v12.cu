// SPDX-License-Identifier: Apache-2.0
// XFP MoE GEMM v12 — A-row SMEM cache (static, rigid K_SMEM_MAX=4096).
//
// Identical algorithm to MoE v11, but for each (token, expert) block we
// load the block-uniform A-row once into __shared__ s_A[4096] and all 8
// warps share it. Covers all Qwen/GLM MoE shapes (K=2048) with 2×
// headroom.
//
// Correctness invariant: output bitwise identical to MoE v11.

#define XFP_CORE_USE_SMEM_A
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "xfp_gemm_core.cuh"

namespace multiquant {

template <int BITS>
static void launch_moe(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor topk_weights,
    int M, int N, int K, int K_packed,
    int top_k, int flat_per_expert, int num_valid_tokens)
{
    int num_token_blocks = static_cast<int>(sorted_token_ids.size(0));

    dim3 block(XFP_BLOCK_SIZE);
    dim3 grid(
        (N + XFP_WARPS_PER_BLOCK - 1) / XFP_WARPS_PER_BLOCK,
        num_token_blocks
    );

    const float* tw_ptr = (topk_weights.defined() && topk_weights.numel() > 0)
        ? topk_weights.data_ptr<float>() : nullptr;

    MoEPolicy::Params params{
        sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        tw_ptr,
        M, top_k, flat_per_expert, num_valid_tokens
    };

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_templated_kernel<BITS, MoEPolicy><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(codebook.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        N, K, K_packed, params);
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
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32, "B_packed must be int32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16, "codebook must be fp16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bf16");
    TORCH_CHECK(bits >= 2 && bits <= 4, "bits must be 2/3/4");
    TORCH_CHECK(K % 2 == 0,
                "xfp_moe_gemm v12: K must be even, got K=", K);
    TORCH_CHECK(K <= MoEPolicy::K_SMEM_MAX,
                "xfp_moe_gemm v12: K=", K, " exceeds K_SMEM_MAX=",
                MoEPolicy::K_SMEM_MAX,
                " — use XFP_MOE_KERNEL=v11 for this shape");

    int M = static_cast<int>(A.size(0));
    int vals_per_word = (bits == 2) ? 16 : (bits == 3) ? 10 : 8;
    int K_packed = (static_cast<int>(K) + vals_per_word - 1) / vals_per_word;

    if (bits == 2)      launch_moe<2>(A, B_packed, codebook, C, sorted_token_ids, expert_ids, topk_weights,
                                      M, static_cast<int>(N), static_cast<int>(K), K_packed,
                                      static_cast<int>(top_k), static_cast<int>(flat_per_expert),
                                      static_cast<int>(num_valid_tokens));
    else if (bits == 3) launch_moe<3>(A, B_packed, codebook, C, sorted_token_ids, expert_ids, topk_weights,
                                      M, static_cast<int>(N), static_cast<int>(K), K_packed,
                                      static_cast<int>(top_k), static_cast<int>(flat_per_expert),
                                      static_cast<int>(num_valid_tokens));
    else if (bits == 4) launch_moe<4>(A, B_packed, codebook, C, sorted_token_ids, expert_ids, topk_weights,
                                      M, static_cast<int>(N), static_cast<int>(K), K_packed,
                                      static_cast<int>(top_k), static_cast<int>(flat_per_expert),
                                      static_cast<int>(num_valid_tokens));
    else TORCH_CHECK(false, "xfp_moe_gemm v12: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_moe_gemm", &multiquant::xfp_moe_gemm,
          "XFP MoE v12: static SMEM A-row cache");
}
