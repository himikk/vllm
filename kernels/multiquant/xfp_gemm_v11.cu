// SPDX-License-Identifier: Apache-2.0
// XFP Linear GEMM v11 — thin wrapper around xfp_gemm_core template.
//
// Same algorithm as v10 (SHFL codebook lookup + unified K-loop). The
// inner loop now lives in xfp_gemm_core.cuh and is shared with the MoE
// variant. Future optimisations (cp.async, MMA, LOP3 dequant) go into
// the core header and are picked up by both Linear and MoE automatically.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "xfp_gemm_core.cuh"

namespace multiquant {

template <int BITS>
static void launch_linear(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C, int K, int N, int K_packed)
{
    int M = static_cast<int>(A.size(0));

    dim3 block(XFP_BLOCK_SIZE);
    dim3 grid(
        (N + XFP_WARPS_PER_BLOCK - 1) / XFP_WARPS_PER_BLOCK,
        M
    );

    LinearPolicy::Params params{M};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_templated_kernel<BITS, LinearPolicy><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(codebook.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        N, K, K_packed, params);
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

    if (bits == 2)      launch_linear<2>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 3) launch_linear<3>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 4) launch_linear<4>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else TORCH_CHECK(false, "xfp_gemm: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP v11: Linear wrapper around shared core template");
}
