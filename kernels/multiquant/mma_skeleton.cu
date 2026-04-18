// SPDX-License-Identifier: Apache-2.0
// MMA skeleton — minimal mma.m16n8k16.bf16.bf16.f32 test on SM121.
//
// Purpose: verify Tensor-Core MMA + ldmatrix layout is correct before
// integrating XFP-decode. One block computes C[16, 8] = A[16, K] @ B[K, 8]
// for small K (multiple of 16) using Tensor Cores.
//
// Thread layout for mma.m16n8k16:
//   A: [16, 16] bf16 — each thread holds 8 elements (4 fragments × 2 bf16)
//   B: [16, 8]  bf16 — each thread holds 4 elements (2 fragments × 2 bf16)
//   C: [16, 8]  f32  — each thread holds 4 fp32 accumulators
// PTX layout (from NVIDIA docs, m16n8k16):
//   A fragment organization: group of 4 2-tiles of 8×2 bf16 each,
//     lane i ∈ [0,3] row, i/4 col, holds 2 bf16 per fragment.
//   B fragment organization: 2 tiles of 8×4 bf16, lane i row=i%8, col=i/8.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

namespace mma_skeleton {

__device__ __forceinline__ uint32_t __bfloat1622uint(__nv_bfloat162 v) {
    uint32_t r;
    asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(*(uint32_t*)&v));
    return r;
}

#define WARP_SIZE 32

// K-tile = 16 (one MMA K-step).
// Kernel computes one tile C[16, 8] = A[16, K] @ B[K, 8].
// 1 warp per output tile. Grid = (1, 1), block = (32).
__global__ void mma_skeleton_kernel(
    const __nv_bfloat16* __restrict__ A,  // [16, K]
    const __nv_bfloat16* __restrict__ B,  // [K, 8]
    float* __restrict__ C,                 // [16, 8]
    int K)
{
    int lane = threadIdx.x;

    // Accumulator (4 fp32 / lane for m16n8 output)
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // K loop
    for (int k = 0; k < K; k += 16) {
        // --- Load A tile [16, 16] into lane fragments ---
        // mma m16n8k16 A layout: lane i holds
        //   (row=i/4, col=(i%4)*2)   and (row+8, col)       fragment 0,1
        //   (row=i/4, col=(i%4)*2+8) and (row+8, col+8)     fragment 2,3
        // 8 bf16 per lane. Pack into uint32[4] for ldmatrix-compat later.
        uint32_t a_frag[4];
        {
            int row_low = lane / 4;
            int row_high = row_low + 8;
            int col_base = (lane % 4) * 2;
            __nv_bfloat16 a0 = A[(row_low) * K + k + col_base + 0];
            __nv_bfloat16 a1 = A[(row_low) * K + k + col_base + 1];
            __nv_bfloat16 a2 = A[(row_high) * K + k + col_base + 0];
            __nv_bfloat16 a3 = A[(row_high) * K + k + col_base + 1];
            __nv_bfloat16 a4 = A[(row_low) * K + k + col_base + 8];
            __nv_bfloat16 a5 = A[(row_low) * K + k + col_base + 8 + 1];
            __nv_bfloat16 a6 = A[(row_high) * K + k + col_base + 8];
            __nv_bfloat16 a7 = A[(row_high) * K + k + col_base + 8 + 1];
            a_frag[0] = __bfloat1622uint(__nv_bfloat162(a0, a1));
            a_frag[1] = __bfloat1622uint(__nv_bfloat162(a2, a3));
            a_frag[2] = __bfloat1622uint(__nv_bfloat162(a4, a5));
            a_frag[3] = __bfloat1622uint(__nv_bfloat162(a6, a7));
        }

        // --- Load B tile [16, 8] into lane fragments ---
        // mma m16n8k16 B layout (K=16 rows, N=8 cols, 4 threads per col):
        //   col = lane / 4    (range 0..7)
        //   row = (lane%4)*2  (range 0, 2, 4, 6)
        //   frag 0: (row, col), (row+1, col)     — K rows 0..7
        //   frag 1: (row+8, col), (row+9, col)   — K rows 8..15
        uint32_t b_frag[2];
        {
            int col = lane / 4;
            int row_base = (lane % 4) * 2;
            __nv_bfloat16 b0 = B[(k + row_base + 0) * 8 + col];
            __nv_bfloat16 b1 = B[(k + row_base + 1) * 8 + col];
            __nv_bfloat16 b2 = B[(k + row_base + 8) * 8 + col];
            __nv_bfloat16 b3 = B[(k + row_base + 9) * 8 + col];
            b_frag[0] = __bfloat1622uint(__nv_bfloat162(b0, b1));
            b_frag[1] = __bfloat1622uint(__nv_bfloat162(b2, b3));
        }

        // --- MMA: C += A @ B ---
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1])
        );
    }

    // --- Store C [16, 8] ---
    // mma m16n8k16 output layout: lane i holds
    //   c[0] at (row=i/4,   col=(i%4)*2)
    //   c[1] at (row=i/4,   col=(i%4)*2+1)
    //   c[2] at (row=i/4+8, col=(i%4)*2)
    //   c[3] at (row=i/4+8, col=(i%4)*2+1)
    int row_low = lane / 4;
    int col_base = (lane % 4) * 2;
    C[(row_low) * 8 + col_base + 0] = c[0];
    C[(row_low) * 8 + col_base + 1] = c[1];
    C[(row_low + 8) * 8 + col_base + 0] = c[2];
    C[(row_low + 8) * 8 + col_base + 1] = c[3];
}

void mma_skeleton(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t K)
{
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bf16");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be f32");
    TORCH_CHECK(A.sizes() == torch::IntArrayRef({16, K}), "A shape mismatch");
    TORCH_CHECK(B.sizes() == torch::IntArrayRef({K, 8}), "B shape mismatch");
    TORCH_CHECK(C.sizes() == torch::IntArrayRef({16, 8}), "C shape mismatch");
    TORCH_CHECK(K % 16 == 0, "K must be multiple of 16");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mma_skeleton_kernel<<<1, WARP_SIZE, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        C.data_ptr<float>(),
        static_cast<int>(K));
}

}  // namespace mma_skeleton

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma_skeleton", &mma_skeleton::mma_skeleton,
          "mma.m16n8k16.bf16.bf16.f32 skeleton for verification");
}
