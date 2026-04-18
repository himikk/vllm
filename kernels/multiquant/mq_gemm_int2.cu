// SPDX-License-Identifier: Apache-2.0
// MultiQuant Marlin2 — Fused INT2 Dequant + GEMM kernel.
//
// C[M,N] = A[M,K] @ dequant(B_q[K/16,N], scales[K/gs,N], zeros[K/gs,Nz])
//
// INT2 packing: 16 × 2-bit values per int32, LSB-first.
// Tiling: BLOCK_K=32 (2 int32 words), BLOCK_N=128 (threads), M per block.
// Each thread handles 4 N columns, processes BLOCK_K K values per iteration.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

#define MQ2_BLOCK_K 32
#define MQ2_BLOCK_N 32  // threads per block (each handles 4 cols = 128 N)

template <int M_COUNT>
__global__ void mq_gemm_int2_kernel(
    const half* __restrict__ A,          // [M, K]
    const uint32_t* __restrict__ B_q,    // [K/16, N] packed int2
    const half* __restrict__ B_scales,   // [n_groups, N]
    const uint32_t* __restrict__ B_zeros,// [n_groups, N_zp] packed int2
    half* __restrict__ C,                // [M, N]
    int M, int N, int K, int group_size, int N_zp
) {
    int tid = threadIdx.x;
    int n_base = blockIdx.x * MQ2_BLOCK_N * 4;  // 4 cols per thread
    int m_base = blockIdx.y * M_COUNT;
    int k_start = blockIdx.z * MQ2_BLOCK_K;
    int k_end = min(k_start + MQ2_BLOCK_K, K);

    int n = n_base + tid * 4;
    if (n >= N) return;

    // Preload A tile into shared memory
    __shared__ half smem_a[M_COUNT][MQ2_BLOCK_K];
    if (k_start + tid < K && tid < MQ2_BLOCK_K) {
        for (int m = 0; m < M_COUNT; m++) {
            int mi = m_base + m;
            smem_a[m][tid] = (mi < M) ? A[mi * K + k_start + tid] : __float2half(0.0f);
        }
    }
    __syncthreads();

    // Accumulators for 4 N columns × M_COUNT rows
    float acc[M_COUNT][4] = {};

    // Process K in chunks of 16 (one int32 word per column)
    int group = k_start / group_size;
    int next_group = (group + 1) * group_size;

    // Load initial scales and zeros for 4 columns
    half scales[4];
    int zeros[4];
    for (int j = 0; j < 4; j++) {
        int col = n + j;
        if (col < N) {
            scales[j] = B_scales[group * N + col];
            int zp_word_idx = col / 16;
            int zp_bit_off = (col % 16) * 2;
            uint32_t zp_word = B_zeros[group * N_zp + zp_word_idx];
            zeros[j] = (zp_word >> zp_bit_off) & 0x3;
        }
    }

    for (int k = k_start; k < k_end; k += 16) {
        // Check group boundary
        if (k >= next_group && k < K) {
            group = k / group_size;
            next_group = (group + 1) * group_size;
            for (int j = 0; j < 4; j++) {
                int col = n + j;
                if (col < N) {
                    scales[j] = B_scales[group * N + col];
                    int zp_word_idx = col / 16;
                    int zp_bit_off = (col % 16) * 2;
                    uint32_t zp_word = B_zeros[group * N_zp + zp_word_idx];
                    zeros[j] = (zp_word >> zp_bit_off) & 0x3;
                }
            }
        }

        // Load 4 packed int32 words (one per column)
        int packed_row = k / 16;
        uint32_t w[4];
        for (int j = 0; j < 4; j++) {
            int col = n + j;
            w[j] = (col < N && packed_row < (K / 16)) ?
                    B_q[packed_row * N + col] : 0;
        }

        // Dequant 16 values per column and multiply-accumulate
        int k_local = k - k_start;  // offset into smem_a
        int vals_this_iter = min(16, k_end - k);

        for (int i = 0; i < vals_this_iter; i++) {
            for (int j = 0; j < 4; j++) {
                int qval = (w[j] >> (i * 2)) & 0x3;
                float dq = __half2float(scales[j]) * (float)(qval - zeros[j]);
                for (int m = 0; m < M_COUNT; m++) {
                    acc[m][j] += dq * __half2float(smem_a[m][k_local + i]);
                }
            }
        }
    }

    // Write output (atomicAdd for K-split accumulation)
    for (int m = 0; m < M_COUNT; m++) {
        int mi = m_base + m;
        if (mi >= M) break;
        for (int j = 0; j < 4; j++) {
            int col = n + j;
            if (col < N) {
                atomicAdd(reinterpret_cast<half*>(&C[mi * N + col]),
                          __float2half(acc[m][j]));
            }
        }
    }
}

void mq_gemm_int2(
    torch::Tensor A,       // [M, K] float16
    torch::Tensor B_q,     // [K/16, N] int32
    torch::Tensor scales,  // [n_groups, N] float16
    torch::Tensor zeros,   // [n_groups, N_zp] int32
    torch::Tensor C,       // [M, N] float16 (pre-allocated, zeroed)
    int group_size
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B_q.size(1);
    int N_zp = zeros.size(1);

    dim3 block(MQ2_BLOCK_N);
    dim3 grid(
        (N + MQ2_BLOCK_N * 4 - 1) / (MQ2_BLOCK_N * 4),
        (M + 3) / 4,  // M_COUNT=4
        (K + MQ2_BLOCK_K - 1) / MQ2_BLOCK_K
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mq_gemm_int2_kernel<4><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(A.data_ptr()),
        reinterpret_cast<const uint32_t*>(B_q.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr()),
        reinterpret_cast<const uint32_t*>(zeros.data_ptr<int32_t>()),
        reinterpret_cast<half*>(C.data_ptr()),
        M, N, K, group_size, N_zp
    );
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mq_gemm_int2", &multiquant::mq_gemm_int2,
          "MultiQuant Marlin2: Fused INT2 dequant+GEMM (A @ dequant(B))");
}
