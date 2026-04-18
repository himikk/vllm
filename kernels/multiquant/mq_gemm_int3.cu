// SPDX-License-Identifier: Apache-2.0
// MultiQuant Marlin3 — Fused INT3 Dequant + GEMM kernel.
//
// C[M,N] = A[M,K] @ dequant(B_q[ceil(K*3/32),N], scales, zeros)
//
// INT3 bit-stream: 3 int32 words encode 32 × 3-bit values.
// Boundary crossings at positions 10 and 21 handled explicitly.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

#define MQ3_BLOCK_K 32
#define MQ3_BLOCK_N 32

// Extract one 3-bit value from 96-bit window (3 × uint32)
__device__ __forceinline__ int extract_3bit(uint32_t w0, uint32_t w1, uint32_t w2, int i) {
    if (i == 10) return (w0 >> 30) | ((w1 & 0x1) << 2);
    if (i == 21) return ((w1 >> 31) & 0x1) | ((w2 & 0x3) << 1);
    if (i < 10)  return (w0 >> (i * 3)) & 0x7;
    if (i < 21)  return (w1 >> ((i - 11) * 3 + 1)) & 0x7;
    return (w2 >> ((i - 22) * 3 + 2)) & 0x7;
}

// Extract 3-bit zero point from packed zeros row
__device__ __forceinline__ int extract_3bit_zp(
    const uint32_t* zeros_row, int col, int N_zp
) {
    // zeros packed as bit-stream: 3 int32 per 32 zero-point values
    int block = col / 32;
    int pos = col % 32;
    int base = block * 3;
    uint32_t z0 = (base < N_zp) ? zeros_row[base] : 0;
    uint32_t z1 = (base + 1 < N_zp) ? zeros_row[base + 1] : 0;
    uint32_t z2 = (base + 2 < N_zp) ? zeros_row[base + 2] : 0;
    return extract_3bit(z0, z1, z2, pos);
}

template <int M_COUNT>
__global__ void mq_gemm_int3_kernel(
    const half* __restrict__ A,
    const uint32_t* __restrict__ B_q,     // [K_packed, N]
    const half* __restrict__ B_scales,    // [n_groups, N]
    const uint32_t* __restrict__ B_zeros, // [n_groups, N_zp]
    half* __restrict__ C,
    int M, int N, int K, int K_packed,
    int group_size, int N_zp
) {
    int tid = threadIdx.x;
    int n_base = blockIdx.x * MQ3_BLOCK_N * 4;
    int m_base = blockIdx.y * M_COUNT;
    int k_block = blockIdx.z;  // which 32-value K block
    int k_start = k_block * 32;
    int k_end = min(k_start + 32, K);

    int n = n_base + tid * 4;
    if (n >= N) return;

    // Preload A tile
    __shared__ half smem_a[M_COUNT][MQ3_BLOCK_K];
    if (k_start + tid < K && tid < MQ3_BLOCK_K) {
        for (int m = 0; m < M_COUNT; m++) {
            int mi = m_base + m;
            smem_a[m][tid] = (mi < M) ? A[mi * K + k_start + tid] : __float2half(0.0f);
        }
    }
    __syncthreads();

    float acc[M_COUNT][4] = {};

    int group = k_start / group_size;

    // Load scales and zeros for 4 columns
    half scales[4];
    int zeros[4];
    for (int j = 0; j < 4; j++) {
        int col = n + j;
        if (col < N) {
            scales[j] = B_scales[group * N + col];
            zeros[j] = extract_3bit_zp(B_zeros + group * N_zp, col, N_zp);
        }
    }

    // Load 3 int32 words per column (96 bits = 32 × 3-bit values)
    int packed_base = k_block * 3;  // 3 rows in packed per 32 K values

    for (int j = 0; j < 4; j++) {
        int col = n + j;
        if (col >= N) continue;

        uint32_t w0 = (packed_base < K_packed) ? B_q[packed_base * N + col] : 0;
        uint32_t w1 = (packed_base + 1 < K_packed) ? B_q[(packed_base + 1) * N + col] : 0;
        uint32_t w2 = (packed_base + 2 < K_packed) ? B_q[(packed_base + 2) * N + col] : 0;

        int num_vals = min(32, k_end - k_start);
        for (int i = 0; i < num_vals; i++) {
            int qval = extract_3bit(w0, w1, w2, i);

            // Check group boundary
            int k = k_start + i;
            int g = k / group_size;
            if (g != group) {
                group = g;
                scales[j] = B_scales[group * N + col];
                zeros[j] = extract_3bit_zp(B_zeros + group * N_zp, col, N_zp);
            }

            float dq = __half2float(scales[j]) * (float)(qval - zeros[j]);
            for (int m = 0; m < M_COUNT; m++) {
                acc[m][j] += dq * __half2float(smem_a[m][i]);
            }
        }
    }

    // Write with atomicAdd (K-split)
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

void mq_gemm_int3(
    torch::Tensor A,       // [M, K] float16
    torch::Tensor B_q,     // [K_packed, N] int32
    torch::Tensor scales,  // [n_groups, N] float16
    torch::Tensor zeros,   // [n_groups, N_zp] int32
    torch::Tensor C,       // [M, N] float16 (pre-allocated, zeroed)
    int K,                 // actual K dimension
    int group_size
) {
    int M = A.size(0);
    int N = B_q.size(1);
    int K_packed = B_q.size(0);
    int N_zp = zeros.size(1);

    dim3 block(MQ3_BLOCK_N);
    dim3 grid(
        (N + MQ3_BLOCK_N * 4 - 1) / (MQ3_BLOCK_N * 4),
        (M + 3) / 4,
        (K + 31) / 32
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mq_gemm_int3_kernel<4><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(A.data_ptr()),
        reinterpret_cast<const uint32_t*>(B_q.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr()),
        reinterpret_cast<const uint32_t*>(zeros.data_ptr<int32_t>()),
        reinterpret_cast<half*>(C.data_ptr()),
        M, N, K, K_packed, group_size, N_zp
    );
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mq_gemm_int3", &multiquant::mq_gemm_int3,
          "MultiQuant Marlin3: Fused INT3 dequant+GEMM (A @ dequant(B))");
}
