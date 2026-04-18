// SPDX-License-Identifier: Apache-2.0
// TurboQuant Fused Compressed Decode Attention — Full V Decompression
//
// Single kernel: compressed K → scores → softmax → compressed V → FULL decompress → accumulate
// V reconstruction: v_recon[d] = vn * (sum_j Pi[d,j]*centroids[idx[j]] + corr*rn*(sum_j S[d,j]*sign[j]))
//
// Grid: (num_q_tokens * num_q_heads)
// Block: (HEAD_DIM) — one thread per dimension
//
// Score: thread 0 serial (K score)
// V decompress: all threads parallel (each thread computes one output dim via GEMV)

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace turboquant {

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

// Full V decompression decode kernel
// Output: fully reconstructed V vectors (no post-loop GEMV needed)
template <int HEAD_DIM, int MSE_BITS, int N_CENTROIDS>
__global__ void tq_fused_decode_kernel(
    // Precomputed query projections
    const float* __restrict__ q_rot,       // [num_q, num_q_heads, D]
    const float* __restrict__ q_proj,      // [num_q, num_q_heads, D]
    // Compressed KV cache: (num_blocks, 2, block_size, num_kv_heads, packed_size)
    const uint8_t* __restrict__ kv_cache,
    // Reconstruction matrices
    const float* __restrict__ Pi,          // [D, D] rotation matrix
    const float* __restrict__ S,           // [D, D] QJL projection matrix
    const float* __restrict__ centroids,   // [N_CENTROIDS]
    // Block table
    const int* __restrict__ block_table,   // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    // Output: fully reconstructed attention output
    float* __restrict__ output,            // [num_q, num_q_heads, D]
    // Dims
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int packed_size,
    int max_blocks_per_seq,
    float attn_scale,
    float correction_scale,
    // KV cache strides (in bytes, since uint8)
    int stride_block,
    int stride_kv,
    int stride_slot,
    int stride_head
) {
    const int qh_idx = blockIdx.x;
    const int q_token = qh_idx / num_q_heads;
    const int q_head = qh_idx % num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int tid = threadIdx.x;
    constexpr int D = HEAD_DIM;
    constexpr int MSE_BYTES = (D * MSE_BITS + 7) / 8;
    constexpr int QJL_BYTES = (D + 7) / 8;
    constexpr int MASK = (1 << MSE_BITS) - 1;

    int seq_len = seq_lens[q_token];

    // Shared memory
    __shared__ float s_q_rot[D];
    __shared__ float s_q_proj[D];
    __shared__ float s_centroids[N_CENTROIDS];
    __shared__ int s_v_idx[D];       // V indices for full decompression
    __shared__ float s_v_sign[D];    // V signs for full decompression

    int q_base = (q_token * num_q_heads + q_head) * D;
    for (int i = tid; i < D; i += blockDim.x) {
        s_q_rot[i] = q_rot[q_base + i];
        s_q_proj[i] = q_proj[q_base + i];
    }
    if (tid < N_CENTROIDS) s_centroids[tid] = centroids[tid];
    __syncthreads();

    // Online softmax state
    float m_prev = -INFINITY;
    float d_prev = 0.0f;
    float v_acc = 0.0f;  // per-thread output accumulator (one per dim)

    for (int pos = 0; pos < seq_len; pos++) {
        int bi = pos / block_size;
        int bo = pos % block_size;
        int phys_block = block_table[q_token * max_blocks_per_seq + bi];

        // K base address
        int k_base = phys_block * stride_block
                   + 0 * stride_kv
                   + bo * stride_slot
                   + kv_head * stride_head;
        const uint8_t* k_packed = kv_cache + k_base;

        // === K Score (thread 0, serial over D) ===
        __shared__ float s_score;
        if (tid == 0) {
            float term1 = 0.0f;
            if constexpr (MSE_BITS == 1) {
                for (int b = 0; b < MSE_BYTES; b++) {
                    uint8_t byte_val = k_packed[b];
                    for (int k = 0; k < 8 && (b*8+k) < D; k++) {
                        int j = b*8+k;
                        int idx = (byte_val >> k) & 1;
                        term1 += s_q_rot[j] * s_centroids[idx];
                    }
                }
            } else if constexpr (MSE_BITS == 2) {
                for (int b = 0; b < MSE_BYTES; b++) {
                    uint8_t byte_val = k_packed[b];
                    for (int k = 0; k < 4 && (b*4+k) < D; k++) {
                        int j = b*4+k;
                        int idx = (byte_val >> (k*2)) & MASK;
                        term1 += s_q_rot[j] * s_centroids[idx];
                    }
                }
            } else if constexpr (MSE_BITS == 3) {
                for (int j = 0; j < D; j++) {
                    int bit_off = j*3;
                    int byte_idx = bit_off/8;
                    int bit_idx = bit_off%8;
                    int val = (k_packed[byte_idx] >> bit_idx);
                    if (bit_idx > 5 && byte_idx+1 < MSE_BYTES)
                        val |= (k_packed[byte_idx+1] << (8-bit_idx));
                    term1 += s_q_rot[j] * s_centroids[val & MASK];
                }
            }

            float term2 = 0.0f;
            const uint8_t* k_signs = k_packed + MSE_BYTES;
            for (int b = 0; b < QJL_BYTES; b++) {
                uint8_t bv = k_signs[b];
                for (int k = 0; k < 8 && (b*8+k) < D; k++) {
                    int j = b*8+k;
                    float sv = ((bv >> k) & 1) ? 1.0f : -1.0f;
                    term2 += s_q_proj[j] * sv;
                }
            }

            int norm_off = MSE_BYTES + QJL_BYTES;
            uint16_t vn_u16 = k_packed[norm_off] | (k_packed[norm_off+1] << 8);
            uint16_t rn_u16 = k_packed[norm_off+2] | (k_packed[norm_off+3] << 8);
            float vn = __half2float(*reinterpret_cast<const __half*>(&vn_u16));
            float rn = __half2float(*reinterpret_cast<const __half*>(&rn_u16));

            s_score = vn * (term1 + correction_scale * rn * term2) * attn_scale;
        }
        __syncthreads();
        float score = s_score;

        // === Online softmax ===
        float m_new = fmaxf(m_prev, score);
        float exp_prev = expf(m_prev - m_new);
        float exp_cur = expf(score - m_new);
        float d_new = d_prev * exp_prev + exp_cur;

        // Rescale accumulator
        v_acc *= exp_prev;

        // === V: Full Decompression (all threads parallel) ===
        if (tid < D) {
            int v_base = phys_block * stride_block
                       + 1 * stride_kv
                       + bo * stride_slot
                       + kv_head * stride_head;
            const uint8_t* v_packed = kv_cache + v_base;

            // Step 1: Each thread unpacks its own V index + sign → shared memory
            int v_idx;
            if constexpr (MSE_BITS == 1) {
                int b_idx = tid / 8; int k = tid % 8;
                v_idx = (v_packed[b_idx] >> k) & 1;
            } else if constexpr (MSE_BITS == 2) {
                int b_idx = tid / 4; int k = tid % 4;
                v_idx = (v_packed[b_idx] >> (k*2)) & MASK;
            } else {
                int bit_off = tid * MSE_BITS;
                int byte_idx = bit_off / 8;
                int bit_idx = bit_off % 8;
                v_idx = (v_packed[byte_idx] >> bit_idx);
                if (bit_idx + MSE_BITS > 8 && byte_idx+1 < MSE_BYTES)
                    v_idx |= (v_packed[byte_idx+1] << (8-bit_idx));
                v_idx &= MASK;
            }

            int s_byte = MSE_BYTES + tid / 8;
            int s_bit = tid % 8;
            float v_sign = ((v_packed[s_byte] >> s_bit) & 1) ? 1.0f : -1.0f;

            s_v_idx[tid] = v_idx;
            s_v_sign[tid] = v_sign;
        }
        __syncthreads();

        // Step 2: Full V decompression via GEMV
        // v_recon[tid] = vn * (sum_j Pi[tid*D+j] * centroids[idx[j]]
        //                    + corr * rn * sum_j S[tid*D+j] * sign[j])
        if (tid < D) {
            const uint8_t* v_packed = kv_cache
                + phys_block * stride_block
                + 1 * stride_kv
                + bo * stride_slot
                + kv_head * stride_head;

            int vn_off = MSE_BYTES + QJL_BYTES;
            uint16_t vvn_u16 = v_packed[vn_off] | (v_packed[vn_off+1] << 8);
            uint16_t vrn_u16 = v_packed[vn_off+2] | (v_packed[vn_off+3] << 8);
            float v_vn = __half2float(*reinterpret_cast<const __half*>(&vvn_u16));
            float v_rn = __half2float(*reinterpret_cast<const __half*>(&vrn_u16));

            // GEMV: v_mse[tid] = sum_j Pi[j, tid] * centroids[idx[j]]
            //   Python: centroids[idx] @ Pi → v[tid] = sum_j c[idx[j]] * Pi[j, tid]
            //   Pi[j, tid] = Pi[j*D + tid] in row-major
            // GEMV: v_qjl[tid] = sum_j S[j, tid] * sign[j]
            //   Python: signs @ S → v[tid] = sum_j sign[j] * S[j, tid]
            float v_mse = 0.0f;
            float v_qjl = 0.0f;
            for (int j = 0; j < D; j++) {
                v_mse += Pi[j * D + tid] * s_centroids[s_v_idx[j]];
                v_qjl += S[j * D + tid] * s_v_sign[j];
            }

            float v_recon = v_vn * (v_mse + correction_scale * v_rn * v_qjl);
            v_acc += exp_cur * v_recon;
        }

        m_prev = m_new;
        d_prev = d_new;
        __syncthreads();  // ensure s_v_idx/s_v_sign not overwritten before all threads done
    }

    // Normalize and store
    if (tid < D && d_prev > 0.0f) {
        output[q_base + tid] = v_acc / d_prev;
    }
}


void tq_fused_decode_attention(
    torch::Tensor q_rot,
    torch::Tensor q_proj,
    torch::Tensor kv_cache,
    torch::Tensor Pi,
    torch::Tensor S,
    torch::Tensor centroids,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor output,
    int head_dim,
    int mse_bits,
    int n_centroids,
    float attn_scale,
    // KV cache strides
    int stride_block,
    int stride_kv,
    int stride_slot,
    int stride_head
) {
    int num_q = q_rot.size(0);
    int num_q_heads = q_rot.size(1);
    int num_kv_heads = kv_cache.size(3);
    int block_size = kv_cache.size(2);
    int packed_size = kv_cache.size(4);
    int max_blocks = block_table.size(1);
    float correction = sqrtf(M_PI_2) / static_cast<float>(head_dim);

    dim3 grid(num_q * num_q_heads);
    int threads = ((head_dim + 31) / 32) * 32;
    dim3 block(threads);

    #define LAUNCH(HD, MB, NC) \
        tq_fused_decode_kernel<HD, MB, NC><<<grid, block>>>( \
            q_rot.data_ptr<float>(), q_proj.data_ptr<float>(), \
            kv_cache.data_ptr<uint8_t>(), \
            Pi.data_ptr<float>(), S.data_ptr<float>(), \
            centroids.data_ptr<float>(), \
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(), \
            output.data_ptr<float>(), \
            num_q_heads, num_kv_heads, block_size, packed_size, \
            max_blocks, attn_scale, correction, \
            stride_block, stride_kv, stride_slot, stride_head);

    // HEAD_DIM=64
    if (head_dim == 64 && mse_bits == 1) { LAUNCH(64, 1, 2); }
    else if (head_dim == 64 && mse_bits == 2) { LAUNCH(64, 2, 4); }
    else if (head_dim == 64 && mse_bits == 3) { LAUNCH(64, 3, 8); }
    // HEAD_DIM=128
    else if (head_dim == 128 && mse_bits == 1) { LAUNCH(128, 1, 2); }
    else if (head_dim == 128 && mse_bits == 2) { LAUNCH(128, 2, 4); }
    else if (head_dim == 128 && mse_bits == 3) { LAUNCH(128, 3, 8); }
    // HEAD_DIM=256
    else if (head_dim == 256 && mse_bits == 1) { LAUNCH(256, 1, 2); }
    else if (head_dim == 256 && mse_bits == 2) { LAUNCH(256, 2, 4); }
    else if (head_dim == 256 && mse_bits == 3) { LAUNCH(256, 3, 8); }
    // HEAD_DIM=512
    else if (head_dim == 512 && mse_bits == 1) { LAUNCH(512, 1, 2); }
    else if (head_dim == 512 && mse_bits == 2) { LAUNCH(512, 2, 4); }
    else if (head_dim == 512 && mse_bits == 3) { LAUNCH(512, 3, 8); }
    else {
        TORCH_CHECK(false, "TQ fused decode: unsupported config head_dim=",
                    head_dim, " mse_bits=", mse_bits);
    }
    #undef LAUNCH
}

}  // namespace turboquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq_fused_decode_attention", &turboquant::tq_fused_decode_attention,
          "TQ fused decode with full V decompression");
}
