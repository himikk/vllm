// SPDX-License-Identifier: Apache-2.0
// TurboQuant Compressed Attention Score Kernel
//
// Computes attention scores DIRECTLY from compressed KV cache.
// No decompression needed — reads packed indices + sign bits.
//
// score[q, k] = vec_norm * (term1 + correction * res_norm * term2)
//   term1 = sum_j q_rot[j] * centroids[idx[j]]   ← centroid gather
//   term2 = sum_j q_proj[j] * (2*sign[j] - 1)    ← sign-bit multiply
//
// where q_rot = query @ Pi^T, q_proj = query @ S^T (precomputed per query)
//
// Packed layout per KV vector (TQ3, D=64):
//   [0:16]   MSE indices (64 coords × 2 bits / 8 = 16 bytes)
//   [16:24]  QJL signs   (64 coords × 1 bit / 8 = 8 bytes)
//   [24:26]  vec_norm    (float16 = 2 bytes)
//   [26:28]  res_norm    (float16 = 2 bytes)
//   Total: 28 bytes

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

namespace turboquant {

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

// Compute attention scores from compressed K-cache for one query head
// against all cached tokens.
//
// Grid: (num_q_tokens * num_q_heads, num_cache_blocks)
// Block: (BLOCK_SIZE)
template <int HEAD_DIM, int MSE_BITS, int N_CENTROIDS>
__global__ void tq_compressed_score_kernel(
    // Precomputed rotated/projected queries
    const float* __restrict__ q_rot,       // [num_q, num_q_heads, HEAD_DIM]
    const float* __restrict__ q_proj,      // [num_q, num_q_heads, HEAD_DIM]
    // Compressed K-cache
    const uint8_t* __restrict__ k_cache,   // [num_blocks, block_size, num_kv_heads, packed_size]
    // Centroids
    const float* __restrict__ centroids,   // [N_CENTROIDS]
    // Block table + seq lens
    const int* __restrict__ block_table,   // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    // Output scores
    float* __restrict__ scores,            // [num_q, num_q_heads, max_seq_len]
    // Dims
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int packed_size,
    int max_blocks_per_seq,
    int max_seq_len,
    float attn_scale,
    float correction_scale  // sqrt(pi/2) / HEAD_DIM
) {
    const int qh_idx = blockIdx.x;  // query_token * num_q_heads + q_head
    const int cache_block_tile = blockIdx.y;
    const int tid = threadIdx.x;

    const int q_token = qh_idx / num_q_heads;
    const int q_head = qh_idx % num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    constexpr int D = HEAD_DIM;
    constexpr int MSE_BYTES = (D * MSE_BITS + 7) / 8;
    constexpr int QJL_BYTES = (D + 7) / 8;
    constexpr int MASK = (1 << MSE_BITS) - 1;

    // Check sequence length
    int seq_len = seq_lens[q_token];
    int start_pos = cache_block_tile * block_size;
    if (start_pos >= seq_len) return;

    // Get physical block index
    int phys_block = block_table[q_token * max_blocks_per_seq + cache_block_tile];

    // Load query vectors into shared memory (ALL threads participate)
    __shared__ float s_q_rot[D];
    __shared__ float s_q_proj[D];
    __shared__ float s_centroids[N_CENTROIDS];

    int q_base = (q_token * num_q_heads + q_head) * D;
    // Use multiple threads to cooperatively load D elements
    for (int i = tid; i < D; i += blockDim.x) {
        s_q_rot[i] = q_rot[q_base + i];
        s_q_proj[i] = q_proj[q_base + i];
    }
    if (tid < N_CENTROIDS) {
        s_centroids[tid] = centroids[tid];
    }
    __syncthreads();

    // Only threads with tid < block_size process tokens
    int tok_in_block = tid;
    if (tok_in_block >= block_size) return;

    int abs_pos = start_pos + tok_in_block;
    if (abs_pos >= seq_len) return;

    // Load packed K-vector for this token
    int entry_base = (phys_block * block_size + tok_in_block) * num_kv_heads * packed_size
                   + kv_head * packed_size;
    const uint8_t* packed = k_cache + entry_base;

    // === Term 1: <q_rot, centroids[idx]> ===
    float term1 = 0.0f;
    if constexpr (MSE_BITS == 2) {
        // 4 indices per byte
        for (int b = 0; b < MSE_BYTES; b++) {
            uint8_t byte_val = packed[b];
            for (int k = 0; k < 4 && (b * 4 + k) < D; k++) {
                int j = b * 4 + k;
                int idx = (byte_val >> (k * 2)) & MASK;
                term1 += s_q_rot[j] * s_centroids[idx];
            }
        }
    } else if constexpr (MSE_BITS == 3) {
        for (int j = 0; j < D; j++) {
            int bit_off = j * 3;
            int byte_idx = bit_off / 8;
            int bit_idx = bit_off % 8;
            int val = (packed[byte_idx] >> bit_idx);
            if (bit_idx > 5 && byte_idx + 1 < MSE_BYTES)
                val |= (packed[byte_idx + 1] << (8 - bit_idx));
            int idx = val & MASK;
            term1 += s_q_rot[j] * s_centroids[idx];
        }
    }

    // === Term 2: <q_proj, signs> ===
    float term2 = 0.0f;
    const uint8_t* sign_bytes = packed + MSE_BYTES;
    for (int b = 0; b < QJL_BYTES; b++) {
        uint8_t byte_val = sign_bytes[b];
        for (int k = 0; k < 8 && (b * 8 + k) < D; k++) {
            int j = b * 8 + k;
            float sign_val = ((byte_val >> k) & 1) ? 1.0f : -1.0f;
            term2 += s_q_proj[j] * sign_val;
        }
    }

    // === Load norms ===
    int norm_off = MSE_BYTES + QJL_BYTES;
    // Read 2 bytes as float16
    uint16_t vn_u16 = packed[norm_off] | (packed[norm_off + 1] << 8);
    uint16_t rn_u16 = packed[norm_off + 2] | (packed[norm_off + 3] << 8);
    float vec_norm = __half2float(*reinterpret_cast<const __half*>(&vn_u16));
    float res_norm = __half2float(*reinterpret_cast<const __half*>(&rn_u16));

    // === Combine ===
    float score = vec_norm * (term1 + correction_scale * res_norm * term2);
    score *= attn_scale;

    // Store
    int score_idx = (q_token * num_q_heads + q_head) * max_seq_len + abs_pos;
    scores[score_idx] = score;
}


// Python-callable wrapper
void tq_compressed_attention_score(
    torch::Tensor q_rot,        // [num_q, num_q_heads, head_dim] float32
    torch::Tensor q_proj,       // [num_q, num_q_heads, head_dim] float32
    torch::Tensor k_cache,      // [num_blocks, block_size, num_kv_heads, packed_size] uint8
    torch::Tensor centroids,    // [n_centroids] float32
    torch::Tensor block_table,  // [num_seqs, max_blocks_per_seq] int32
    torch::Tensor seq_lens,     // [num_seqs] int32
    torch::Tensor scores,       // [num_q, num_q_heads, max_seq_len] float32 (output)
    int head_dim,
    int mse_bits,
    int n_centroids,
    float attn_scale
) {
    int num_q = q_rot.size(0);
    int num_q_heads = q_rot.size(1);
    int num_kv_heads = k_cache.size(2);
    int block_size = k_cache.size(1);
    int packed_size = k_cache.size(3);
    int max_blocks_per_seq = block_table.size(1);
    int max_seq_len = scores.size(2);
    float correction = sqrtf(M_PI_2) / static_cast<float>(head_dim);

    int num_cache_blocks = (max_seq_len + block_size - 1) / block_size;
    dim3 grid(num_q * num_q_heads, num_cache_blocks);
    // Need at least head_dim threads to load q_rot/q_proj into shared memory
    int threads = max(block_size, head_dim);
    // Round up to multiple of 32 (warp size)
    threads = ((threads + 31) / 32) * 32;
    dim3 block(threads);

    // Dispatch on head_dim and mse_bits
    if (head_dim == 64 && mse_bits == 2 && n_centroids == 4) {
        tq_compressed_score_kernel<64, 2, 4><<<grid, block>>>(
            q_rot.data_ptr<float>(), q_proj.data_ptr<float>(),
            k_cache.data_ptr<uint8_t>(), centroids.data_ptr<float>(),
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
            scores.data_ptr<float>(),
            num_q_heads, num_kv_heads, block_size, packed_size,
            max_blocks_per_seq, max_seq_len, attn_scale, correction);
    } else if (head_dim == 64 && mse_bits == 3 && n_centroids == 8) {
        tq_compressed_score_kernel<64, 3, 8><<<grid, block>>>(
            q_rot.data_ptr<float>(), q_proj.data_ptr<float>(),
            k_cache.data_ptr<uint8_t>(), centroids.data_ptr<float>(),
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
            scores.data_ptr<float>(),
            num_q_heads, num_kv_heads, block_size, packed_size,
            max_blocks_per_seq, max_seq_len, attn_scale, correction);
    } else {
        // TODO: add D=128, D=256 instantiations when needed
        TORCH_CHECK(false, "TQ score: unsupported head_dim=", head_dim,
            " mse_bits=", mse_bits, " n_centroids=", n_centroids);
    }
}

}  // namespace turboquant
