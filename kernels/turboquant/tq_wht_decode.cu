// SPDX-License-Identifier: Apache-2.0
// TurboQuant v2: WHT Block-Compression Fused Decode Attention
//
// Uses Walsh-Hadamard Transform (WHT) on 32-element blocks instead of D×D rotation.
// Each warp (32 threads) handles one WHT block. WHT inverse via __shfl_xor_sync.
//
// Grid: (num_q_tokens * num_q_heads)
// Block: (HEAD_DIM) threads — organized as HEAD_DIM/32 warps
//
// K score: each warp computes one block's dot product, then sum across warps
// V reconstruct: each warp does WHT inverse on its block via warp shuffles
//
// No Pi/S matrices needed. No shared memory for WHT.
// Performance: O(D * log2(32)) instead of O(D^2) per token.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>  // at::cuda::getCurrentCUDAStream()

namespace turboquant {

#define WARP_SIZE 32
#define WHT_STAGES 5  // log2(32) = 5

// Fixed sign-flip pattern (must match Python wht.py WHT_SIGNS_32)
__constant__ float WHT_SIGNS[32] = {
    +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
    +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1,
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

// In-warp WHT forward: apply sign flips, butterfly stages, normalize
// Each thread holds one of 32 values. After this, thread t holds wht[t].
__device__ __forceinline__ float warp_wht_forward(float val, int lane) {
    // Step 1: sign flip
    val *= WHT_SIGNS[lane];
    // Step 2: 5 butterfly stages
    for (int stage = 0; stage < WHT_STAGES; stage++) {
        int mask = 1 << stage;
        float other = __shfl_xor_sync(0xffffffff, val, mask);
        // If lane's bit at position 'stage' is 0: val = val + other
        // If lane's bit at position 'stage' is 1: val = other - val (= -(val - other))
        if (lane & mask)
            val = other - val;
        else
            val = val + other;
    }
    // Step 3: normalize
    val *= 0.17677669529663688f;  // 1/sqrt(32)
    return val;
}

// In-warp WHT inverse: butterfly stages, normalize, undo sign flips
__device__ __forceinline__ float warp_wht_inverse(float val, int lane) {
    // Step 1: butterfly stages (same as forward)
    for (int stage = 0; stage < WHT_STAGES; stage++) {
        int mask = 1 << stage;
        float other = __shfl_xor_sync(0xffffffff, val, mask);
        if (lane & mask)
            val = other - val;
        else
            val = val + other;
    }
    // Step 2: normalize
    val *= 0.17677669529663688f;  // 1/sqrt(32)
    // Step 3: undo sign flips
    val *= WHT_SIGNS[lane];
    return val;
}

// 2-bit centroids for N(0,1) Lloyd-Max
__constant__ float WHT_CENTROIDS_2BIT[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f,
};

// 3-bit centroids for N(0,1) Lloyd-Max (must match Python)
__constant__ float WHT_CENTROIDS_3BIT[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f,
};

// 4-bit centroids for N(0,1) Lloyd-Max
__constant__ float WHT_CENTROIDS_4BIT[16] = {
    -2.7326f, -2.0690f, -1.6181f, -1.2563f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2563f,  1.6181f,  2.0690f,  2.7326f,
};

// Main WHT decode kernel
// Accepts bfloat16 query and output — zero-copy from vLLM tensors.
// Accepts int64 block_table and seq_lens — no Python .int() needed.
// WHT applied to query in-kernel.
template <int HEAD_DIM, int MSE_BITS, int BLOCK_SIZE = 32>
__global__ void tq_wht_fused_decode_kernel(
    // Raw query [num_q, num_q_heads, D] — bfloat16, WHT applied in-kernel
    const __nv_bfloat16* __restrict__ q_wht,
    // Compressed KV cache: (num_blocks, 2, block_size_kv, num_kv_heads, packed_size)
    const uint8_t* __restrict__ kv_cache,
    // Block table
    const int* __restrict__ block_table,   // [num_seqs, max_blocks_per_seq] int32
    const int* __restrict__ seq_lens,      // [num_seqs] int32
    // Output: [num_q, num_q_heads, D] bfloat16 — writes directly to vLLM output
    __nv_bfloat16* __restrict__ output,
    // Dims
    int num_q_heads,
    int num_kv_heads,
    int block_size_kv,      // KV cache block size (e.g., 16)
    int packed_size,        // bytes per head per slot
    int max_blocks_per_seq,
    float attn_scale,
    // KV cache strides (in bytes)
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
    constexpr int N_BLOCKS_WHT = D / BLOCK_SIZE;  // number of WHT blocks per vector
    constexpr int BYTES_PER_BLOCK = (MSE_BITS == 2) ? 10 : (MSE_BITS == 3) ? 14 : (MSE_BITS == 4) ? 18 : 0;

    // Which WHT block and lane within that block
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int wht_block = warp_id;  // each warp handles one WHT block

    int q_base = (q_token * num_q_heads + q_head) * D;
    int seq_len = seq_lens[q_token];
    if (seq_len <= 0) {
        if (tid < D) output[q_base + tid] = __float2bfloat16(0.0f);
        return;
    }
    int max_seq = max_blocks_per_seq * block_size_kv;
    if (seq_len > max_seq) seq_len = max_seq;

    // Load bfloat16 query → float, then apply WHT in-kernel
    float my_q_wht = (tid < D) ? __bfloat162float(q_wht[q_base + tid]) : 0.0f;
    // WHT forward on query: each warp transforms its 32-element block
    if (tid < D) {
        my_q_wht = warp_wht_forward(my_q_wht, lane);
    }

    // Centroids pointer
    const float* centroids = (MSE_BITS == 2) ? WHT_CENTROIDS_2BIT :
                             (MSE_BITS == 3) ? WHT_CENTROIDS_3BIT : WHT_CENTROIDS_4BIT;

    // Online softmax state
    float m_prev = -INFINITY;
    float d_prev = 0.0f;
    float v_acc = 0.0f;  // per-thread accumulator

    for (int pos = 0; pos < seq_len; pos++) {
        int bi = pos / block_size_kv;
        int bo = pos % block_size_kv;
        if (bi >= max_blocks_per_seq) break;
        int phys_block = block_table[q_token * max_blocks_per_seq + bi];
        if (phys_block < 0) continue;  // skip invalid blocks

        // ── K Score: dot(q_wht, k_wht) in WHT space ──────────────
        // Each warp unpacks one WHT block of K and computes partial dot product
        int k_base = phys_block * stride_block
                   + 0 * stride_kv
                   + bo * stride_slot
                   + kv_head * stride_head;

        float k_score_partial = 0.0f;
        if (wht_block < N_BLOCKS_WHT) {
            const uint8_t* k_block = kv_cache + k_base + wht_block * BYTES_PER_BLOCK;

            // Unpack index for this lane
            int idx;
            if constexpr (MSE_BITS == 2) {
                int byte_idx = lane / 4;
                int shift = (lane % 4) * 2;
                idx = (k_block[byte_idx] >> shift) & 0x3;
            } else if constexpr (MSE_BITS == 3) {
                // 3-bit split: lower 2 bits in qs[8], upper 1 bit in qr[4]
                int qs_byte = lane / 4;
                int qs_shift = (lane % 4) * 2;
                int low2 = (k_block[qs_byte] >> qs_shift) & 0x3;
                int qr_byte = 8 + lane / 8;  // qr starts at byte 8
                int qr_shift = lane % 8;
                int hi1 = (k_block[qr_byte] >> qr_shift) & 0x1;
                idx = low2 | (hi1 << 2);
            } else if constexpr (MSE_BITS == 4) {
                // 4-bit: 2 indices per byte
                int byte_idx = lane / 2;
                int shift = (lane % 2) * 4;
                idx = (k_block[byte_idx] >> shift) & 0xF;
            }

            // Unpack gamma (fp16, last 2 bytes of block)
            int gamma_off = BYTES_PER_BLOCK - 2;
            uint16_t gamma_u16 = k_block[gamma_off] | (k_block[gamma_off + 1] << 8);
            float gamma = __half2float(*reinterpret_cast<const __half*>(&gamma_u16));

            // K value in WHT space = gamma * centroid[idx]
            float k_wht_val = gamma * centroids[idx];

            // Dot product: q_wht[tid] * k_wht[tid] (both in WHT space)
            k_score_partial = my_q_wht * k_wht_val;
        }

        // Reduce within warp → block score
        float warp_sum = warp_reduce_sum(k_score_partial);

        // Sum across warps via shared memory
        __shared__ float s_warp_sums[32];  // max 32 warps
        if (lane == 0 && warp_id < N_BLOCKS_WHT)
            s_warp_sums[warp_id] = warp_sum;
        __syncthreads();

        float score = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < N_BLOCKS_WHT; w++)
                score += s_warp_sums[w];
            score *= attn_scale;
        }
        // Broadcast score to all threads
        __shared__ float s_score;
        if (tid == 0) s_score = score;
        __syncthreads();
        score = s_score;

        // ── Online Softmax ────────────────────────────────────────
        float m_new = fmaxf(m_prev, score);
        float exp_prev = expf(m_prev - m_new);
        float exp_cur = expf(score - m_new);
        float d_new = d_prev * exp_prev + exp_cur;

        v_acc *= exp_prev;

        // ── V Reconstruct + Accumulate ────────────────────────────
        if (wht_block < N_BLOCKS_WHT && tid < D) {
            int v_base = phys_block * stride_block
                       + 1 * stride_kv
                       + bo * stride_slot
                       + kv_head * stride_head;

            const uint8_t* v_block = kv_cache + v_base + wht_block * BYTES_PER_BLOCK;

            // Unpack V index
            int v_idx;
            if constexpr (MSE_BITS == 2) {
                int byte_idx = lane / 4;
                int shift = (lane % 4) * 2;
                v_idx = (v_block[byte_idx] >> shift) & 0x3;
            } else if constexpr (MSE_BITS == 3) {
                int qs_byte = lane / 4;
                int qs_shift = (lane % 4) * 2;
                int low2 = (v_block[qs_byte] >> qs_shift) & 0x3;
                int qr_byte = 8 + lane / 8;
                int qr_shift = lane % 8;
                int hi1 = (v_block[qr_byte] >> qr_shift) & 0x1;
                v_idx = low2 | (hi1 << 2);
            } else if constexpr (MSE_BITS == 4) {
                int byte_idx = lane / 2;
                int shift = (lane % 2) * 4;
                v_idx = (v_block[byte_idx] >> shift) & 0xF;
            }

            // Unpack V gamma
            int gamma_off = BYTES_PER_BLOCK - 2;
            uint16_t vg_u16 = v_block[gamma_off] | (v_block[gamma_off + 1] << 8);
            float v_gamma = __half2float(*reinterpret_cast<const __half*>(&vg_u16));

            // V in WHT space
            float v_wht = v_gamma * centroids[v_idx];

            // WHT inverse via warp shuffles
            float v_recon = warp_wht_inverse(v_wht, lane);

            // Accumulate weighted V
            v_acc += exp_cur * v_recon;
        }

        m_prev = m_new;
        d_prev = d_new;
        __syncthreads();
    }

    // Normalize and store as bfloat16 (zero-copy to vLLM output)
    if (tid < D) {
        float result = (d_prev > 0.0f) ? (v_acc / d_prev) : 0.0f;
        output[q_base + tid] = __float2bfloat16(result);
    }
}


// ── C++ wrapper ───────────────────────────────────────────────

void tq_wht_fused_decode_attention(
    torch::Tensor q_wht,        // [num_q, num_q_heads, D] bfloat16 (raw query, WHT in-kernel)
    torch::Tensor kv_cache,      // [num_blocks, 2, block_size, num_kv_heads, packed_size] uint8
    torch::Tensor block_table,   // [num_seqs, max_blocks_per_seq] int32
    torch::Tensor seq_lens,      // [num_seqs] int32
    torch::Tensor output,        // [num_q, num_q_heads, D] bfloat16 (direct output)
    int head_dim,
    int mse_bits,
    float attn_scale,
    int stride_block,
    int stride_kv,
    int stride_slot,
    int stride_head
) {
    int num_q = q_wht.size(0);
    int num_q_heads = q_wht.size(1);
    int num_kv_heads = kv_cache.size(3);
    int block_size_kv = kv_cache.size(2);
    int packed_size = kv_cache.size(4);
    int max_blocks = block_table.size(1);

    dim3 grid(num_q * num_q_heads);
    // One thread per dimension, organized as D/32 warps
    int threads = ((head_dim + 31) / 32) * 32;
    dim3 block(threads);

    // Launch on current PyTorch stream (critical for CUDA Graph capture).
    // Without this, kernel runs on default stream 0 and graph replay
    // produces corrupt output because the kernel was never recorded.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH_WHT(HD, MB) \
        tq_wht_fused_decode_kernel<HD, MB><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(q_wht.data_ptr()), \
            kv_cache.data_ptr<uint8_t>(), \
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(), \
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), \
            num_q_heads, num_kv_heads, block_size_kv, packed_size, \
            max_blocks, attn_scale, \
            stride_block, stride_kv, stride_slot, stride_head);

    if (head_dim == 128 && mse_bits == 2) { LAUNCH_WHT(128, 2); }
    else if (head_dim == 128 && mse_bits == 3) { LAUNCH_WHT(128, 3); }
    else if (head_dim == 128 && mse_bits == 4) { LAUNCH_WHT(128, 4); }
    else if (head_dim == 256 && mse_bits == 2) { LAUNCH_WHT(256, 2); }
    else if (head_dim == 256 && mse_bits == 3) { LAUNCH_WHT(256, 3); }
    else if (head_dim == 256 && mse_bits == 4) { LAUNCH_WHT(256, 4); }
    else if (head_dim == 64 && mse_bits == 2) { LAUNCH_WHT(64, 2); }
    else if (head_dim == 64 && mse_bits == 3) { LAUNCH_WHT(64, 3); }
    else if (head_dim == 64 && mse_bits == 4) { LAUNCH_WHT(64, 4); }
    else if (head_dim == 512 && mse_bits == 2) { LAUNCH_WHT(512, 2); }
    else if (head_dim == 512 && mse_bits == 3) { LAUNCH_WHT(512, 3); }
    else if (head_dim == 512 && mse_bits == 4) { LAUNCH_WHT(512, 4); }
    else {
        TORCH_CHECK(false, "WHT fused decode: unsupported config head_dim=",
                    head_dim, " mse_bits=", mse_bits);
    }
    #undef LAUNCH_WHT
}

}  // namespace turboquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq_wht_fused_decode_attention", &turboquant::tq_wht_fused_decode_attention,
          "TQ WHT fused decode with warp-shuffle V decompression");
}
