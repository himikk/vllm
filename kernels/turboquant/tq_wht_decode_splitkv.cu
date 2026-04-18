// SPDX-License-Identifier: Apache-2.0
// TurboQuant v3: Split-KV WHT Decode Attention
//
// Split-KV parallelism: splits KV positions across multiple thread blocks.
// Stage 1: Each block processes a chunk of KV positions independently.
// Stage 2: Merge kernel combines partial softmax states.
//
// Grid Stage 1: (num_q_tokens * num_q_heads, num_splits)
// Grid Stage 2: (num_q_tokens * num_q_heads)
// Block: (HEAD_DIM) threads = HEAD_DIM/32 warps
//
// vs original: 8× more blocks → 60%+ GPU utilization (was 8%)
// ~3× fewer __syncthreads per position (score reduction optimized)

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>

namespace turboquant {

#define WARP_SIZE 32
#define WHT_STAGES 5
#define MAX_SPLITS 32

// Reuse WHT constants and functions from tq_wht_decode.cu
__constant__ float SKV_WHT_SIGNS[32] = {
    +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
    +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1,
};

__constant__ float SKV_CENTROIDS_2BIT[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f,
};

__constant__ float SKV_CENTROIDS_3BIT[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f,
};

__constant__ float SKV_CENTROIDS_4BIT[16] = {
    -2.7326f, -2.0690f, -1.6181f, -1.2563f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2563f,  1.6181f,  2.0690f,  2.7326f,
};

__device__ __forceinline__ float skv_warp_reduce_sum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

__device__ __forceinline__ float skv_warp_wht_forward(float val, int lane) {
    val *= SKV_WHT_SIGNS[lane];
    for (int stage = 0; stage < WHT_STAGES; stage++) {
        int mask = 1 << stage;
        float other = __shfl_xor_sync(0xffffffff, val, mask);
        if (lane & mask) val = other - val;
        else val = val + other;
    }
    val *= 0.17677669529663688f;
    return val;
}

__device__ __forceinline__ float skv_warp_wht_inverse(float val, int lane) {
    for (int stage = 0; stage < WHT_STAGES; stage++) {
        int mask = 1 << stage;
        float other = __shfl_xor_sync(0xffffffff, val, mask);
        if (lane & mask) val = other - val;
        else val = val + other;
    }
    val *= 0.17677669529663688f;
    val *= SKV_WHT_SIGNS[lane];
    return val;
}

// ================================================================
// Stage 1: Split-KV Decode Kernel
// Each block processes [split_start, split_end) KV positions.
// Writes partial (m, d, v_acc[D]) to split_buf.
// ================================================================
template <int HEAD_DIM, int MSE_BITS, int BLOCK_SIZE = 32>
__global__ void tq_wht_splitkv_stage1(
    const __nv_bfloat16* __restrict__ q_raw,
    const uint8_t* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    // Split output: [num_q * num_q_heads, num_splits, D + 2] float32
    // Layout per split: [v_acc[0..D-1], m, d]
    float* __restrict__ split_buf,
    int num_q_heads,
    int num_kv_heads,
    int block_size_kv,
    int packed_size,
    int max_blocks_per_seq,
    float attn_scale,
    int stride_block,
    int stride_kv,
    int stride_slot,
    int stride_head,
    int num_splits
) {
    const int qh_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int q_token = qh_idx / num_q_heads;
    const int q_head = qh_idx % num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int tid = threadIdx.x;
    constexpr int D = HEAD_DIM;
    constexpr int N_BLOCKS_WHT = D / BLOCK_SIZE;
    constexpr int BYTES_PER_BLOCK = (MSE_BITS == 2) ? 10 : (MSE_BITS == 3) ? 14 : (MSE_BITS == 4) ? 18 : 0;
    constexpr int SPLIT_STRIDE = D + 2;  // v_acc[D] + m + d

    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int wht_block = warp_id;

    int seq_len = seq_lens[q_token];
    int max_seq = max_blocks_per_seq * block_size_kv;
    if (seq_len > max_seq) seq_len = max_seq;

    // Compute this split's range
    int positions_per_split = (seq_len + num_splits - 1) / num_splits;
    int split_start = split_idx * positions_per_split;
    int split_end = min(split_start + positions_per_split, seq_len);

    // Output pointer for this split
    float* my_split = split_buf + (qh_idx * num_splits + split_idx) * SPLIT_STRIDE;

    // Handle empty split
    if (split_start >= split_end || seq_len <= 0) {
        if (tid < D) my_split[tid] = 0.0f;
        if (tid == 0) {
            my_split[D] = -INFINITY;  // m
            my_split[D + 1] = 0.0f;   // d
        }
        return;
    }

    // Load query and apply WHT
    int q_base = (q_token * num_q_heads + q_head) * D;
    float my_q_wht = (tid < D) ? __bfloat162float(q_raw[q_base + tid]) : 0.0f;
    if (tid < D) {
        my_q_wht = skv_warp_wht_forward(my_q_wht, lane);
    }

    const float* centroids = (MSE_BITS == 2) ? SKV_CENTROIDS_2BIT :
                             (MSE_BITS == 3) ? SKV_CENTROIDS_3BIT : SKV_CENTROIDS_4BIT;

    // Online softmax state (local to this split)
    float m_prev = -INFINITY;
    float d_prev = 0.0f;
    float v_acc = 0.0f;

    // Shared memory for inter-warp score reduction
    __shared__ float s_warp_sums[32];
    __shared__ float s_score;

    for (int pos = split_start; pos < split_end; pos++) {
        int bi = pos / block_size_kv;
        int bo = pos % block_size_kv;
        if (bi >= max_blocks_per_seq) break;
        int phys_block = block_table[q_token * max_blocks_per_seq + bi];
        if (phys_block < 0) continue;

        // ── K Score ──────────────────────────────────────────
        int k_base = phys_block * stride_block
                   + 0 * stride_kv
                   + bo * stride_slot
                   + kv_head * stride_head;

        float k_score_partial = 0.0f;
        if (wht_block < N_BLOCKS_WHT) {
            const uint8_t* k_block = kv_cache + k_base + wht_block * BYTES_PER_BLOCK;

            int idx;
            if constexpr (MSE_BITS == 2) {
                // 2-bit: 4 indices per byte, 8 bytes for 32 values
                int byte_idx = lane / 4;
                int shift = (lane % 4) * 2;
                idx = (k_block[byte_idx] >> shift) & 0x3;
            } else if constexpr (MSE_BITS == 3) {
                int qs_byte = lane / 4;
                int qs_shift = (lane % 4) * 2;
                int low2 = (k_block[qs_byte] >> qs_shift) & 0x3;
                int qr_byte = 8 + lane / 8;
                int qr_shift = lane % 8;
                int hi1 = (k_block[qr_byte] >> qr_shift) & 0x1;
                idx = low2 | (hi1 << 2);
            } else if constexpr (MSE_BITS == 4) {
                int byte_idx = lane / 2;
                int shift = (lane % 2) * 4;
                idx = (k_block[byte_idx] >> shift) & 0xF;
            }

            int gamma_off = BYTES_PER_BLOCK - 2;
            uint16_t gamma_u16 = k_block[gamma_off] | (k_block[gamma_off + 1] << 8);
            float gamma = __half2float(*reinterpret_cast<const __half*>(&gamma_u16));
            float k_wht_val = gamma * centroids[idx];
            k_score_partial = my_q_wht * k_wht_val;
        }

        // Warp reduce + inter-warp sum
        float warp_sum = skv_warp_reduce_sum(k_score_partial);
        if (lane == 0 && warp_id < N_BLOCKS_WHT)
            s_warp_sums[warp_id] = warp_sum;
        __syncthreads();

        // Warp-0 reduces across warps (no thread-0 serial loop)
        float score;
        if (warp_id == 0) {
            float wval = (lane < N_BLOCKS_WHT) ? s_warp_sums[lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                wval += __shfl_down_sync(0xffffffff, wval, off);
            if (lane == 0) s_score = wval * attn_scale;
        }
        __syncthreads();
        score = s_score;

        // ── Online Softmax ───────────────────────────────────
        float m_new = fmaxf(m_prev, score);
        float exp_prev = expf(m_prev - m_new);
        float exp_cur = expf(score - m_new);
        float d_new = d_prev * exp_prev + exp_cur;
        v_acc *= exp_prev;

        // ── V Reconstruct ────────────────────────────────────
        if (wht_block < N_BLOCKS_WHT && tid < D) {
            int v_base = phys_block * stride_block
                       + 1 * stride_kv
                       + bo * stride_slot
                       + kv_head * stride_head;

            const uint8_t* v_block = kv_cache + v_base + wht_block * BYTES_PER_BLOCK;

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

            int gamma_off = BYTES_PER_BLOCK - 2;
            uint16_t vg_u16 = v_block[gamma_off] | (v_block[gamma_off + 1] << 8);
            float v_gamma = __half2float(*reinterpret_cast<const __half*>(&vg_u16));
            float v_wht = v_gamma * centroids[v_idx];
            float v_recon = skv_warp_wht_inverse(v_wht, lane);
            v_acc += exp_cur * v_recon;
        }

        m_prev = m_new;
        d_prev = d_new;
    }

    // Write partial result: v_acc[D], m, d
    if (tid < D) my_split[tid] = v_acc;
    if (tid == 0) {
        my_split[D] = m_prev;
        my_split[D + 1] = d_prev;
    }
}


// ================================================================
// Stage 2: Merge Kernel
// Combines partial softmax states from all splits into final output.
// ================================================================
template <int HEAD_DIM>
__global__ void tq_wht_splitkv_merge(
    const float* __restrict__ split_buf,  // [num_qh, num_splits, D+2]
    __nv_bfloat16* __restrict__ output,   // [num_q, num_q_heads, D] bf16
    int num_q_heads,
    int num_splits
) {
    const int qh_idx = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int D = HEAD_DIM;
    constexpr int SPLIT_STRIDE = D + 2;

    if (tid >= D) return;

    // First pass: find global max
    float m_global = -INFINITY;
    for (int s = 0; s < num_splits; s++) {
        const float* sp = split_buf + (qh_idx * num_splits + s) * SPLIT_STRIDE;
        float m_s = sp[D];  // m for this split
        m_global = fmaxf(m_global, m_s);
    }

    // Second pass: merge
    float d_global = 0.0f;
    float v_global = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        const float* sp = split_buf + (qh_idx * num_splits + s) * SPLIT_STRIDE;
        float m_s = sp[D];
        float d_s = sp[D + 1];
        float v_s = sp[tid];  // v_acc[tid]

        float scale = expf(m_s - m_global);
        d_global += d_s * scale;
        v_global += v_s * scale;
    }

    // Normalize and write output
    int q_token = qh_idx / num_q_heads;
    int q_head = qh_idx % num_q_heads;
    int out_idx = (q_token * num_q_heads + q_head) * D + tid;
    float result = (d_global > 0.0f) ? (v_global / d_global) : 0.0f;
    output[out_idx] = __float2bfloat16(result);
}


// ── C++ wrapper ───────────────────────────────────────────────

void tq_wht_splitkv_decode(
    torch::Tensor q_raw,         // [num_q, num_q_heads, D] bfloat16
    torch::Tensor kv_cache,      // uint8
    torch::Tensor block_table,   // [num_seqs, max_blocks] int32
    torch::Tensor seq_lens,      // [num_seqs] int32
    torch::Tensor split_buf,     // [num_qh, num_splits, D+2] float32 (pre-allocated)
    torch::Tensor output,        // [num_q, num_q_heads, D] bfloat16
    int head_dim,
    int mse_bits,
    float attn_scale,
    int stride_block,
    int stride_kv,
    int stride_slot,
    int stride_head,
    int num_splits
) {
    int num_q = q_raw.size(0);
    int num_q_heads = q_raw.size(1);
    int num_kv_heads = kv_cache.size(3);
    int block_size_kv = kv_cache.size(2);
    int packed_size = kv_cache.size(4);
    int max_blocks = block_table.size(1);
    int num_qh = num_q * num_q_heads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Stage 1: Split-KV decode
    {
        dim3 grid(num_qh, num_splits);
        int threads = ((head_dim + 31) / 32) * 32;
        dim3 block(threads);

        #define LAUNCH_S1(HD, MB) \
            tq_wht_splitkv_stage1<HD, MB><<<grid, block, 0, stream>>>( \
                reinterpret_cast<const __nv_bfloat16*>(q_raw.data_ptr()), \
                kv_cache.data_ptr<uint8_t>(), \
                block_table.data_ptr<int>(), seq_lens.data_ptr<int>(), \
                split_buf.data_ptr<float>(), \
                num_q_heads, num_kv_heads, block_size_kv, packed_size, \
                max_blocks, attn_scale, \
                stride_block, stride_kv, stride_slot, stride_head, \
                num_splits);

        if (head_dim == 256 && mse_bits == 2) { LAUNCH_S1(256, 2); }
        else if (head_dim == 256 && mse_bits == 3) { LAUNCH_S1(256, 3); }
        else if (head_dim == 256 && mse_bits == 4) { LAUNCH_S1(256, 4); }
        else if (head_dim == 128 && mse_bits == 2) { LAUNCH_S1(128, 2); }
        else if (head_dim == 128 && mse_bits == 3) { LAUNCH_S1(128, 3); }
        else if (head_dim == 128 && mse_bits == 4) { LAUNCH_S1(128, 4); }
        else if (head_dim == 64 && mse_bits == 2) { LAUNCH_S1(64, 2); }
        else if (head_dim == 64 && mse_bits == 3) { LAUNCH_S1(64, 3); }
        else if (head_dim == 64 && mse_bits == 4) { LAUNCH_S1(64, 4); }
        else {
            TORCH_CHECK(false, "Split-KV decode: unsupported head_dim=",
                        head_dim, " mse_bits=", mse_bits);
        }
        #undef LAUNCH_S1
    }

    // Stage 2: Merge
    {
        dim3 grid(num_qh);
        dim3 block(((head_dim + 31) / 32) * 32);

        #define LAUNCH_S2(HD) \
            tq_wht_splitkv_merge<HD><<<grid, block, 0, stream>>>( \
                split_buf.data_ptr<float>(), \
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), \
                num_q_heads, num_splits);

        if (head_dim == 256) { LAUNCH_S2(256); }
        else if (head_dim == 128) { LAUNCH_S2(128); }
        else if (head_dim == 64) { LAUNCH_S2(64); }
        else { LAUNCH_S2(256); }  // fallback
        #undef LAUNCH_S2
    }
}

}  // namespace turboquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq_wht_splitkv_decode", &turboquant::tq_wht_splitkv_decode,
          "Split-KV WHT fused decode attention (Stage 1 + Stage 2)");
}
