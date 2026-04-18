// SPDX-License-Identifier: Apache-2.0
// TurboQuant: Block-Rotation Fused Decode Attention
//
// Same structure as tq_wht_decode.cu but uses per-block random orthogonal
// rotation instead of fixed Walsh-Hadamard Transform.
//
// Pi_blocks: [n_blocks, 32, 32] float32 — per-layer rotation matrices.
// Each warp loads its block's Pi row (32 floats) and uses warp shuffles
// to compute the matrix-vector product.
//
// Grid: (num_q_tokens * num_q_heads)
// Block: (HEAD_DIM) threads — organized as HEAD_DIM/32 warps

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>

namespace turboquant {

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum_br(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

// Warp-level forward rotation: output[lane] = sum_j Pi[lane][j] * input[j]
__device__ __forceinline__ float warp_rotate_forward(
    float val, const float* Pi_row
) {
    float result = 0.0f;
    #pragma unroll
    for (int j = 0; j < WARP_SIZE; j++) {
        float input_j = __shfl_sync(0xffffffff, val, j);
        result += Pi_row[j] * input_j;
    }
    return result;
}

// Warp-level inverse rotation: output[lane] = sum_j Pi[j][lane] * input[j]
// = sum_j Pi^T[lane][j] * input[j]
// Since Pi is orthogonal, Pi^{-1} = Pi^T.
// Pi_col[j] = Pi[j * WARP_SIZE + lane] for each j.
__device__ __forceinline__ float warp_rotate_inverse(
    float val, int lane, const float* Pi_block  // [32, 32] for this block
) {
    float result = 0.0f;
    #pragma unroll
    for (int j = 0; j < WARP_SIZE; j++) {
        float input_j = __shfl_sync(0xffffffff, val, j);
        // Pi^T[lane][j] = Pi[j][lane]
        result += Pi_block[j * WARP_SIZE + lane] * input_j;
    }
    return result;
}

// 3-bit centroids for N(0,1) Lloyd-Max
__constant__ float BROT_CENTROIDS_3BIT[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f,
};

// 4-bit centroids
__constant__ float BROT_CENTROIDS_4BIT[16] = {
    -2.7326f, -2.0690f, -1.6181f, -1.2563f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2563f,  1.6181f,  2.0690f,  2.7326f,
};

template <int HEAD_DIM, int MSE_BITS, int BLOCK_SIZE = 32>
__global__ void tq_blockrot_fused_decode_kernel(
    const __nv_bfloat16* __restrict__ q_raw,     // [num_q, num_q_heads, D] bf16
    const float* __restrict__ Pi_blocks,          // [n_blocks, 32, 32] float32
    const uint8_t* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ output,
    int num_q_heads,
    int num_kv_heads,
    int block_size_kv,
    int packed_size,
    int max_blocks_per_seq,
    float attn_scale,
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
    constexpr int N_BLOCKS = D / BLOCK_SIZE;
    constexpr int BYTES_PER_BLOCK = (MSE_BITS == 3) ? 14 : (MSE_BITS == 4) ? 18 : 0;

    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int rot_block = warp_id;

    int q_base = (q_token * num_q_heads + q_head) * D;
    int seq_len = seq_lens[q_token];
    if (seq_len <= 0) {
        if (tid < D) output[q_base + tid] = __float2bfloat16(0.0f);
        return;
    }
    int max_seq = max_blocks_per_seq * block_size_kv;
    if (seq_len > max_seq) seq_len = max_seq;

    // Load raw bf16 query → float
    float my_q_raw = (tid < D) ? __bfloat162float(q_raw[q_base + tid]) : 0.0f;

    // Forward rotation on query: q_rot = Pi @ q
    // Each warp rotates its 32-element block
    float my_q_rot = 0.0f;
    if (tid < D) {
        const float* Pi_row = Pi_blocks + rot_block * BLOCK_SIZE * BLOCK_SIZE
                            + lane * BLOCK_SIZE;
        my_q_rot = warp_rotate_forward(my_q_raw, Pi_row);
    }

    const float* centroids = (MSE_BITS == 3) ? BROT_CENTROIDS_3BIT : BROT_CENTROIDS_4BIT;

    // Pointer to this block's Pi matrix (for V inverse rotation)
    const float* my_Pi_block = Pi_blocks + rot_block * BLOCK_SIZE * BLOCK_SIZE;

    // Online softmax state
    float m_prev = -INFINITY;
    float d_prev = 0.0f;
    float v_acc = 0.0f;

    for (int pos = 0; pos < seq_len; pos++) {
        int bi = pos / block_size_kv;
        int bo = pos % block_size_kv;
        if (bi >= max_blocks_per_seq) break;
        int phys_block = block_table[q_token * max_blocks_per_seq + bi];
        if (phys_block < 0) continue;

        // ── K Score ──────────────────────────────────────────────
        int k_base = phys_block * stride_block
                   + 0 * stride_kv
                   + bo * stride_slot
                   + kv_head * stride_head;

        float k_score_partial = 0.0f;
        if (rot_block < N_BLOCKS) {
            const uint8_t* k_block = kv_cache + k_base + rot_block * BYTES_PER_BLOCK;

            int idx;
            if constexpr (MSE_BITS == 3) {
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

            // K in rotated space
            float k_rot_val = gamma * centroids[idx];

            // Dot product in rotated space: q_rot . k_rot
            k_score_partial = my_q_rot * k_rot_val;
        }

        float warp_sum = warp_reduce_sum_br(k_score_partial);

        __shared__ float s_warp_sums[32];
        if (lane == 0 && warp_id < N_BLOCKS)
            s_warp_sums[warp_id] = warp_sum;
        __syncthreads();

        float score = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < N_BLOCKS; w++)
                score += s_warp_sums[w];
            score *= attn_scale;
        }
        __shared__ float s_score;
        if (tid == 0) s_score = score;
        __syncthreads();
        score = s_score;

        // ── Online Softmax ───────────────────────────────────────
        float m_new = fmaxf(m_prev, score);
        float exp_prev = expf(m_prev - m_new);
        float exp_cur = expf(score - m_new);
        float d_new = d_prev * exp_prev + exp_cur;

        v_acc *= exp_prev;

        // ── V Reconstruct ────────────────────────────────────────
        if (rot_block < N_BLOCKS && tid < D) {
            int v_base = phys_block * stride_block
                       + 1 * stride_kv
                       + bo * stride_slot
                       + kv_head * stride_head;

            const uint8_t* v_block = kv_cache + v_base + rot_block * BYTES_PER_BLOCK;

            int v_idx;
            if constexpr (MSE_BITS == 3) {
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

            // V in rotated space
            float v_rot = v_gamma * centroids[v_idx];

            // Inverse rotation via warp shuffles: Pi^T @ v_rot
            float v_recon = warp_rotate_inverse(v_rot, lane, my_Pi_block);

            v_acc += exp_cur * v_recon;
        }

        m_prev = m_new;
        d_prev = d_new;
        __syncthreads();
    }

    if (tid < D) {
        float result = (d_prev > 0.0f) ? (v_acc / d_prev) : 0.0f;
        output[q_base + tid] = __float2bfloat16(result);
    }
}


// ── C++ wrapper ───────────────────────────────────────────────

void tq_blockrot_fused_decode_attention(
    torch::Tensor q_raw,        // [num_q, num_q_heads, D] bfloat16
    torch::Tensor Pi_blocks,    // [n_blocks, 32, 32] float32
    torch::Tensor kv_cache,     // [num_blocks, 2, block_size, num_kv_heads, packed_size] uint8
    torch::Tensor block_table,  // [num_seqs, max_blocks_per_seq] int32
    torch::Tensor seq_lens,     // [num_seqs] int32
    torch::Tensor output,       // [num_q, num_q_heads, D] bfloat16
    int head_dim,
    int mse_bits,
    float attn_scale,
    int stride_block,
    int stride_kv,
    int stride_slot,
    int stride_head
) {
    int num_q = q_raw.size(0);
    int num_q_heads = q_raw.size(1);
    int num_kv_heads = kv_cache.size(3);
    int block_size_kv = kv_cache.size(2);
    int packed_size = kv_cache.size(4);
    int max_blocks = block_table.size(1);

    dim3 grid(num_q * num_q_heads);
    int threads = ((head_dim + 31) / 32) * 32;
    dim3 block(threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH_BROT(HD, MB) \
        tq_blockrot_fused_decode_kernel<HD, MB><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(q_raw.data_ptr()), \
            Pi_blocks.data_ptr<float>(), \
            kv_cache.data_ptr<uint8_t>(), \
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(), \
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), \
            num_q_heads, num_kv_heads, block_size_kv, packed_size, \
            max_blocks, attn_scale, \
            stride_block, stride_kv, stride_slot, stride_head);

    if (head_dim == 128 && mse_bits == 3) { LAUNCH_BROT(128, 3); }
    else if (head_dim == 128 && mse_bits == 4) { LAUNCH_BROT(128, 4); }
    else if (head_dim == 256 && mse_bits == 3) { LAUNCH_BROT(256, 3); }
    else if (head_dim == 256 && mse_bits == 4) { LAUNCH_BROT(256, 4); }
    else if (head_dim == 64 && mse_bits == 3) { LAUNCH_BROT(64, 3); }
    else if (head_dim == 64 && mse_bits == 4) { LAUNCH_BROT(64, 4); }
    else if (head_dim == 512 && mse_bits == 3) { LAUNCH_BROT(512, 3); }
    else if (head_dim == 512 && mse_bits == 4) { LAUNCH_BROT(512, 4); }
    else {
        TORCH_CHECK(false, "Block-rot fused decode: unsupported config head_dim=",
                    head_dim, " mse_bits=", mse_bits);
    }
    #undef LAUNCH_BROT
}

}  // namespace turboquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq_blockrot_fused_decode_attention",
          &turboquant::tq_blockrot_fused_decode_attention,
          "Block-rotation fused decode attention with warp-shuffle V decompression");
}
