// SPDX-License-Identifier: Apache-2.0
// RotorQuant Fused Compressed Decode Attention — Full V Decompression
//
// Like TQ kernel but V-MSE uses Clifford sandwich inverse instead of Pi GEMV.
// QJL correction still uses S GEMV (same as TQ).
//
// Grid: (num_q_tokens * num_q_heads)
// Block: (HEAD_DIM) — one thread per dimension
//
// V-MSE decompression: Clifford inverse per 3-dim group (shared memory)
// V-QJL correction: S GEMV per dimension (same as TQ)

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace rotorquant {

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

// Cl(3,0) geometric product
__device__ __forceinline__ void geo_prod(
    const float a[8], const float b[8], float c[8]
) {
    c[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
          - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];
    c[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] + a[4]*b[2]
          - a[3]*b[5] + a[5]*b[3] + a[6]*b[7] + a[7]*b[6];
    c[2] = a[0]*b[2] + a[2]*b[0] + a[1]*b[4] - a[4]*b[1]
          - a[3]*b[6] + a[6]*b[3] - a[5]*b[7] - a[7]*b[5];
    c[3] = a[0]*b[3] + a[3]*b[0] + a[1]*b[5] - a[5]*b[1]
          + a[2]*b[6] - a[6]*b[2] + a[4]*b[7] + a[7]*b[4];
    c[4] = a[0]*b[4] + a[4]*b[0] + a[1]*b[2] - a[2]*b[1]
          + a[5]*b[6] - a[6]*b[5] + a[3]*b[7] - a[7]*b[3];
    c[5] = a[0]*b[5] + a[5]*b[0] + a[1]*b[3] - a[3]*b[1]
          - a[4]*b[6] + a[6]*b[4] - a[2]*b[7] + a[7]*b[2];
    c[6] = a[0]*b[6] + a[6]*b[0] + a[2]*b[3] - a[3]*b[2]
          + a[4]*b[5] - a[5]*b[4] + a[1]*b[7] - a[7]*b[1];
    c[7] = a[0]*b[7] + a[7]*b[0] + a[1]*b[6] - a[6]*b[1]
          - a[2]*b[5] + a[5]*b[2] + a[3]*b[4] - a[4]*b[3];
}

// Clifford inverse rotation for 3 values at group g:
// embed → R̃·x·R → extract
__device__ __forceinline__ void clifford_inverse_group(
    float x0, float x1, float x2,
    const float* rotor,  // [8] for this group
    float &out0, float &out1, float &out2
) {
    // Embed as multivector
    float mv[8] = {0, x0, x1, x2, 0, 0, 0, 0};

    // Rotor reverse: negate grade-2 and grade-3
    float Rr[8] = {rotor[0], rotor[1], rotor[2], rotor[3],
                   -rotor[4], -rotor[5], -rotor[6], -rotor[7]};
    float R[8];
    for (int i = 0; i < 8; i++) R[i] = rotor[i];

    // Inverse sandwich: R̃ * mv * R
    float tmp[8], result[8];
    geo_prod(Rr, mv, tmp);
    geo_prod(tmp, R, result);

    out0 = result[1];
    out1 = result[2];
    out2 = result[3];
}

template <int HEAD_DIM, int MSE_BITS, int N_CENTROIDS>
__global__ void rq_fused_decode_kernel(
    const float* __restrict__ q_rot,
    const float* __restrict__ q_proj,
    const uint8_t* __restrict__ kv_cache,
    const float* __restrict__ rotors,     // [n_groups, 8] Clifford rotors
    const float* __restrict__ S,          // [D, D] QJL matrix
    const float* __restrict__ centroids,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ output,
    int num_q_heads, int num_kv_heads,
    int block_size, int packed_size, int max_blocks_per_seq,
    float attn_scale, float correction_scale,
    int stride_block, int stride_kv, int stride_slot, int stride_head
) {
    constexpr int D = HEAD_DIM;
    constexpr int MSE_BYTES = (D * MSE_BITS + 7) / 8;
    constexpr int QJL_BYTES = (D + 7) / 8;
    constexpr int MASK = (1 << MSE_BITS) - 1;
    const int N_GROUPS = (D + 2) / 3;

    const int qh_idx = blockIdx.x;
    const int q_token = qh_idx / num_q_heads;
    const int q_head = qh_idx % num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int tid = threadIdx.x;

    int seq_len = seq_lens[q_token];

    __shared__ float s_q_rot[D];
    __shared__ float s_q_proj[D];
    __shared__ float s_centroids[N_CENTROIDS];
    __shared__ int s_v_idx[D];
    __shared__ float s_v_sign[D];

    int q_base = (q_token * num_q_heads + q_head) * D;
    for (int i = tid; i < D; i += blockDim.x) {
        s_q_rot[i] = q_rot[q_base + i];
        s_q_proj[i] = q_proj[q_base + i];
    }
    if (tid < N_CENTROIDS) s_centroids[tid] = centroids[tid];
    __syncthreads();

    float m_prev = -INFINITY;
    float d_prev = 0.0f;
    float v_acc = 0.0f;

    for (int pos = 0; pos < seq_len; pos++) {
        int bi = pos / block_size;
        int bo = pos % block_size;
        int phys_block = block_table[q_token * max_blocks_per_seq + bi];

        // K score (same as TQ — thread 0 serial)
        int k_base = phys_block * stride_block + 0 * stride_kv
                   + bo * stride_slot + kv_head * stride_head;
        const uint8_t* k_packed = kv_cache + k_base;

        __shared__ float s_score;
        if (tid == 0) {
            float term1 = 0.0f;
            if constexpr (MSE_BITS == 1) {
                for (int b = 0; b < MSE_BYTES; b++) {
                    uint8_t bv = k_packed[b];
                    for (int k = 0; k < 8 && (b*8+k) < D; k++) {
                        term1 += s_q_rot[b*8+k] * s_centroids[(bv >> k) & 1];
                    }
                }
            } else if constexpr (MSE_BITS == 2) {
                for (int b = 0; b < MSE_BYTES; b++) {
                    uint8_t bv = k_packed[b];
                    for (int k = 0; k < 4 && (b*4+k) < D; k++) {
                        int j = b*4+k;
                        term1 += s_q_rot[j] * s_centroids[(bv >> (k*2)) & MASK];
                    }
                }
            } else if constexpr (MSE_BITS == 3) {
                for (int j = 0; j < D; j++) {
                    int bo2 = j*3; int bi2 = bo2/8; int bs2 = bo2%8;
                    int val = (k_packed[bi2] >> bs2);
                    if (bs2 > 5 && bi2+1 < MSE_BYTES)
                        val |= (k_packed[bi2+1] << (8-bs2));
                    term1 += s_q_rot[j] * s_centroids[val & MASK];
                }
            }
            float term2 = 0.0f;
            for (int b = 0; b < QJL_BYTES; b++) {
                uint8_t bv = k_packed[MSE_BYTES + b];
                for (int k = 0; k < 8 && (b*8+k) < D; k++) {
                    term2 += s_q_proj[b*8+k] * (((bv >> k) & 1) ? 1.0f : -1.0f);
                }
            }
            int no = MSE_BYTES + QJL_BYTES;
            uint16_t vn_u16 = k_packed[no] | (k_packed[no+1] << 8);
            uint16_t rn_u16 = k_packed[no+2] | (k_packed[no+3] << 8);
            float vn = __half2float(*(const __half*)&vn_u16);
            float rn = __half2float(*(const __half*)&rn_u16);
            s_score = vn * (term1 + correction_scale * rn * term2) * attn_scale;
        }
        __syncthreads();
        float score = s_score;

        float m_new = fmaxf(m_prev, score);
        float exp_prev = expf(m_prev - m_new);
        float exp_cur = expf(score - m_new);
        float d_new = d_prev * exp_prev + exp_cur;
        v_acc *= exp_prev;

        // V: unpack indices + signs → shared memory
        if (tid < D) {
            int v_base = phys_block * stride_block + 1 * stride_kv
                       + bo * stride_slot + kv_head * stride_head;
            const uint8_t* vp = kv_cache + v_base;
            int v_idx;
            if constexpr (MSE_BITS == 1) {
                v_idx = (vp[tid/8] >> (tid%8)) & 1;
            } else if constexpr (MSE_BITS == 2) {
                v_idx = (vp[tid/4] >> ((tid%4)*2)) & MASK;
            } else {
                int boff = tid * MSE_BITS; int byi = boff/8; int bsi = boff%8;
                v_idx = (vp[byi] >> bsi);
                if (bsi + MSE_BITS > 8 && byi+1 < MSE_BYTES)
                    v_idx |= (vp[byi+1] << (8-bsi));
                v_idx &= MASK;
            }
            s_v_idx[tid] = v_idx;
            s_v_sign[tid] = ((vp[MSE_BYTES + tid/8] >> (tid%8)) & 1) ? 1.0f : -1.0f;
        }
        __syncthreads();

        // V decompression: Clifford inverse per group + QJL GEMV
        if (tid < D) {
            const uint8_t* vp = kv_cache + phys_block * stride_block
                + 1 * stride_kv + bo * stride_slot + kv_head * stride_head;
            int no = MSE_BYTES + QJL_BYTES;
            uint16_t vvn = vp[no] | (vp[no+1] << 8);
            uint16_t vrn = vp[no+2] | (vp[no+3] << 8);
            float v_vn = __half2float(*(const __half*)&vvn);
            float v_rn = __half2float(*(const __half*)&vrn);

            // MSE part: Clifford inverse for this dim's group
            int g = tid / 3;
            int g_off = tid % 3;
            float c0 = s_centroids[s_v_idx[g*3]];
            float c1 = (g*3+1 < D) ? s_centroids[s_v_idx[g*3+1]] : 0.0f;
            float c2 = (g*3+2 < D) ? s_centroids[s_v_idx[g*3+2]] : 0.0f;
            float o0, o1, o2;
            clifford_inverse_group(c0, c1, c2, &rotors[g*8], o0, o1, o2);
            float v_mse = (g_off == 0) ? o0 : (g_off == 1) ? o1 : o2;

            // QJL part: S GEMV (same as TQ)
            float v_qjl = 0.0f;
            for (int j = 0; j < D; j++) {
                v_qjl += S[j * D + tid] * s_v_sign[j];
            }

            float v_recon = v_vn * (v_mse + correction_scale * v_rn * v_qjl);
            v_acc += exp_cur * v_recon;
        }

        m_prev = m_new;
        d_prev = d_new;
        __syncthreads();
    }

    if (tid < D && d_prev > 0.0f) {
        output[q_base + tid] = v_acc / d_prev;
    }
}


void rq_fused_decode_attention(
    torch::Tensor q_rot, torch::Tensor q_proj,
    torch::Tensor kv_cache,
    torch::Tensor rotors, torch::Tensor S, torch::Tensor centroids,
    torch::Tensor block_table, torch::Tensor seq_lens,
    torch::Tensor output,
    int head_dim, int mse_bits, int n_centroids, float attn_scale,
    int stride_block, int stride_kv, int stride_slot, int stride_head
) {
    int num_q = q_rot.size(0);
    int num_q_heads = q_rot.size(1);
    int num_kv_heads = kv_cache.size(3);
    int block_size = kv_cache.size(2);
    int packed_size = kv_cache.size(4);
    int max_blocks = block_table.size(1);
    float correction = sqrtf(M_PI_2) / (float)head_dim;

    int grid = num_q * num_q_heads;
    int threads = ((head_dim + 31) / 32) * 32;

    #define LAUNCH(HD, MB, NC) \
        rq_fused_decode_kernel<HD, MB, NC><<<grid, threads>>>( \
            q_rot.data_ptr<float>(), q_proj.data_ptr<float>(), \
            kv_cache.data_ptr<uint8_t>(), \
            rotors.data_ptr<float>(), S.data_ptr<float>(), \
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
    else { TORCH_CHECK(false, "RQ fused decode: unsupported config head_dim=",
                        head_dim, " mse_bits=", mse_bits); }
    #undef LAUNCH
}

}  // namespace rotorquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rq_fused_decode_attention", &rotorquant::rq_fused_decode_attention,
          "RQ fused decode with Clifford V decompression");
}
