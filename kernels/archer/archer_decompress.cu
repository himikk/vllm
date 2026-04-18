// SPDX-License-Identifier: Apache-2.0
// Archer decompress kernel — unpack TQ3/TQ4 packed weights to BF16/FP16.
//
// Decompresses packed uint8 weight rows back to float for GEMM.
// This is the INVERSE of TurboQuant compression:
//   1. Unpack MSE indices from bit-packed uint8
//   2. Centroid lookup → quantized values
//   3. Inverse rotation: Pi^T @ quantized (tiled GEMV)
//   4. QJL correction: sqrt(π/2)/D * res_norm * (signs @ S)
//   5. Scale by row_norm
//
// Grid:  (num_rows,)  — one block per weight row
// Block: (BLOCK_SIZE,) — threads cooperatively decompress one row
//
// Output: W_out[row, 0..D-1] = row_norm * (Pi^T @ centroids[idx] + QJL_correction)

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cmath>

namespace archer {

constexpr int BLOCK_SIZE = 256;
constexpr int TILE_K = 32;

// ── Type helpers (reuse from TQ) ────────────────────────────────────────
template <typename T>
__device__ __forceinline__ float to_float(T val);
template <> __device__ __forceinline__ float to_float<c10::Half>(c10::Half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }

template <typename T>
__device__ __forceinline__ T from_float(float val);
template <> __device__ __forceinline__ c10::Half from_float<c10::Half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float v) { return static_cast<at::BFloat16>(v); }
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_float<double>(double v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ double from_float<double>(float v) { return static_cast<double>(v); }

// ── Tiled column-access GEMV: out[tid] = Σ_j M[j*D + tid] * vec[j] ────
// This computes Pi^T @ x = (transpose of Pi) @ x
// which is equivalent to reading Pi column-wise.
template <int D>
__device__ __forceinline__ float tiled_gemv_col(
    const float* __restrict__ M,
    const float* __restrict__ s_vec,
    float* __restrict__ s_tile,
    int tid
) {
    float acc = 0.0f;
    for (int t = 0; t < D; t += TILE_K) {
        if (tid < TILE_K && (t + tid) < D) {
            s_tile[tid] = s_vec[t + tid];
        }
        __syncthreads();
        if (tid < D) {
            int kmax = min(TILE_K, D - t);
            #pragma unroll 8
            for (int k = 0; k < kmax; k++) {
                acc += M[(t + k) * D + tid] * s_tile[k];
            }
        }
        __syncthreads();
    }
    return acc;
}

// ═══════════════════════════════════════════════════════════════════════
// Main Archer decompress kernel
// ═══════════════════════════════════════════════════════════════════════
template <typename scalar_t, int HEAD_SIZE, int N_CENTROIDS, int MSE_BITS>
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
archer_decompress_kernel(
    const uint8_t* __restrict__ packed_W,  // [num_rows, packed_size]
    scalar_t* __restrict__ W_out,          // [num_rows, D]
    const float* __restrict__ Pi,          // [D, D] rotation matrix
    const float* __restrict__ S,           // [D, D] QJL projection
    const float* __restrict__ centroids,   // [N_CENTROIDS]
    int num_rows,
    int packed_size
) {
    const int row = blockIdx.x;
    if (row >= num_rows) return;
    const int tid = threadIdx.x;
    constexpr int D = HEAD_SIZE;
    constexpr int MSE_BYTES = (D * MSE_BITS + 7) / 8;
    constexpr int QJL_BYTES = (D + 7) / 8;
    constexpr int MASK = (1 << MSE_BITS) - 1;

    // ── Shared memory layout ─────────────────────────────────────────
    //   [0..D-1]           s_quantized: centroid values after lookup
    //   [D..2D-1]          s_work: for inverse rotation result
    //   [2D..2D+TK-1]      s_tile: GEMV tile
    //   [2D+TK..2D+TK+NC]  s_centroids
    //   [2D+TK+NC..2D+TK+NC+D] s_signs: QJL signs
    extern __shared__ float smem[];
    float* s_quantized = smem;
    float* s_work      = smem + D;
    float* s_tile      = smem + 2 * D;
    float* s_centroids = smem + 2 * D + TILE_K;
    float* s_signs     = smem + 2 * D + TILE_K + N_CENTROIDS;

    const uint8_t* row_packed = packed_W + row * packed_size;

    // Load centroids to shared
    if (tid < N_CENTROIDS) {
        s_centroids[tid] = centroids[tid];
    }
    __syncthreads();

    // ── Step 1: Unpack MSE indices → centroid values ─────────────────
    // Bitstream unpack: handles non-power-of-2 bit widths (e.g. 3-bit)
    for (int j = tid; j < D; j += BLOCK_SIZE) {
        int bit_offset = j * MSE_BITS;
        int byte_idx = bit_offset / 8;
        int bit_shift = bit_offset % 8;
        int val = row_packed[byte_idx] >> bit_shift;
        int spill = bit_shift + MSE_BITS - 8;
        if (spill > 0 && byte_idx + 1 < MSE_BYTES) {
            val |= row_packed[byte_idx + 1] << (MSE_BITS - spill);
        }
        int idx = val & MASK;
        s_quantized[j] = s_centroids[idx];
    }
    __syncthreads();

    // ── Step 2: Inverse rotation: Pi^T @ s_quantized → s_work ───────
    if (tid < D) {
        s_work[tid] = tiled_gemv_col<D>(Pi, s_quantized, s_tile, tid);
    }
    __syncthreads();

    // ── Step 3: Unpack QJL signs ─────────────────────────────────────
    for (int j = tid; j < D; j += BLOCK_SIZE) {
        int byte_idx = MSE_BYTES + j / 8;
        int bit_pos = j % 8;
        uint8_t sign_byte = row_packed[byte_idx];
        s_signs[j] = ((sign_byte >> bit_pos) & 1) ? 1.0f : -1.0f;
    }
    __syncthreads();

    // ── Step 4: QJL correction: signs @ S → then scale ──────────────
    // correction[tid] = Σ_j signs[j] * S[j*D + tid]
    float qjl_acc = 0.0f;
    if (tid < D) {
        for (int t = 0; t < D; t += TILE_K) {
            // Load signs chunk into tile
            if (tid < TILE_K && (t + tid) < D) {
                s_tile[tid] = s_signs[t + tid];
            }
            __syncthreads();
            int kmax = min(TILE_K, D - t);
            #pragma unroll 8
            for (int k = 0; k < kmax; k++) {
                qjl_acc += s_tile[k] * S[(t + k) * D + tid];
            }
            __syncthreads();
        }
    }

    // ── Step 5: Unpack norms and write output ────────────────────────
    // Norms are at offset MSE_BYTES + QJL_BYTES
    __shared__ float s_row_norm, s_res_norm;
    if (tid == 0) {
        int norm_offset = MSE_BYTES + QJL_BYTES;
        // float16 norms: 2 bytes each
        uint16_t vn_bits = row_packed[norm_offset] | (row_packed[norm_offset + 1] << 8);
        uint16_t rn_bits = row_packed[norm_offset + 2] | (row_packed[norm_offset + 3] << 8);
        s_row_norm = __half2float(*reinterpret_cast<__half*>(&vn_bits));
        s_res_norm = __half2float(*reinterpret_cast<__half*>(&rn_bits));
    }
    __syncthreads();

    // Combine: row_norm * (x_mse + correction_scale * res_norm * qjl_correction)
    constexpr float CORR_SCALE = 1.2533141373155003f / D;  // sqrt(π/2) / D
    if (tid < D) {
        float mse_val = s_work[tid];
        float qjl_val = CORR_SCALE * s_res_norm * qjl_acc;
        float result = s_row_norm * (mse_val + qjl_val);
        W_out[row * D + tid] = from_float<scalar_t>(result);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Host launcher
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t, int D, int NC, int BITS>
void launch_decompress(
    const uint8_t* packed, scalar_t* out,
    const float* Pi, const float* S, const float* centroids,
    int num_rows, int packed_size, cudaStream_t stream
) {
    // Shared memory: 2*D + TILE_K + NC + D floats
    int smem = (3 * D + TILE_K + NC) * sizeof(float);
    archer_decompress_kernel<scalar_t, D, NC, BITS>
        <<<num_rows, BLOCK_SIZE, smem, stream>>>(
            packed, out, Pi, S, centroids, num_rows, packed_size
        );
}

// Torch entry point
void archer_decompress(
    torch::Tensor packed_W,    // [num_rows, packed_size] uint8
    torch::Tensor W_out,       // [num_rows, D] bf16/fp16/fp32
    torch::Tensor Pi,          // [D, D] float32
    torch::Tensor S,           // [D, D] float32
    torch::Tensor centroids,   // [NC] float32
    int head_size,
    int n_centroids
) {
    int num_rows = packed_W.size(0);
    int packed_size = packed_W.size(1);
    cudaStream_t stream = 0;  // default stream

    // Dispatch on dtype + head_size + n_centroids
    #define DISPATCH(DTYPE, D, NC, BITS) \
        launch_decompress<DTYPE, D, NC, BITS>( \
            packed_W.data_ptr<uint8_t>(), \
            W_out.data_ptr<DTYPE>(), \
            Pi.data_ptr<float>(), \
            S.data_ptr<float>(), \
            centroids.data_ptr<float>(), \
            num_rows, packed_size, stream)

    // Only BF16/FP16/FP32 — no double (avoids template instantiation issue)
    auto dtype = W_out.scalar_type();
    TORCH_CHECK(dtype == at::ScalarType::BFloat16 ||
                dtype == at::ScalarType::Half ||
                dtype == at::ScalarType::Float,
                "archer_decompress: only BF16/FP16/FP32 supported, got ", dtype);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dtype, "archer_decompress", [&] {
            // TQ3: MSE_BITS=2, N_CENTROIDS=4
            if (n_centroids == 4) {
                if (head_size == 128)       DISPATCH(scalar_t, 128, 4, 2);
                else if (head_size == 256)  DISPATCH(scalar_t, 256, 4, 2);
                else if (head_size == 512)  DISPATCH(scalar_t, 512, 4, 2);
                else if (head_size == 768)  DISPATCH(scalar_t, 768, 4, 2);
                else if (head_size == 1024) DISPATCH(scalar_t, 1024, 4, 2);
                else if (head_size == 1536) DISPATCH(scalar_t, 1536, 4, 2);
                else if (head_size == 2048) DISPATCH(scalar_t, 2048, 4, 2);
                else if (head_size == 4096) DISPATCH(scalar_t, 4096, 4, 2);
                else if (head_size == 5120) DISPATCH(scalar_t, 5120, 4, 2);
                else if (head_size == 10240) DISPATCH(scalar_t, 10240, 4, 2);
                else TORCH_CHECK(false, "Unsupported head_size for NC=4: ", head_size);
            }
            // TQ4: MSE_BITS=3, N_CENTROIDS=8
            else if (n_centroids == 8) {
                if (head_size == 128)       DISPATCH(scalar_t, 128, 8, 3);
                else if (head_size == 256)  DISPATCH(scalar_t, 256, 8, 3);
                else if (head_size == 512)  DISPATCH(scalar_t, 512, 8, 3);
                else if (head_size == 768)  DISPATCH(scalar_t, 768, 8, 3);
                else if (head_size == 1024) DISPATCH(scalar_t, 1024, 8, 3);
                else if (head_size == 1536) DISPATCH(scalar_t, 1536, 8, 3);
                else if (head_size == 2048) DISPATCH(scalar_t, 2048, 8, 3);
                else if (head_size == 4096) DISPATCH(scalar_t, 4096, 8, 3);
                else if (head_size == 5120) DISPATCH(scalar_t, 5120, 8, 3);
                else if (head_size == 10240) DISPATCH(scalar_t, 10240, 8, 3);
                else TORCH_CHECK(false, "Unsupported head_size for NC=8: ", head_size);
            }
            else {
                TORCH_CHECK(false, "Unsupported n_centroids: ", n_centroids);
            }
        }
    );

    #undef DISPATCH
}

}  // namespace archer

// ── JIT extension binding ───────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("archer_decompress", &archer::archer_decompress,
          "Archer decompress: packed uint8 → float (TQ inverse)");
}
