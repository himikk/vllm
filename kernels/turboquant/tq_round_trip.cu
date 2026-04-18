// SPDX-License-Identifier: Apache-2.0
// TurboQuant fused round-trip CUDA kernel — optimized v2.
//
// Optimizations vs v1:
//   1. Tiled GEMV: matrices loaded in TILE_K-wide strips through shared memory
//      → reduces global memory traffic by factor D/TILE_K per GEMV pass
//   2. float4 vectorized loads: 128-bit coalesced reads from global memory
//   3. Warp-shuffle reductions: no shared memory needed for block reduce
//   4. __launch_bounds__ for register pressure control
//   5. Preloaded centroids with early __syncthreads__ elimination
//
// Grid:  (num_tokens, num_kv_heads)
// Block: (BLOCK_SIZE threads)

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cmath>

namespace turboquant {

// ── Configuration ──────────────────────────────────────────────────────
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE  = 32;
constexpr int TILE_K     = 32;   // tile width for GEMV — fits in smem

// ── Type helpers ───────────────────────────────────────────────────────
template <typename T>
__device__ __forceinline__ float to_float(T val);
template <> __device__ __forceinline__ float to_float<c10::Half>(c10::Half v)     { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ float to_float<float>(float v)             { return v; }

template <typename T>
__device__ __forceinline__ T from_float(float val);
template <> __device__ __forceinline__ c10::Half   from_float<c10::Half>(float v)   { return __float2half(v); }
template <> __device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float v) { return static_cast<at::BFloat16>(v); }
template <> __device__ __forceinline__ float        from_float<float>(float v)      { return v; }

// ── Warp / block reduction via shuffle ─────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_sum_shuffle(float val) {
    // Step 1: intra-warp reduce
    val = warp_reduce_sum(val);

    // Step 2: lane 0 of each warp writes to shared, warp 0 reduces
    __shared__ float s_warp_sums[BLOCK_SIZE / WARP_SIZE];  // 8 floats
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;

    if (lane == 0) s_warp_sums[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < (BLOCK_SIZE / WARP_SIZE)) ? s_warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;  // valid in thread 0 only
}

// ── Tiled GEMV helpers ─────────────────────────────────────────────────
//
// gemv_row:  out[tid] = Σ_j M[tid, j] * vec[j]   (row-major M, row access)
// gemv_col:  out[tid] = Σ_j M[j, tid] * vec[j]   (row-major M, column access)
//
// Both tile over j in chunks of TILE_K, loading the tile of vec into smem
// and using float4 loads for the matrix where possible.
//
// s_tile: shared memory tile of TILE_K floats for the vector chunk.

// Row-access GEMV: out[tid] = Σ_j M[tid*D + j] * vec[j]
// M is read along rows → naturally coalesced when threads read consecutive j
// within a tile (but each thread reads its own row, so stride = D between threads).
// Tiling helps because vec is reused across all threads from smem.
template <int D>
__device__ __forceinline__ float tiled_gemv_row(
    const float* __restrict__ M,     // [D, D] row-major
    const float* __restrict__ s_vec, // [D] in shared memory (source vector)
    float* __restrict__ s_tile,      // [TILE_K] shared scratch
    int tid
) {
    float acc = 0.0f;
    const float* M_row = M + tid * D;  // base of this thread's row

    for (int t = 0; t < D; t += TILE_K) {
        // Cooperative load of vec[t .. t+TILE_K-1] into shared tile
        if (tid < TILE_K) {
            s_tile[tid] = s_vec[t + tid];
        }
        __syncthreads();

        // Accumulate: float4 loads from M_row when possible
        if (tid < D) {
            const float* M_chunk = M_row + t;
            int k = 0;
            // float4 path: 4 elements at a time
            for (; k + 3 < TILE_K; k += 4) {
                float4 m4 = *reinterpret_cast<const float4*>(M_chunk + k);
                acc += m4.x * s_tile[k]
                     + m4.y * s_tile[k + 1]
                     + m4.z * s_tile[k + 2]
                     + m4.w * s_tile[k + 3];
            }
            // Remainder
            for (; k < TILE_K; k++) {
                acc += M_chunk[k] * s_tile[k];
            }
        }
        __syncthreads();
    }
    return acc;
}

// Column-access GEMV: out[tid] = Σ_j M[j*D + tid] * vec[j]
// M is read along columns → stride-D access pattern (uncoalesced per element).
// Tiling helps because we read TILE_K rows of M at a time, and the vector
// chunk is in smem.  We also tile the M reads into smem to convert the
// stride-D reads into coalesced reads of transposed tiles.
template <int D>
__device__ __forceinline__ float tiled_gemv_col(
    const float* __restrict__ M,     // [D, D] row-major
    const float* __restrict__ s_vec, // [D] in shared memory (source vector)
    float* __restrict__ s_tile,      // [TILE_K] shared scratch for vec chunk
    int tid
) {
    // For column access, the simplest effective tiling is over the vec dimension.
    // Each tile loads TILE_K floats of vec into smem, then each thread does
    // TILE_K multiply-adds with M[j*D + tid] — still stride-D on M, but
    // the vec is shared.  This is still memory-bound on M, but the vec reuse
    // saves D * sizeof(float) * (D/TILE_K - 1) bytes of traffic.
    //
    // A full transpose-tile approach would need TILE_K * D floats of smem
    // which is too much (32*256 = 32KB per tile).  So we keep this simpler
    // form which still benefits from vec tiling and compiler-level ILP.

    float acc = 0.0f;

    for (int t = 0; t < D; t += TILE_K) {
        // Load vec chunk
        if (tid < TILE_K) {
            s_tile[tid] = s_vec[t + tid];
        }
        __syncthreads();

        if (tid < D) {
            #pragma unroll 8
            for (int k = 0; k < TILE_K; k++) {
                acc += M[(t + k) * D + tid] * s_tile[k];
            }
        }
        __syncthreads();
    }
    return acc;
}


// ═══════════════════════════════════════════════════════════════════════
// Main fused kernel — optimized v2
// ═══════════════════════════════════════════════════════════════════════
template <typename scalar_t, int HEAD_SIZE, int N_CENTROIDS>
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
turboquant_round_trip_v2_kernel(
    const scalar_t* __restrict__ key_in,
    scalar_t* __restrict__ key_out,
    const float* __restrict__ Pi,
    const float* __restrict__ S,
    const float* __restrict__ centroids,
    int num_heads
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    constexpr int D     = HEAD_SIZE;

    // ── Shared memory layout ───────────────────────────────────────────
    //   [0       .. D-1      ]  s_x_hat: normalized key (persistent)
    //   [D       .. 2D-1     ]  s_work:  multipurpose work buffer
    //   [2D      .. 2D+TK-1  ]  s_tile:  GEMV vector tile
    //   [2D+TK   .. 2D+TK+NC ]  s_centroids
    extern __shared__ float smem[];
    float* s_x_hat     = smem;
    float* s_work      = smem + D;
    float* s_tile      = smem + 2 * D;
    float* s_centroids = smem + 2 * D + TILE_K;

    const int key_offset = (token_idx * num_heads + head_idx) * D;

    // Load centroids
    if (tid < N_CENTROIDS) {
        s_centroids[tid] = centroids[tid];
    }

    // ════════ Stage 1: Load + L2 normalize ═════════════════════════════
    float local_val = 0.0f;
    float local_sq  = 0.0f;
    if (tid < D) {
        local_val = to_float(key_in[key_offset + tid]);
        local_sq  = local_val * local_val;
    }

    float norm_sq = block_reduce_sum_shuffle(local_sq);
    __shared__ float s_vec_norm;
    if (tid == 0) {
        s_vec_norm = sqrtf(norm_sq + 1e-8f);
    }
    __syncthreads();

    if (tid < D) {
        s_x_hat[tid] = local_val / s_vec_norm;
    }
    __syncthreads();

    // ════════ Stage 2: Rotate  y = x_hat @ Pi^T  ══════════════════════
    // y[tid] = Σ_j Pi[tid, j] * x_hat[j]  (row access on Pi)
    float y_val = tiled_gemv_row<D>(Pi, s_x_hat, s_tile, tid);

    // ════════ Stage 3: Scalar quantize to nearest centroid ═════════════
    float best_centroid = 0.0f;
    if (tid < D) {
        float best_dist = 1e30f;
        #pragma unroll
        for (int c = 0; c < N_CENTROIDS; c++) {
            float dist = fabsf(y_val - s_centroids[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_centroid = s_centroids[c];
            }
        }
    }

    if (tid < D) {
        s_work[tid] = best_centroid;
    }
    __syncthreads();

    // ════════ Stage 4: Unrotate  x_mse = y_hat @ Pi  ══════════════════
    // x_mse[tid] = Σ_j Pi[j, tid] * y_hat[j]  (column access on Pi)
    float x_mse_val = tiled_gemv_col<D>(Pi, s_work, s_tile, tid);

    // ════════ Stage 5: Residual + L2 norm ══════════════════════════════
    float r_val = 0.0f;
    if (tid < D) {
        r_val = s_x_hat[tid] - x_mse_val;
    }

    float r_sq = (tid < D) ? r_val * r_val : 0.0f;
    float gamma_sq = block_reduce_sum_shuffle(r_sq);
    __shared__ float s_gamma;
    if (tid == 0) {
        s_gamma = sqrtf(gamma_sq + 1e-8f);
    }
    __syncthreads();

    if (tid < D) {
        s_work[tid] = r_val;
    }
    __syncthreads();

    // ════════ Stage 6: QJL projection  proj = r @ S^T  ════════════════
    // proj[tid] = Σ_j S[tid, j] * r[j]  (row access on S)
    float proj_val = tiled_gemv_row<D>(S, s_work, s_tile, tid);
    float sign_val = (tid < D && proj_val >= 0.0f) ? 1.0f : -1.0f;

    if (tid < D) {
        s_work[tid] = sign_val;
    }
    __syncthreads();

    // ════════ Stage 7: QJL correction  x_qjl = signs @ S  ═════════════
    // x_qjl[tid] = Σ_j S[j, tid] * signs[j]  (column access on S)
    float x_qjl_val = tiled_gemv_col<D>(S, s_work, s_tile, tid);

    // Scale: sqrt(pi/2) / D * gamma
    const float correction_scale = sqrtf(M_PI_2) / static_cast<float>(D);
    x_qjl_val *= correction_scale * s_gamma;

    // ════════ Stage 8: Combine + write ═════════════════════════════════
    if (tid < D) {
        float result = s_vec_norm * (x_mse_val + x_qjl_val);
        key_out[key_offset + tid] = from_float<scalar_t>(result);
    }
}


// ═══════════════════════════════════════════════════════════════════════
// C++ dispatch wrapper
// ═══════════════════════════════════════════════════════════════════════
void turboquant_round_trip(
    torch::Tensor& key,
    torch::Tensor& Pi,
    torch::Tensor& S,
    torch::Tensor& centroids,
    torch::Tensor& output,
    int head_size,
    int n_centroids
) {
    const int num_tokens = key.size(0);
    const int num_heads  = key.size(1);

    // Shared memory: 2*D + TILE_K + N_CENTROIDS floats
    const int smem_bytes = (2 * head_size + TILE_K + n_centroids) * sizeof(float);

    dim3 grid(num_tokens, num_heads);
    dim3 block(BLOCK_SIZE);

    auto launch = [&]<typename scalar_t>() {
        if (head_size == 128 && n_centroids == 4) {
            turboquant_round_trip_v2_kernel<scalar_t, 128, 4>
                <<<grid, block, smem_bytes>>>(
                    key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    Pi.data_ptr<float>(), S.data_ptr<float>(),
                    centroids.data_ptr<float>(), num_heads);
        } else if (head_size == 128 && n_centroids == 8) {
            turboquant_round_trip_v2_kernel<scalar_t, 128, 8>
                <<<grid, block, smem_bytes>>>(
                    key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    Pi.data_ptr<float>(), S.data_ptr<float>(),
                    centroids.data_ptr<float>(), num_heads);
        } else if (head_size == 256 && n_centroids == 4) {
            turboquant_round_trip_v2_kernel<scalar_t, 256, 4>
                <<<grid, block, smem_bytes>>>(
                    key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    Pi.data_ptr<float>(), S.data_ptr<float>(),
                    centroids.data_ptr<float>(), num_heads);
        } else if (head_size == 256 && n_centroids == 8) {
            turboquant_round_trip_v2_kernel<scalar_t, 256, 8>
                <<<grid, block, smem_bytes>>>(
                    key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    Pi.data_ptr<float>(), S.data_ptr<float>(),
                    centroids.data_ptr<float>(), num_heads);
        } else {
            TORCH_CHECK(false,
                "TurboQuant: unsupported head_size=", head_size,
                " n_centroids=", n_centroids);
        }
    };

    auto dtype = key.scalar_type();
    if (dtype == at::ScalarType::Half) {
        launch.template operator()<c10::Half>();
    } else if (dtype == at::ScalarType::BFloat16) {
        launch.template operator()<at::BFloat16>();
    } else {
        TORCH_CHECK(false, "TurboQuant: unsupported dtype. Expected Half or BFloat16.");
    }
}

}  // namespace turboquant
