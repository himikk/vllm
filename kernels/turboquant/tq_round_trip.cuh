// SPDX-License-Identifier: Apache-2.0
// TurboQuant fused round-trip kernel header.
//
// Fuses normalize → rotate → quantize → reconstruct → residual → QJL
// into a single kernel launch. Replaces 10 PyTorch kernel launches.
#pragma once

#include <torch/extension.h>
#include <cuda_fp16.h>

namespace turboquant {

// Warp-level sum reduction using shuffle.
__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

// Block-level sum reduction: each warp reduces, then warp 0 combines.
__inline__ __device__ float block_reduce_sum(float val, float* shared,
                                              int tid, int num_threads) {
  int warp_id = tid / 32;
  int lane = tid % 32;
  val = warp_reduce_sum(val);
  if (lane == 0) shared[warp_id] = val;
  __syncthreads();
  if (warp_id == 0) {
    val = (lane < (num_threads / 32)) ? shared[lane] : 0.0f;
    val = warp_reduce_sum(val);
  }
  __syncthreads();
  return val;
}

void turboquant_round_trip(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size] input/output
    torch::Tensor& Pi,         // [head_size, head_size] rotation matrix
    torch::Tensor& S,          // [head_size, head_size] QJL projection
    torch::Tensor& centroids,  // [n_centroids] Lloyd-Max centroids
    torch::Tensor& output,     // [num_tokens, num_heads, head_size] output
    int head_size,
    int n_centroids);

}  // namespace turboquant
