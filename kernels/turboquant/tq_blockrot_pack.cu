// SPDX-License-Identifier: Apache-2.0
// TurboQuant: Fused Block-Rotation Pack Kernel
//
// Same as tq_wht_pack.cu but uses per-block random orthogonal rotation
// instead of fixed Walsh-Hadamard Transform.
//
// Grid: (N, D/32) — one warp per rotation block per vector
// Block: (32) — exactly one warp
//
// Same packed format as WHT: [qs(8) | qr(4) | gamma(2)] per 32 values = 14 bytes.
// Pi_blocks: [n_blocks, 32, 32] float32 rotation matrices.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

namespace turboquant {

#define WARP_SIZE 32

// 3-bit thresholds for N(0,1) Lloyd-Max quantization
__constant__ float BROT_THRESHOLDS_3BIT[7] = {
    -1.7479f, -1.0500f, -0.5005f, 0.0f, 0.5005f, 1.0500f, 1.7479f,
};

// Warp-level matrix-vector multiply via shuffles.
// Each thread holds one element of the input vector (val).
// Pi_row points to this thread's row of the 32x32 rotation matrix.
// Returns: output[lane] = sum_j Pi[lane][j] * input[j]
__device__ __forceinline__ float warp_matmul_rotate(
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

// Warp-reduce max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

// 3-bit threshold quantization: returns index 0-7
__device__ __forceinline__ int threshold_quantize_3bit(float x) {
    int idx = 0;
    if (x > BROT_THRESHOLDS_3BIT[0]) idx = 1;
    if (x > BROT_THRESHOLDS_3BIT[1]) idx = 2;
    if (x > BROT_THRESHOLDS_3BIT[2]) idx = 3;
    if (x > BROT_THRESHOLDS_3BIT[3]) idx = 4;
    if (x > BROT_THRESHOLDS_3BIT[4]) idx = 5;
    if (x > BROT_THRESHOLDS_3BIT[5]) idx = 6;
    if (x > BROT_THRESHOLDS_3BIT[6]) idx = 7;
    return idx;
}

// Fused Block-Rotation Pack Kernel (3-bit)
// Input: [N, D] bfloat16, Output: [N, packed_size] uint8
// Pi_blocks: [n_blocks, 32, 32] float32
template <int HEAD_DIM, int BLOCK_SIZE = 32>
__global__ void tq_blockrot_pack_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ Pi_blocks,  // [n_blocks, 32, 32]
    uint8_t* __restrict__ output,
    int N,
    int packed_size
) {
    constexpr int BYTES_PER_BLOCK = 14;  // 8 qs + 4 qr + 2 gamma

    int vec_idx = blockIdx.x;
    int rot_block = blockIdx.y;
    int lane = threadIdx.x;

    if (vec_idx >= N) return;

    // 1. Load bfloat16 input → float
    float val = __bfloat162float(input[vec_idx * HEAD_DIM + rot_block * BLOCK_SIZE + lane]);

    // 2. Block rotation forward: val = Pi_row . input_vector
    // Load this thread's row of the rotation matrix
    const float* Pi_row = Pi_blocks + rot_block * BLOCK_SIZE * BLOCK_SIZE + lane * BLOCK_SIZE;
    val = warp_matmul_rotate(val, Pi_row);

    // 3. Per-block amax (warp reduce)
    float amax = warp_reduce_max(fabsf(val));
    amax = fmaxf(amax, 1e-10f);

    // 4. Gamma + normalize + quantize
    float gamma = amax / 2.1519f;
    float normalized = val / gamma;
    int idx = threshold_quantize_3bit(normalized);

    // 5. Cooperative bitpack (identical to WHT pack)
    uint8_t* out = output + vec_idx * packed_size + rot_block * BYTES_PER_BLOCK;

    // 5a. Lower 2 bits → qs[8 bytes]: 4 indices per byte
    int low2 = idx & 0x3;
    {
        int base = (lane / 4) * 4;
        int b0 = __shfl_sync(0xffffffff, low2, base);
        int b1 = __shfl_sync(0xffffffff, low2, base + 1);
        int b2 = __shfl_sync(0xffffffff, low2, base + 2);
        int b3 = __shfl_sync(0xffffffff, low2, base + 3);
        if (lane % 4 == 0) {
            out[lane / 4] = (uint8_t)(b0 | (b1 << 2) | (b2 << 4) | (b3 << 6));
        }
    }

    // 5b. Upper 1 bit → qr[4 bytes]: 8 bits per byte
    int hi1 = (idx >> 2) & 1;
    {
        int base = (lane / 8) * 8;
        int byte_val = 0;
        for (int k = 0; k < 8; k++)
            byte_val |= (__shfl_sync(0xffffffff, hi1, base + k) << k);
        if (lane % 8 == 0) {
            out[8 + lane / 8] = (uint8_t)(byte_val & 0xFF);
        }
    }

    // 5c. Gamma as FP16 (2 bytes) — lane 0 writes
    if (lane == 0) {
        __half gamma_h = __float2half(gamma);
        uint16_t gamma_u16 = *reinterpret_cast<uint16_t*>(&gamma_h);
        out[12] = (uint8_t)(gamma_u16 & 0xFF);
        out[13] = (uint8_t)((gamma_u16 >> 8) & 0xFF);
    }
}


// Fused Block-Rotation Pack + KV-Cache Write
template <int HEAD_DIM, int BLOCK_SIZE = 32>
__global__ void tq_blockrot_pack_to_cache_kernel(
    const __nv_bfloat16* __restrict__ input,  // [num_tokens, num_heads, D] bf16
    const float* __restrict__ Pi_blocks,       // [n_blocks, 32, 32] float32
    uint8_t* __restrict__ kv_cache,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int kv_idx,
    int block_size_kv,
    int packed_size,
    int stride_block,
    int stride_kv,
    int stride_slot,
    int stride_head
) {
    constexpr int BYTES_PER_BLOCK = 14;

    int token_idx = blockIdx.x;
    int rot_block = blockIdx.y;
    int lane = threadIdx.x;

    if (token_idx >= num_tokens) return;
    int slot = slot_mapping[token_idx];
    if (slot < 0) return;

    int bi = slot / block_size_kv;
    int bo = slot % block_size_kv;

    const float* Pi_row = Pi_blocks + rot_block * BLOCK_SIZE * BLOCK_SIZE + lane * BLOCK_SIZE;

    for (int head = 0; head < num_heads; head++) {
        float val = __bfloat162float(
            input[(token_idx * num_heads + head) * HEAD_DIM
                  + rot_block * BLOCK_SIZE + lane]);

        val = warp_matmul_rotate(val, Pi_row);

        float amax = warp_reduce_max(fabsf(val));
        amax = fmaxf(amax, 1e-10f);
        float gamma = amax / 2.1519f;
        float normalized = val / gamma;
        int idx = threshold_quantize_3bit(normalized);

        uint8_t* out = kv_cache
            + bi * stride_block + kv_idx * stride_kv
            + bo * stride_slot + head * stride_head
            + rot_block * BYTES_PER_BLOCK;

        int low2 = idx & 0x3;
        {
            int base = (lane / 4) * 4;
            int b0 = __shfl_sync(0xffffffff, low2, base);
            int b1 = __shfl_sync(0xffffffff, low2, base + 1);
            int b2 = __shfl_sync(0xffffffff, low2, base + 2);
            int b3 = __shfl_sync(0xffffffff, low2, base + 3);
            if (lane % 4 == 0)
                out[lane / 4] = (uint8_t)(b0 | (b1 << 2) | (b2 << 4) | (b3 << 6));
        }
        int hi1 = (idx >> 2) & 1;
        {
            int base = (lane / 8) * 8;
            int byte_val = 0;
            for (int k = 0; k < 8; k++)
                byte_val |= (__shfl_sync(0xffffffff, hi1, base + k) << k);
            if (lane % 8 == 0)
                out[8 + lane / 8] = (uint8_t)(byte_val & 0xFF);
        }
        if (lane == 0) {
            __half gamma_h = __float2half(gamma);
            uint16_t gamma_u16 = *reinterpret_cast<uint16_t*>(&gamma_h);
            out[12] = (uint8_t)(gamma_u16 & 0xFF);
            out[13] = (uint8_t)((gamma_u16 >> 8) & 0xFF);
        }
    }
}


// C++ wrapper — original (pack to buffer)
void tq_blockrot_pack(
    torch::Tensor input,      // [N, D] bfloat16
    torch::Tensor Pi_blocks,  // [n_blocks, 32, 32] float32
    torch::Tensor output      // [N, packed_size] uint8 (pre-allocated)
) {
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(Pi_blocks.dim() == 3, "Pi_blocks must be 3D [n_blocks, 32, 32]");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(Pi_blocks.dtype() == torch::kFloat32, "Pi_blocks must be float32");
    TORCH_CHECK(output.dtype() == torch::kUInt8, "output must be uint8");

    int N = input.size(0);
    int D = input.size(1);
    int packed_size = output.size(1);

    dim3 grid(N, D / 32);
    dim3 block(32);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    #define BROT_PACK_LAUNCH(HD) \
        tq_blockrot_pack_kernel<HD><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), \
            Pi_blocks.data_ptr<float>(), \
            output.data_ptr<uint8_t>(), N, packed_size)

    if (D == 256) { BROT_PACK_LAUNCH(256); }
    else if (D == 128) { BROT_PACK_LAUNCH(128); }
    else if (D == 64) { BROT_PACK_LAUNCH(64); }
    else if (D == 512) { BROT_PACK_LAUNCH(512); }
    else { TORCH_CHECK(false, "tq_blockrot_pack: unsupported D=", D); }
    #undef BROT_PACK_LAUNCH
}

// C++ wrapper — fused pack + KV cache write
void tq_blockrot_pack_to_cache(
    torch::Tensor input,         // [num_tokens, num_heads, D] bfloat16
    torch::Tensor Pi_blocks,     // [n_blocks, 32, 32] float32
    torch::Tensor kv_cache,
    torch::Tensor slot_mapping,  // [num_tokens] int32
    int kv_idx,
    int stride_block, int stride_kv, int stride_slot, int stride_head
) {
    TORCH_CHECK(input.dim() == 3, "input must be 3D [tokens, heads, D]");
    int num_tokens = input.size(0);
    int num_heads = input.size(1);
    int D = input.size(2);
    int block_size_kv = kv_cache.size(2);
    int packed_size = (D / 32) * 14;

    dim3 grid(num_tokens, D / 32);
    dim3 block(32);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    #define BROT_PC_LAUNCH(HD) \
        tq_blockrot_pack_to_cache_kernel<HD><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), \
            Pi_blocks.data_ptr<float>(), \
            kv_cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int>(), \
            num_tokens, num_heads, kv_idx, block_size_kv, packed_size, \
            stride_block, stride_kv, stride_slot, stride_head)

    if (D == 256) { BROT_PC_LAUNCH(256); }
    else if (D == 128) { BROT_PC_LAUNCH(128); }
    else if (D == 64) { BROT_PC_LAUNCH(64); }
    else if (D == 512) { BROT_PC_LAUNCH(512); }
    else { TORCH_CHECK(false, "tq_blockrot_pack_to_cache: unsupported D=", D); }
    #undef BROT_PC_LAUNCH
}

}  // namespace turboquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq_blockrot_pack", &turboquant::tq_blockrot_pack,
          "Fused block-rotation pack: bf16 → rotate → quantize → bitpack → uint8");
    m.def("tq_blockrot_pack_to_cache", &turboquant::tq_blockrot_pack_to_cache,
          "Fused block-rotation pack + direct KV cache write");
}
