// SPDX-License-Identifier: Apache-2.0
// TurboQuant v2: Fused WHT Pack Kernel
//
// Single kernel: raw float vector → WHT → amax → quantize → bitpack → uint8
// Replaces ~25 Python tensor-op launches with 1 CUDA launch.
//
// Grid: (N, D/32) — one warp per WHT block per vector
// Block: (32) — exactly one warp
//
// Same WHT transform as tq_wht_decode.cu (warp_wht_forward).
// Same packed format: [qs(8) | qr(4) | gamma(2)] per 32 values = 14 bytes.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

namespace turboquant {

#define WARP_SIZE 32
#define WHT_STAGES 5

// Fixed sign-flip pattern (must match Python wht.py and tq_wht_decode.cu)
__constant__ float PACK_WHT_SIGNS[32] = {
    +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
    +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1,
};

// 3-bit thresholds for N(0,1) Lloyd-Max quantization
__constant__ float WHT_THRESHOLDS_3BIT[7] = {
    -1.7479f, -1.0500f, -0.5005f, 0.0f, 0.5005f, 1.0500f, 1.7479f,
};

// WHT forward via warp shuffles (identical to decode kernel)
__device__ __forceinline__ float warp_wht_forward(float val, int lane) {
    val *= PACK_WHT_SIGNS[lane];
    for (int stage = 0; stage < WHT_STAGES; stage++) {
        int mask = 1 << stage;
        float other = __shfl_xor_sync(0xffffffff, val, mask);
        if (lane & mask)
            val = other - val;
        else
            val = val + other;
    }
    val *= 0.17677669529663688f;  // 1/sqrt(32)
    return val;
}

// Warp-reduce max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

// 2-bit thresholds and outermost centroid
__constant__ float WHT_THRESHOLDS_2BIT[3] = {-0.9816f, 0.0f, 0.9816f};
static constexpr float WHT_OUTERMOST_2BIT = 1.5104f;

// 4-bit thresholds for N(0,1) Lloyd-Max (15 boundaries)
__constant__ float WHT_THRESHOLDS_4BIT[15] = {
    -2.4008f, -1.8436f, -1.4372f, -1.0993f, -0.7996f, -0.5224f, -0.2582f,
     0.0f,     0.2582f,  0.5224f,  0.7996f,  1.0993f,  1.4372f,  1.8436f,
     2.4008f,
};

// 2-bit threshold quantization: returns index 0-3
__device__ __forceinline__ int threshold_quantize_2bit(float x) {
    int idx = 0;
    if (x > WHT_THRESHOLDS_2BIT[0]) idx = 1;
    if (x > WHT_THRESHOLDS_2BIT[1]) idx = 2;
    if (x > WHT_THRESHOLDS_2BIT[2]) idx = 3;
    return idx;
}

// 3-bit threshold quantization: returns index 0-7
__device__ __forceinline__ int threshold_quantize_3bit(float x) {
    int idx = 0;
    if (x > WHT_THRESHOLDS_3BIT[0]) idx = 1;
    if (x > WHT_THRESHOLDS_3BIT[1]) idx = 2;
    if (x > WHT_THRESHOLDS_3BIT[2]) idx = 3;
    if (x > WHT_THRESHOLDS_3BIT[3]) idx = 4;
    if (x > WHT_THRESHOLDS_3BIT[4]) idx = 5;
    if (x > WHT_THRESHOLDS_3BIT[5]) idx = 6;
    if (x > WHT_THRESHOLDS_3BIT[6]) idx = 7;
    return idx;
}

// 4-bit threshold quantization: returns index 0-15
__device__ __forceinline__ int threshold_quantize_4bit(float x) {
    int idx = 0;
    #pragma unroll
    for (int i = 0; i < 15; i++) {
        if (x > WHT_THRESHOLDS_4BIT[i]) idx = i + 1;
    }
    return idx;
}

// Fused WHT Pack Kernel (3-bit)
// Input: [N, D] float32, Output: [N, packed_size] uint8
// packed_size = (D/32) * 14 bytes
template <int HEAD_DIM, int BLOCK_SIZE = 32>
__global__ void tq_wht_pack_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    int N,
    int packed_size
) {
    constexpr int BYTES_PER_BLOCK = 14;  // 8 qs + 4 qr + 2 gamma

    int vec_idx = blockIdx.x;
    int wht_block = blockIdx.y;
    int lane = threadIdx.x;

    if (vec_idx >= N) return;

    // 1. Load bfloat16 input → float
    float val = __bfloat162float(input[vec_idx * HEAD_DIM + wht_block * BLOCK_SIZE + lane]);

    // 2. WHT forward
    val = warp_wht_forward(val, lane);

    // 3. Per-block amax (warp reduce)
    float amax = warp_reduce_max(fabsf(val));
    amax = fmaxf(amax, 1e-10f);

    // 4. Gamma + normalize + quantize
    float gamma = amax / 2.1519f;
    float normalized = val / gamma;
    int idx = threshold_quantize_3bit(normalized);

    // 5. Cooperative bitpack
    // Output offset for this WHT block
    uint8_t* out = output + vec_idx * packed_size + wht_block * BYTES_PER_BLOCK;

    // 5a. Lower 2 bits → qs[8 bytes]: 4 indices per byte
    int low2 = idx & 0x3;
    // All lanes read all low2 values; only group leaders write
    {
        int base = (lane / 4) * 4;  // group start lane
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


// Fused WHT Pack + KV-Cache Write Kernel
// Same quantization as above, but writes directly to KV cache via slot_mapping.
// Eliminates the separate scatter tensor op.
// Grid: (num_tokens, D/32), each token processes all its heads sequentially.
template <int HEAD_DIM, int MSE_BITS, int BLOCK_SIZE = 32>
__global__ void tq_wht_pack_to_cache_kernel(
    const __nv_bfloat16* __restrict__ input,  // [num_tokens, num_heads, D] bf16
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
    constexpr int BYTES_PER_BLOCK = (MSE_BITS == 2) ? 10 : (MSE_BITS == 3) ? 14 : 18;
    constexpr float OUTERMOST = (MSE_BITS == 2) ? 1.5104f : (MSE_BITS == 3) ? 2.1519f : 2.7326f;

    int token_idx = blockIdx.x;
    int wht_block = blockIdx.y;
    int lane = threadIdx.x;

    if (token_idx >= num_tokens) return;
    int slot = slot_mapping[token_idx];
    if (slot < 0) return;

    int bi = slot / block_size_kv;
    int bo = slot % block_size_kv;

    for (int head = 0; head < num_heads; head++) {
        float val = __bfloat162float(
            input[(token_idx * num_heads + head) * HEAD_DIM
                  + wht_block * BLOCK_SIZE + lane]);

        val = warp_wht_forward(val, lane);

        float amax = warp_reduce_max(fabsf(val));
        amax = fmaxf(amax, 1e-10f);

        float gamma = amax / OUTERMOST;
        float normalized = val / gamma;

        int idx;
        if constexpr (MSE_BITS == 2) {
            idx = threshold_quantize_2bit(normalized);
        } else if constexpr (MSE_BITS == 3) {
            idx = threshold_quantize_3bit(normalized);
        } else {
            idx = threshold_quantize_4bit(normalized);
        }

        uint8_t* out = kv_cache
            + bi * stride_block + kv_idx * stride_kv
            + bo * stride_slot + head * stride_head
            + wht_block * BYTES_PER_BLOCK;

        if constexpr (MSE_BITS == 2) {
            // 2-bit: 4 indices per byte → 8 bytes for 32 values + 2 gamma
            int low2 = idx & 0x3;
            int base = (lane / 4) * 4;
            int b0 = __shfl_sync(0xffffffff, low2, base);
            int b1 = __shfl_sync(0xffffffff, low2, base + 1);
            int b2 = __shfl_sync(0xffffffff, low2, base + 2);
            int b3 = __shfl_sync(0xffffffff, low2, base + 3);
            if (lane % 4 == 0)
                out[lane / 4] = (uint8_t)(b0 | (b1 << 2) | (b2 << 4) | (b3 << 6));
            if (lane == 0) {
                __half gamma_h = __float2half(gamma);
                uint16_t gamma_u16 = *reinterpret_cast<uint16_t*>(&gamma_h);
                out[8] = (uint8_t)(gamma_u16 & 0xFF);
                out[9] = (uint8_t)((gamma_u16 >> 8) & 0xFF);
            }
        } else if constexpr (MSE_BITS == 3) {
            // 3-bit: qs[8] + qr[4] + gamma[2] = 14 bytes
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
        } else if constexpr (MSE_BITS == 4) {
            // 4-bit: 16 bytes (2 indices per byte) + 2 gamma = 18 bytes
            // Lane k writes to byte[k/2], nibble (k%2)*4 (matches decode layout)
            int nibble = idx & 0xF;
            {
                int base = (lane / 2) * 2;
                int n0 = __shfl_sync(0xffffffff, nibble, base);
                int n1 = __shfl_sync(0xffffffff, nibble, base + 1);
                if (lane % 2 == 0)
                    out[lane / 2] = (uint8_t)(n0 | (n1 << 4));
            }
            if (lane == 0) {
                __half gamma_h = __float2half(gamma);
                uint16_t gamma_u16 = *reinterpret_cast<uint16_t*>(&gamma_h);
                out[16] = (uint8_t)(gamma_u16 & 0xFF);
                out[17] = (uint8_t)((gamma_u16 >> 8) & 0xFF);
            }
        }
    }
}


// C++ wrapper — original (pack to buffer)
void tq_wht_pack(
    torch::Tensor input,    // [N, D] bfloat16 — zero-copy from vLLM
    torch::Tensor output    // [N, packed_size] uint8 (pre-allocated)
) {
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(output.dtype() == torch::kUInt8, "output must be uint8");

    int N = input.size(0);
    int D = input.size(1);
    int packed_size = output.size(1);

    dim3 grid(N, D / 32);
    dim3 block(32);  // one warp
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    #define PACK_LAUNCH(HD) \
        tq_wht_pack_kernel<HD><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), \
            output.data_ptr<uint8_t>(), N, packed_size)

    if (D == 256) { PACK_LAUNCH(256); }
    else if (D == 128) { PACK_LAUNCH(128); }
    else if (D == 64) { PACK_LAUNCH(64); }
    else if (D == 512) { PACK_LAUNCH(512); }
    else { TORCH_CHECK(false, "tq_wht_pack: unsupported D=", D); }
    #undef PACK_LAUNCH
}

// C++ wrapper — fused pack + KV cache write
void tq_wht_pack_to_cache(
    torch::Tensor input,         // [num_tokens, num_heads, D] bfloat16
    torch::Tensor kv_cache,      // [num_blocks, 2, block_size, num_heads, max_packed] uint8
    torch::Tensor slot_mapping,  // [num_tokens] int32
    int kv_idx,                  // 0=K, 1=V
    int stride_block,
    int stride_kv,
    int stride_slot,
    int stride_head,
    int mse_bits                 // 2, 3, or 4
) {
    TORCH_CHECK(input.dim() == 3, "input must be 3D [tokens, heads, D]");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(slot_mapping.dtype() == torch::kInt32, "slot_mapping must be int32");

    int num_tokens = input.size(0);
    int num_heads = input.size(1);
    int D = input.size(2);
    int block_size_kv = kv_cache.size(2);
    int bpb = (mse_bits == 2) ? 10 : (mse_bits == 3) ? 14 : 18;
    int packed_size = (D / 32) * bpb;

    dim3 grid(num_tokens, D / 32);
    dim3 block(32);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    #define PACK_CACHE_LAUNCH(HD, MB) \
        tq_wht_pack_to_cache_kernel<HD, MB><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), \
            kv_cache.data_ptr<uint8_t>(), \
            slot_mapping.data_ptr<int>(), \
            num_tokens, num_heads, kv_idx, block_size_kv, packed_size, \
            stride_block, stride_kv, stride_slot, stride_head)

    if (D == 256 && mse_bits == 2) { PACK_CACHE_LAUNCH(256, 2); }
    else if (D == 256 && mse_bits == 3) { PACK_CACHE_LAUNCH(256, 3); }
    else if (D == 256 && mse_bits == 4) { PACK_CACHE_LAUNCH(256, 4); }
    else if (D == 128 && mse_bits == 2) { PACK_CACHE_LAUNCH(128, 2); }
    else if (D == 128 && mse_bits == 3) { PACK_CACHE_LAUNCH(128, 3); }
    else if (D == 128 && mse_bits == 4) { PACK_CACHE_LAUNCH(128, 4); }
    else if (D == 64 && mse_bits == 2) { PACK_CACHE_LAUNCH(64, 2); }
    else if (D == 64 && mse_bits == 3) { PACK_CACHE_LAUNCH(64, 3); }
    else if (D == 64 && mse_bits == 4) { PACK_CACHE_LAUNCH(64, 4); }
    else { TORCH_CHECK(false, "tq_wht_pack_to_cache: unsupported D=", D,
                        " mse_bits=", mse_bits); }
    #undef PACK_CACHE_LAUNCH
}

}  // namespace turboquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq_wht_pack", &turboquant::tq_wht_pack,
          "Fused WHT pack: float → WHT → quantize → bitpack → uint8");
    m.def("tq_wht_pack_to_cache", &turboquant::tq_wht_pack_to_cache,
          "Fused WHT pack + direct KV cache write (eliminates scatter)");
}
