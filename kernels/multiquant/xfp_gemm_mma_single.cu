// SPDX-License-Identifier: Apache-2.0
// XFP MMA single-tile — MMA m16n8k16 with B decoded from XFP packed format.
//
// One block, one MMA tile (M=16, N=8). K-loop decodes B on the fly via
// SHFL lookup (v10 pattern) and writes decoded BF16 into lane fragments
// in mma m16n8k16 B-layout.
//
// Purpose: Phase A step 2 — verify Tensor-Core + XFP-decode integration
// on a minimal single-expert, single-tile kernel before scaling to MoE.
//
// B_packed layout matches v8/v10: [K_groups, N, 32] flattened uint32,
// where K_groups = ceil(K_packed/32), N = output cols (here N=8),
// each group has 32 packed words (one per lane of v8's warp-per-col).
//
// For this single-tile MMA kernel with N=8, we reinterpret the storage:
// each of the 8 output cols has K_packed packed-uint32 words, stored
// consecutively in the "flat" tensor indexed by n_offset = col*32 + lane.
// We read ALL 8 cols' packed words in parallel across the 32 lanes.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace xfp_mma_single {

__device__ __forceinline__ uint32_t __bfloat1622uint(__nv_bfloat162 v) {
    uint32_t r;
    asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(*(uint32_t*)&v));
    return r;
}

#define WARP_SIZE 32

template <int BITS>
__global__ void xfp_mma_single_kernel(
    const __nv_bfloat16* __restrict__ A,   // [16, K]
    const uint32_t* __restrict__ B_packed,  // repacked flat [K_groups, 8, 32]
    const half* __restrict__ codebook,      // [8, 2^BITS]
    float* __restrict__ C,                  // [16, 8] fp32
    int K, int K_packed)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);
    constexpr int N = 8;

    int lane = threadIdx.x;

    // Codebook in SMEM: 8 cols × 16 entries × 4 bytes = 512 bytes.
    // SHFL-per-col would require each lane to hold 8 codebook registers,
    // which the compiler may spill to local memory when indexed by runtime
    // n_col values → __shfl_sync reads from wrong source. SMEM is safer.
    __shared__ float s_cb[N * LUT_SIZE];
    if (lane < LUT_SIZE) {
        #pragma unroll
        for (int n = 0; n < N; n++) {
            s_cb[n * LUT_SIZE + lane] =
                __half2float(codebook[n * LUT_SIZE + lane]);
        }
    }
    __syncwarp();

    // Output accumulator (mma m16n8k16: 4 fp32 per lane)
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // K loop: process 16 K values per iteration (MMA K-tile = 16)
    // At BITS=4: 16 K values = 2 packed words (VALS_PER_WORD=8 per word)
    // At BITS=3: 16 K values span ~2 packed words (10 per word, need handling)
    // At BITS=2: 16 K values = 1 packed word (16 per word)
    //
    // For now, focus on BITS=4: each iteration needs 2 packed words per col.

    // We build B-tile [16 K-rows, 8 N-cols] decoded as BF16 in SMEM,
    // then use direct register loads in mma layout (no ldmatrix yet for
    // simplicity — ldmatrix integration in next iteration).
    __shared__ __nv_bfloat16 s_B[16 * 8];  // 256 bytes

    int n_groups = (K_packed + WARP_SIZE - 1) / WARP_SIZE;
    int k_per_mma = 16;  // MMA K-tile

    for (int k = 0; k < K; k += k_per_mma) {
        // Compute which packed group contains this k range.
        // At BITS=4, k = group_idx * WARP_SIZE * VALS_PER_WORD + ...
        // But WARP_SIZE * VALS_PER_WORD = 32 * 8 = 256 K values per group.
        // So k < 256 is group 0, k in [256, 512) is group 1, etc.
        // Within group: packed_word_idx = (k / VALS_PER_WORD) % WARP_SIZE

        // Decode 16 K values × 8 cols → 128 bf16 into s_B
        // Each lane decodes its portion: lane handles 4 bf16 (128/32).
        // Lane i decodes: k_row = i / 2, n_col = i % 8
        // Wait: 4 bf16 per lane, 32 lanes = 128. 128 = 16*8.
        // Layout: s_B[k_row * 8 + n_col]
        // Per-lane ownership: 4 values
        //   lane l handles (k_row, n_col) pairs:
        //   l=0: (0,0) (0,1) (0,2) (0,3)      – 4 cols of row 0
        //   l=1: (0,4) (0,5) (0,6) (0,7)
        //   l=2: (1,0) ... etc
        //   l=i: k_row = i/2, cols = (i%2)*4 .. (i%2)*4+3

        int l_krow = lane / 2;           // 0..15
        int l_ncol_base = (lane % 2) * 4; // 0 or 4
        int k_abs = k + l_krow;           // absolute K index for this row

        // Find packed word containing k_abs
        int kw_global = k_abs / VALS_PER_WORD;   // packed-word index
        int slot = k_abs % VALS_PER_WORD;         // bit offset in word
        int g = kw_global / WARP_SIZE;            // K-group
        int kw_in_group = kw_global % WARP_SIZE;  // lane-offset in group

        if (k_abs < K) {
            #pragma unroll
            for (int dn = 0; dn < 4; dn++) {
                int n_col = l_ncol_base + dn;
                // Read packed[g, n_col, kw_in_group]
                int offset = g * N * WARP_SIZE + n_col * WARP_SIZE + kw_in_group;
                uint32_t packed = (kw_global < K_packed)
                    ? B_packed[offset] : 0u;
                int idx = (int)((packed >> (slot * BITS)) & MASK);
                float w = s_cb[n_col * LUT_SIZE + idx];
                s_B[l_krow * 8 + n_col] = __float2bfloat16(w);
            }
        } else {
            #pragma unroll
            for (int dn = 0; dn < 4; dn++) {
                s_B[l_krow * 8 + l_ncol_base + dn] = __float2bfloat16(0.0f);
            }
        }
        __syncwarp();  // Ensure s_B is fully written before MMA reads

        // --- Load A tile [16, 16] into lane fragments ---
        uint32_t a_frag[4];
        {
            int row_low = lane / 4;
            int row_high = row_low + 8;
            int col_base = (lane % 4) * 2;
            #define A_LOAD(row, col) A[(row) * K + k + (col)]
            __nv_bfloat16 a0 = A_LOAD(row_low, col_base);
            __nv_bfloat16 a1 = A_LOAD(row_low, col_base + 1);
            __nv_bfloat16 a2 = A_LOAD(row_high, col_base);
            __nv_bfloat16 a3 = A_LOAD(row_high, col_base + 1);
            __nv_bfloat16 a4 = A_LOAD(row_low, col_base + 8);
            __nv_bfloat16 a5 = A_LOAD(row_low, col_base + 9);
            __nv_bfloat16 a6 = A_LOAD(row_high, col_base + 8);
            __nv_bfloat16 a7 = A_LOAD(row_high, col_base + 9);
            #undef A_LOAD
            a_frag[0] = __bfloat1622uint(__nv_bfloat162(a0, a1));
            a_frag[1] = __bfloat1622uint(__nv_bfloat162(a2, a3));
            a_frag[2] = __bfloat1622uint(__nv_bfloat162(a4, a5));
            a_frag[3] = __bfloat1622uint(__nv_bfloat162(a6, a7));
        }

        // --- Load B tile [16, 8] from SMEM into lane fragments ---
        uint32_t b_frag[2];
        {
            int col = lane / 4;
            int row_base = (lane % 4) * 2;
            __nv_bfloat16 b0 = s_B[(row_base + 0) * 8 + col];
            __nv_bfloat16 b1 = s_B[(row_base + 1) * 8 + col];
            __nv_bfloat16 b2 = s_B[(row_base + 8) * 8 + col];
            __nv_bfloat16 b3 = s_B[(row_base + 9) * 8 + col];
            b_frag[0] = __bfloat1622uint(__nv_bfloat162(b0, b1));
            b_frag[1] = __bfloat1622uint(__nv_bfloat162(b2, b3));
        }

        // --- MMA ---
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1])
        );

        __syncwarp();
    }

    // --- Store C [16, 8] fp32 ---
    int row_low = lane / 4;
    int col_base = (lane % 4) * 2;
    C[(row_low) * 8 + col_base + 0] = c[0];
    C[(row_low) * 8 + col_base + 1] = c[1];
    C[(row_low + 8) * 8 + col_base + 0] = c[2];
    C[(row_low + 8) * 8 + col_base + 1] = c[3];
}

void xfp_mma_single(
    torch::Tensor A,         // [16, K] bf16
    torch::Tensor B_packed,  // packed flat
    torch::Tensor codebook,  // [8, 2^BITS] f16
    torch::Tensor C,         // [16, 8] fp32
    int64_t bits,
    int64_t K)
{
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32, "B_packed must be int32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16, "codebook must be f16");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be f32");
    TORCH_CHECK(A.sizes() == torch::IntArrayRef({16, K}), "A shape mismatch");
    TORCH_CHECK(C.sizes() == torch::IntArrayRef({16, 8}), "C shape mismatch");
    TORCH_CHECK(codebook.size(0) == 8, "codebook must be for N=8");

    int vpw = (bits == 2) ? 16 : (bits == 3) ? 10 : 8;
    int K_packed = (static_cast<int>(K) + vpw - 1) / vpw;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (bits == 4) {
        xfp_mma_single_kernel<4><<<1, WARP_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
            reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
            reinterpret_cast<const half*>(codebook.data_ptr()),
            C.data_ptr<float>(),
            static_cast<int>(K), K_packed);
    } else {
        TORCH_CHECK(false, "Only bits=4 supported in single-tile test");
    }
}

}  // namespace xfp_mma_single

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_mma_single", &xfp_mma_single::xfp_mma_single,
          "Single-tile MMA with XFP-decoded B");
}
