// SPDX-License-Identifier: Apache-2.0
// XFP fused GEMM — depack word-aligned indices → codebook LUT → MAC → GEMM.
//
//   C[M, N] = A[M, K] @ codebook[N, 2^BITS][B_packed[K_packed, N]].T
//
// One kernel per BITS ∈ {2, 3, 4}. Word layout (uint32):
//   BITS=2: 16 values/word
//   BITS=3: 10 values/word (2 reserve bits at the top of each word)
//   BITS=4:  8 values/word
//
// Tiling: 4 N columns per thread, M_COUNT rows per block, K split across
// gridDim.z with atomicAdd on the output. Matches the layout convention of
// mq_gemm_int2.cu so the JIT build path and gridDim logic stay consistent.
//
// This kernel has no scales/zeros — the codebook IS the dequant table.
// A per-output-channel fp16 LUT of size 2^BITS (8, 16, or 32 bytes) lives
// in registers for the duration of the tile; we do not cache it in shared
// memory because each thread's 4 N columns each have their own LUT.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace multiquant {

// Threads per block (each handles 4 N columns = 128 N output per block).
#define XFP_BLOCK_N 32
// Words of packed input processed per gridDim.z step. Keep small so that
// gridDim.z stays large enough to saturate SMs; atomicAdd reconciles
// contributions at the end.
#define XFP_WORDS_PER_BLOCK 2

template <int BITS, int M_COUNT>
__global__ void xfp_gemm_kernel(
    const half* __restrict__ A,             // [M, K]
    const uint32_t* __restrict__ B_packed,  // [K_packed, N] uint32
    const half* __restrict__ codebook,      // [N, 2^BITS] fp16
    half* __restrict__ C,                   // [M, N] fp16
    int M, int N, int K, int K_packed)
{
    constexpr int VALS_PER_WORD =
        (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);
    constexpr int BLOCK_COLS = XFP_BLOCK_N * 4;        // 128 N values per block
    constexpr int SMEM_LUT_ELEMS = BLOCK_COLS * LUT_SIZE;

    // Shared-memory staging of the codebook slice for the 128 N-columns
    // this block owns. 128 * 2^BITS fp16 entries = 1/2/4 KB for BITS 2/3/4.
    // All threads in the block hit the same LUT window repeatedly across
    // the K loop, so moving it from L1/global to SMEM removes the per-MAC
    // global traffic that dominated v1.
    __shared__ half s_cb[SMEM_LUT_ELEMS];

    int tid = threadIdx.x;
    int n_base = blockIdx.x * BLOCK_COLS;
    int m_base = blockIdx.y * M_COUNT;
    int kw_start = blockIdx.z * XFP_WORDS_PER_BLOCK;
    int kw_end = kw_start + XFP_WORDS_PER_BLOCK;
    if (kw_end > K_packed) kw_end = K_packed;

    // Cooperative load: the 32 threads in this block together pull
    // BLOCK_COLS * LUT_SIZE fp16 entries from global codebook into s_cb.
    // Each thread handles (BLOCK_COLS * LUT_SIZE / 32) entries.
    for (int idx = tid; idx < SMEM_LUT_ELEMS; idx += XFP_BLOCK_N) {
        int n_local = idx / LUT_SIZE;
        int entry = idx % LUT_SIZE;
        int n_global = n_base + n_local;
        s_cb[idx] = (n_global < N)
            ? codebook[n_global * LUT_SIZE + entry]
            : __float2half(0.0f);
    }
    __syncthreads();

    int n = n_base + tid * 4;
    if (n >= N || m_base >= M) return;

    // Per-column pointers into shared memory. cheap bounds check on n.
    const half* cb0 = s_cb + (tid * 4 + 0) * LUT_SIZE;
    const half* cb1 = s_cb + (tid * 4 + 1) * LUT_SIZE;
    const half* cb2 = s_cb + (tid * 4 + 2) * LUT_SIZE;
    const half* cb3 = s_cb + (tid * 4 + 3) * LUT_SIZE;
    bool v0 = (n + 0 < N);
    bool v1 = (n + 1 < N);
    bool v2_ = (n + 2 < N);
    bool v3_ = (n + 3 < N);

    float acc[M_COUNT][4] = {};

    for (int kw = kw_start; kw < kw_end; kw++) {
        uint32_t w0 = v0 ? B_packed[kw * N + n + 0] : 0u;
        uint32_t w1 = v1 ? B_packed[kw * N + n + 1] : 0u;
        uint32_t w2 = v2_ ? B_packed[kw * N + n + 2] : 0u;
        uint32_t w3 = v3_ ? B_packed[kw * N + n + 3] : 0u;

        int k_base = kw * VALS_PER_WORD;
        int vals_this_word = VALS_PER_WORD;
        if (k_base + vals_this_word > K) vals_this_word = K - k_base;

        #pragma unroll
        for (int i = 0; i < VALS_PER_WORD; i++) {
            if (i >= vals_this_word) break;
            int k = k_base + i;

            int i0 = (int)((w0 >> (i * BITS)) & MASK);
            int i1 = (int)((w1 >> (i * BITS)) & MASK);
            int i2 = (int)((w2 >> (i * BITS)) & MASK);
            int i3 = (int)((w3 >> (i * BITS)) & MASK);

            // SMEM LUT lookups replace v1's global codebook reads.
            float wv0 = __half2float(cb0[i0]);
            float wv1 = __half2float(cb1[i1]);
            float wv2 = __half2float(cb2[i2]);
            float wv3 = __half2float(cb3[i3]);

            #pragma unroll
            for (int m = 0; m < M_COUNT; m++) {
                int mi = m_base + m;
                if (mi >= M) continue;
                float a = __half2float(A[mi * K + k]);
                acc[m][0] += wv0 * a;
                acc[m][1] += wv1 * a;
                acc[m][2] += wv2 * a;
                acc[m][3] += wv3 * a;
            }
        }
    }

    // atomicAdd output (K-split accumulation).
    for (int m = 0; m < M_COUNT; m++) {
        int mi = m_base + m;
        if (mi >= M) break;
        if (v0) atomicAdd(reinterpret_cast<half*>(&C[mi * N + n + 0]),
                          __float2half(acc[m][0]));
        if (v1) atomicAdd(reinterpret_cast<half*>(&C[mi * N + n + 1]),
                          __float2half(acc[m][1]));
        if (v2_) atomicAdd(reinterpret_cast<half*>(&C[mi * N + n + 2]),
                           __float2half(acc[m][2]));
        if (v3_) atomicAdd(reinterpret_cast<half*>(&C[mi * N + n + 3]),
                           __float2half(acc[m][3]));
    }
}

template <int BITS, int M_COUNT>
static inline void launch_xfp_mcount(
    const half* A_ptr, const uint32_t* B_ptr, const half* cb_ptr,
    half* C_ptr, int M, int N, int K, int K_packed)
{
    dim3 block(XFP_BLOCK_N);
    dim3 grid(
        (N + XFP_BLOCK_N * 4 - 1) / (XFP_BLOCK_N * 4),
        (M + M_COUNT - 1) / M_COUNT,
        (K_packed + XFP_WORDS_PER_BLOCK - 1) / XFP_WORDS_PER_BLOCK);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_kernel<BITS, M_COUNT><<<grid, block, 0, stream>>>(
        A_ptr, B_ptr, cb_ptr, C_ptr, M, N, K, K_packed);
}

template <int BITS>
static void launch_xfp(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C, int K)
{
    int M = A.size(0);
    int N = B_packed.size(1);
    int K_packed = B_packed.size(0);

    const half* A_ptr =
        reinterpret_cast<const half*>(A.data_ptr());
    const uint32_t* B_ptr =
        reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>());
    const half* cb_ptr =
        reinterpret_cast<const half*>(codebook.data_ptr());
    half* C_ptr = reinterpret_cast<half*>(C.data_ptr());

    // Pick M_COUNT based on actual M. Decode (M=1 per step) is the
    // common path and suffers from 3/4 dead inner iterations with
    // M_COUNT=4, so we specialize it. Prefill with M>=4 keeps the
    // wider tile for per-block reuse of SMEM codebook and A reads.
    if (M == 1) {
        launch_xfp_mcount<BITS, 1>(
            A_ptr, B_ptr, cb_ptr, C_ptr, M, N, K, K_packed);
    } else if (M < 4) {
        launch_xfp_mcount<BITS, 2>(
            A_ptr, B_ptr, cb_ptr, C_ptr, M, N, K, K_packed);
    } else {
        launch_xfp_mcount<BITS, 4>(
            A_ptr, B_ptr, cb_ptr, C_ptr, M, N, K, K_packed);
    }
}

void xfp_gemm(
    torch::Tensor A,         // [M, K] fp16
    torch::Tensor B_packed,  // [K_packed, N] int32 (uint32 bitwise)
    torch::Tensor codebook,  // [N, 2^bits] fp16
    torch::Tensor C,         // [M, N] fp16 (pre-zeroed)
    int64_t bits,
    int64_t K)
{
    TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() &&
                codebook.is_cuda() && C.is_cuda(),
                "xfp_gemm: all tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "xfp_gemm: A must be float16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32,
                "xfp_gemm: B_packed must be int32 (uint32 bit pattern)");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16,
                "xfp_gemm: codebook must be float16");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "xfp_gemm: C must be float16");
    TORCH_CHECK(A.dim() == 2 && B_packed.dim() == 2 &&
                codebook.dim() == 2 && C.dim() == 2,
                "xfp_gemm: all tensors must be 2D");
    TORCH_CHECK(A.size(1) == K,
                "xfp_gemm: A.size(1) must equal K");
    TORCH_CHECK(A.size(0) == C.size(0) && B_packed.size(1) == C.size(1),
                "xfp_gemm: M/N shape mismatch");
    TORCH_CHECK(codebook.size(0) == B_packed.size(1),
                "xfp_gemm: codebook rows must equal N");

    const int64_t lut_expected = 1LL << bits;
    TORCH_CHECK(codebook.size(1) == lut_expected,
                "xfp_gemm: codebook columns must equal 2^bits (got ",
                codebook.size(1), ", expected ", lut_expected, ")");

    if (bits == 2) launch_xfp<2>(A, B_packed, codebook, C, static_cast<int>(K));
    else if (bits == 3) launch_xfp<3>(A, B_packed, codebook, C, static_cast<int>(K));
    else if (bits == 4) launch_xfp<4>(A, B_packed, codebook, C, static_cast<int>(K));
    else TORCH_CHECK(false, "xfp_gemm: unsupported bits=", bits,
                     " (supported: 2, 3, 4)");
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP fused depack + codebook LUT + GEMM");
}
