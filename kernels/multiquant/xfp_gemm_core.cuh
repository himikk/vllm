// SPDX-License-Identifier: Apache-2.0
// XFP GEMM core — template header shared by Linear and MoE kernels.
//
// Extracted from xfp_gemm_v10.cu and xfp_moe_gemm_v10.cu. The inner loop
// (K-scan, SHFL codebook lookup, warp-reduce) is literally identical
// between the two. Only the prologue (metadata resolution, per-expert
// offsets) and the epilogue (output write, optional topk_weights) differ.
//
// This header is the single source of truth. Each variant (Linear / MoE)
// only provides a Policy struct with:
//   Policy::Params                        — launch-time data
//   Policy::prologue<BITS, LUT>(...) -> Ctx
//   Policy::epilogue(C, N, ctx, acc)
//
// Future optimisations (cp.async, MMA, LOP3 dequant) touch only the core
// template. Linear and MoE automatically inherit them.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace multiquant {

#ifndef XFP_WARP_SIZE
#define XFP_WARP_SIZE 32
#endif

#ifndef XFP_WARPS_PER_BLOCK
#define XFP_WARPS_PER_BLOCK 8
#endif

#define XFP_BLOCK_SIZE (XFP_WARP_SIZE * XFP_WARPS_PER_BLOCK)


// ─── Shared inner-loop template ───────────────────────────────────────

template <int BITS, class Policy>
__device__ __forceinline__ void xfp_gemm_core(
    const __nv_bfloat16* __restrict__ A,
    const uint32_t*      __restrict__ B_packed_base,
    const half*          __restrict__ codebook_base,
    __nv_bfloat16*       __restrict__ C,
    int N, int K, int K_packed,
    typename Policy::Params params)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int warp_id = threadIdx.x / XFP_WARP_SIZE;
    int lane    = threadIdx.x % XFP_WARP_SIZE;

#ifdef XFP_CORE_USE_SMEM_A
    // ── v12: block-uniform A-row resolution + cooperative SMEM load ──
    // block_A_row() must return nullptr iff the WHOLE block is inactive
    // (MoE padding expert/token, Linear m-out-of-range). If non-null,
    // every thread in the block participates in the cooperative load.
    const __nv_bfloat16* block_A = Policy::block_A_row(A, K, params);
    if (block_A == nullptr) return;  // all threads exit together → safe

    // Static SMEM — compile-time size per Policy. No dynamic smem launch.
    __shared__ __nv_bfloat16 s_A[Policy::K_SMEM_MAX];

    // Vectorized bf162 cooperative copy. K must be even (asserted host-side).
    {
        int pair_count = K >> 1;
        const __nv_bfloat162* __restrict__ src2 =
            reinterpret_cast<const __nv_bfloat162*>(block_A);
        __nv_bfloat162* dst2 =
            reinterpret_cast<__nv_bfloat162*>(s_A);
        #pragma unroll 4
        for (int i = threadIdx.x; i < pair_count; i += XFP_BLOCK_SIZE) {
            dst2[i] = src2[i];
        }
    }
    __syncthreads();
#endif

    // Policy resolves all per-block/per-warp metadata: output column n,
    // A_row pointer, per-expert B_packed pointer, per-expert codebook
    // slice pointer, and any routing info for the epilogue.
    auto ctx = Policy::template prologue<BITS, LUT_SIZE>(
        A, B_packed_base, codebook_base, N, K, warp_id, lane, params);
    if (!ctx.active) return;

    // A-source pointer: SMEM (v12) or per-warp global (v11).
#ifdef XFP_CORE_USE_SMEM_A
    const __nv_bfloat16* __restrict__ A_src = s_A;
#else
    const __nv_bfloat16* __restrict__ A_src = ctx.A_row;
#endif

    // ── Codebook in register, SHFL lookup (same as v10) ──
    float my_cb_val = (lane < LUT_SIZE)
        ? __half2float(ctx.codebook_slice[lane])
        : 0.0f;
    // No __syncthreads — warp-local.

    float acc = 0.0f;
    int n_offset = ctx.n * XFP_WARP_SIZE + lane;
    int n_groups = (K_packed + XFP_WARP_SIZE - 1) / XFP_WARP_SIZE;

#ifdef XFP_CORE_USE_CPASYNC
    // ── cp.async double-buffered K-loop ──
    // Prefetch B_packed for stage gi+1 while processing stage gi. Each lane
    // loads 4 bytes (one uint32) per group. SMEM: 2 × WARPS × 32 × 4 = 2 KiB.
    __shared__ uint32_t s_B[2][XFP_WARPS_PER_BLOCK * XFP_WARP_SIZE];

    auto issue_group = [&](int gi, int stage) {
        uint32_t* dst_ptr = &s_B[stage][warp_id * XFP_WARP_SIZE + lane];
        int kw = lane + gi * XFP_WARP_SIZE;
        if (kw < K_packed) {
            const uint32_t* src_ptr =
                &ctx.B_packed[gi * N * XFP_WARP_SIZE + n_offset];
            unsigned s_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(dst_ptr));
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4;\n"
                :: "r"(s_addr), "l"(src_ptr));
        } else {
            *dst_ptr = 0u;  // sync write; OOB lane has no global fetch
        }
    };

    // Prolog: prefetch gi=0
    if (n_groups > 0) issue_group(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    for (int gi = 0; gi < n_groups; gi++) {
        // Prefetch gi+1 into the other stage (if exists)
        if (gi + 1 < n_groups) {
            issue_group(gi + 1, (gi + 1) & 1);
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncwarp();

        uint32_t packed = s_B[gi & 1][warp_id * XFP_WARP_SIZE + lane];
        int kw = lane + gi * XFP_WARP_SIZE;
        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot++) {
            int idx = (int)((packed >> (slot * BITS)) & MASK);
            float w = __shfl_sync(0xffffffff, my_cb_val, idx);
            int k = k_base + slot;
            if (k < K && kw < K_packed) {
                float a = __bfloat162float(A_src[k]);
                acc = fmaf(w, a, acc);
            }
        }
    }
#else
    // ── Synchronous K-loop with safe/tail split (v11 baseline) ──
    // Safe groups: all 32 lanes have kw < K_packed AND k_base+VALS-1 < K.
    // No per-slot predicate needed. Paired bfloat162 A-loads reduce
    // slot-iteration count by 2× (same trick as v8).
    //
    // Tail group: at most one, handles remaining kw < K_packed lanes with
    // per-slot predicate. All 32 lanes still iterate the same number of
    // times (for __shfl_sync correctness).

    // "Safe" = full group + last slot's k still in range
    // K_safe = K_packed where the FULL K is covered (kw*VALS + VALS-1 < K)
    int K_packed_safe = K / VALS_PER_WORD;          // full packed words
    int n_full_groups = K_packed_safe / XFP_WARP_SIZE;  // all lanes valid
    int K_safe_words  = n_full_groups * XFP_WARP_SIZE;

    // Fast path: no bounds checks in the hot loop
    for (int gi = 0; gi < n_full_groups; gi++) {
        int kw = lane + gi * XFP_WARP_SIZE;
        uint32_t packed = ctx.B_packed[gi * N * XFP_WARP_SIZE + n_offset];
        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot += 2) {
            __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(
                A_src + k_base + slot);
            float a0 = __bfloat162float(__low2bfloat16(a2));
            float a1 = __bfloat162float(__high2bfloat16(a2));
            int idx0 = (int)((packed >> (slot * BITS)) & MASK);
            int idx1 = (int)((packed >> ((slot + 1) * BITS)) & MASK);
            float w0 = __shfl_sync(0xffffffff, my_cb_val, idx0);
            float w1 = __shfl_sync(0xffffffff, my_cb_val, idx1);
            acc = fmaf(w0, a0, acc);
            acc = fmaf(w1, a1, acc);
        }
    }

    // Tail: remaining groups (partial or K-unaligned)
    for (int gi = n_full_groups; gi < n_groups; gi++) {
        int kw = lane + gi * XFP_WARP_SIZE;
        uint32_t packed = (kw < K_packed)
            ? ctx.B_packed[gi * N * XFP_WARP_SIZE + n_offset] : 0u;
        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot++) {
            int idx = (int)((packed >> (slot * BITS)) & MASK);
            float w = __shfl_sync(0xffffffff, my_cb_val, idx);
            int k = k_base + slot;
            if (k < K && kw < K_packed) {
                float a = __bfloat162float(A_src[k]);
                acc = fmaf(w, a, acc);
            }
        }
    }
#endif

    // Warp reduction (Lane 0 ends up with the sum)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        Policy::epilogue(C, N, ctx, acc);
    }
}


// ─── Linear policy ────────────────────────────────────────────────────
// Grid: (ceil(N/WARPS_PER_BLOCK), M)
// Each warp produces one output C[m, n_warp]. No routing, no scaling.

struct LinearPolicy {
    // Rigid compile-time SMEM budget for v12 A-row cache.
    // Covers FFN gate/up/down and attn q/k/v up to model-dim 8192.
    // Shapes with K > 8192 (e.g. attn kv_b 17408) must use v11.
    static constexpr int K_SMEM_MAX = 8192;

    struct Params {
        int M;
    };

    struct Ctx {
        bool active;
        int n;
        int m;
        const __nv_bfloat16* A_row;
        const uint32_t*      B_packed;
        const half*          codebook_slice;
    };

    // Block-uniform A-row pointer. Returns nullptr iff whole block is
    // inactive (m out of range). Called BEFORE __syncthreads in v12.
    __device__ static const __nv_bfloat16* block_A_row(
        const __nv_bfloat16* A, int K, const Params& p)
    {
        int m = blockIdx.y;
        return (m < p.M) ? (A + (size_t)m * K) : nullptr;
    }

    template <int BITS, int LUT>
    __device__ static Ctx prologue(
        const __nv_bfloat16* A,
        const uint32_t*      B,
        const half*          cb,
        int N, int K,
        int warp_id, int /*lane*/,
        Params p)
    {
        int n = blockIdx.x * XFP_WARPS_PER_BLOCK + warp_id;
        int m = blockIdx.y;
        Ctx c{};
        c.active = (m < p.M) && (n < N);
        if (!c.active) return c;
        c.n = n;
        c.m = m;
        c.A_row          = A + m * K;
        c.B_packed       = B;                 // no per-expert offset
        c.codebook_slice = cb + n * LUT;
        return c;
    }

    __device__ static void epilogue(
        __nv_bfloat16* C, int N, const Ctx& ctx, float acc)
    {
        C[ctx.m * N + ctx.n] = __float2bfloat16(acc);
    }
};


// ─── MoE policy ───────────────────────────────────────────────────────
// Grid: (ceil(N/WARPS_PER_BLOCK), num_token_blocks)
// Each block owns one (token, expert) pair; warp produces C[token_id, n].

struct MoEPolicy {
    // Rigid compile-time SMEM budget for v12 A-row cache.
    // Covers Qwen/GLM MoE shapes (K=2048 max) with 2× headroom.
    static constexpr int K_SMEM_MAX = 4096;

    struct Params {
        const int32_t* sorted_token_ids;   // [num_tokens_padded]
        const int32_t* expert_ids;         // [num_token_blocks]
        const float*   topk_weights;       // [M * top_k] or nullptr
        int M;
        int top_k;
        int flat_per_expert;               // int32 words per expert in B_packed
        int num_valid_tokens;
    };

    struct Ctx {
        bool active;
        int n;
        int token_id;
        const __nv_bfloat16* A_row;
        const uint32_t*      B_packed;
        const half*          codebook_slice;
        const float*         topk_weights;
    };

    // Block-uniform A-row pointer. Returns nullptr iff the whole block is
    // inactive (padding expert, padding token, or orig_token out of M).
    // All warps of a MoE block share the same orig_token, so this is safe.
    __device__ static const __nv_bfloat16* block_A_row(
        const __nv_bfloat16* A, int K, const Params& p)
    {
        int tb = blockIdx.y;
        int expert_id = p.expert_ids[tb];
        if (expert_id < 0) return nullptr;
        int token_id = p.sorted_token_ids[tb];
        if (token_id >= p.num_valid_tokens) return nullptr;
        int orig = token_id / p.top_k;
        if (orig >= p.M) return nullptr;
        return A + (size_t)orig * K;
    }

    template <int BITS, int LUT>
    __device__ static Ctx prologue(
        const __nv_bfloat16* A,
        const uint32_t*      B,
        const half*          cb,
        int N, int K,
        int warp_id, int /*lane*/,
        Params p)
    {
        Ctx c{};
        int n  = blockIdx.x * XFP_WARPS_PER_BLOCK + warp_id;
        int tb = blockIdx.y;

        int expert_id = p.expert_ids[tb];
        if (expert_id < 0) return c;                    // padding block
        int token_id = p.sorted_token_ids[tb];
        if (token_id >= p.num_valid_tokens) return c;   // padding token

        int orig = token_id / p.top_k;
        if (orig >= p.M || n >= N) return c;

        c.active = true;
        c.n = n;
        c.token_id = token_id;
        c.A_row          = A + orig * K;
        c.B_packed       = B + (size_t)expert_id * p.flat_per_expert;
        c.codebook_slice = cb + (size_t)expert_id * N * LUT + n * LUT;
        c.topk_weights   = p.topk_weights;
        return c;
    }

    __device__ static void epilogue(
        __nv_bfloat16* C, int N, const Ctx& ctx, float acc)
    {
        if (ctx.topk_weights != nullptr) {
            acc *= ctx.topk_weights[ctx.token_id];
        }
        C[ctx.token_id * N + ctx.n] = __float2bfloat16(acc);
    }
};


// ─── Kernel entry point, templated by BITS and Policy ────────────────

template <int BITS, class Policy>
__global__ void xfp_gemm_templated_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint32_t*      __restrict__ B_packed,
    const half*          __restrict__ codebook,
    __nv_bfloat16*       __restrict__ C,
    int N, int K, int K_packed,
    typename Policy::Params params)
{
    xfp_gemm_core<BITS, Policy>(A, B_packed, codebook, C, N, K, K_packed, params);
}

}  // namespace multiquant
