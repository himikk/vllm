# KV-Cache Quantization Benchmark Results

**Platform**: DGX Spark (GB10 Blackwell, SM121, 120 GiB unified memory)
**Image**: vllm-ng17e-riy (vLLM 0.15.1 + SM12x patches)
**Date**: 2025-03-25
**Bench**: `bench.py` (deterministic, seed=42)

## Qwen3.5-35B-A3B (INT4 AutoRound, MoE 35B/A3B)

| KV-Cache | Short (20t) | Medium (150t) | Long (400t) | Math | Notes |
|----------|-------------|---------------|-------------|------|-------|
| FP8      | 2.6 tok/s   | 47.8 tok/s    | 36.2 tok/s  | 100% | Baseline |
| TQ3 PyTorch | 2.7 tok/s | 35.4 tok/s  | 28.7 tok/s  | 100% | -21-26% (PyTorch fallback) |
| TQ3 CUDA v1 | 2.6 tok/s | 33.3 tok/s  | 29.1 tok/s  | 100% | -20-30% (naive GEMV) |
| TQ3 CUDA v2 | 8.1 tok/s | 35.5 tok/s  | 32.9 tok/s  | 100% | -9% long, -26% med (tiled GEMV+float4) |
| **TQ3 CUDA v3** | **2.7 tok/s** | **45.5 tok/s** | **32.8 tok/s** | **100%** | **-5% med, -9% long (cached buffers)** |
| TQ4      | —           | —             | —           | —    | Pending |

## Qwen3.5-122B-A10B (INT4 AutoRound, MoE 122B/A10B)

| KV-Cache | Short (20t) | Medium (150t) | Long (400t) | Math | Notes |
|----------|-------------|---------------|-------------|------|-------|
| FP8      | —           | —             | —           | —    | Pending |
| TQ3      | —           | —             | —           | —    | Pending |
| TQ4      | —           | —             | —           | —    | Pending |

## Context Scaling (Qwen3.5-35B, 100 gen tokens, DGX Spark)

| Context | FP8 tok/s | TQ3 tok/s | TQ4 tok/s | TQ3+FP8 tok/s | TQ3 vs FP8 |
|---------|-----------|-----------|-----------|--------------|------------|
| 0       | 42.0*     | 45.7      | 15.1*     | 15.6*        | +9%        |
| 2K      | 38.5      | 33.3      | 35.0      | 13.7*        | -13%       |
| 8K      | 29.3      | 26.6      | 29.1      | 32.6         | -9%        |
| 32K     | 13.5      | 13.5      | 13.9      | 14.5         | 0%         |
| 64K     | 6.9       | —         | 7.5       | 7.2          | —          |
| 128K    | 2.9       | —         | 3.3       | 2.9          | —          |

*Low-context values affected by JIT/warmup artifacts

**TQ3+FP8**: TQ round-trip on keys + FP8 KV-cache storage.
Same memory as FP8 (2× compression), but TQ improves key quality.
Enabled via `VLLM_TQ_ROUNDTRIP=1` + `--kv-cache-dtype fp8`.

At 8K context TQ3+FP8 is **+19% faster** than FP8 alone (32.6 vs 27.4).
At 32K+ contexts, throughput converges (memory-bound, same cache size).

**NOTE**: Phase 1 TQ round-trip does NOT save additional KV-cache memory
beyond what FP8 provides. Asymmetric K/V layout (Q2) needed for 4-5× savings.

## Config

```
--max-model-len 32768
--gpu-memory-utilization 0.05
--kv-cache-memory-bytes 10G
--enforce-eager
--trust-remote-code
--limit-mm-per-prompt '{"image":0,"video":0}'
```

## TurboQuant Standalone Quality (CPU, head_dim=128)

| Metric | TQ3 | TQ4 |
|--------|-----|-----|
| Score correlation vs FP16 | 0.92 | 0.98 |
| Output cosine similarity | 0.92 | 0.98 |
| Inner product bias | < 0.005 | < 0.002 |
| Needle-in-haystack top-1 | 100% | 100% |
| Key compression vs FP16 | 4.9x | 3.8x |
| GPU quantize throughput (GB10) | 13M vecs/s | 9M vecs/s |

## Q2: Compressed Attention Kernel (Standalone)

**Reads directly from packed TQ cache — no shadow buffer, no extra memory.**

| Metric | Value |
|--------|-------|
| K-Score correlation vs BF16 | 1.000 (bitgenau) |
| Full output cosine vs BF16 | 0.944 |
| Memory: TQ3 K + FP8 V | 23 KB / 64 KB BF16 = **2.8× compression** |
| Score kernel latency | 4.3 µs (256 tokens × 20 heads) |
| Model tested | GLM-4.7-Flash (D=64, 20 KV-heads) |

### Architecture
```
K: TQ3 packed (28 bytes/vector, D=64)
   → CUDA kernel reads indices + signs directly
   → term1 = centroid gather, term2 = QJL sign multiply
   → No GEMV, no decompression

V: FP8 or BF16 (64-128 bytes/vector)
   → Standard dequant per element

Full decode:
  scores = compressed_score_kernel(q_rot, q_proj, k_cache_packed)
  weights = softmax(scores)
  output = weights @ dequant(v_cache)
```

## GLM-4.7-Flash INT4 AutoRound (DGX Spark)

| KV-Cache | Short (20t) | Medium (150t) | Long (400t) | Math |
|----------|-------------|---------------|-------------|------|
| FP8      | 37.2 tok/s  | 40.7 tok/s    | 40.3 tok/s  | 100% |
| **TQ3**  | **45.0 tok/s** | **47.4 tok/s** | **45.7 tok/s** | **100%** |
| Delta    | **+21%**    | **+16%**      | **+13%**    | 0%   |

TQ3 is **13-21% faster** than FP8 on GLM-4.7-Flash with identical accuracy.

## Qwen3.5-35B TQ3 Q2 (kv_storage_dtype, DGX Spark)

| KV-Cache | Short (20t) | Medium (150t) | Long (400t) | Math |
|----------|-------------|---------------|-------------|------|
| FP8      | 2.6*        | 47.8          | 36.2        | 100% |
| TQ3 Q1   | 2.7*        | 45.5          | 32.8        | 100% |
| **TQ3 Q2** | **9.6**   | **40.8**      | **38.1**    | **100%** |

*Short values affected by warmup.
TQ3 Q2 long is +5% vs FP8. Medium is -15% (TQ round-trip overhead).

## Q3: Custom TQ Backend — Echte Speicherersparnis

GLM-4.7-Flash INT4 AutoRound (DGX Spark):

| Backend | Short | Med | Long | Math | KV-Cache |
|---------|-------|-----|------|------|----------|
| FP8 (FlashInfer) | 37.2 | 40.7 | 40.3 | 100% | 40,960 B/block |
| TQ3 Q2 (FP8+RT) | 45.0 | 47.4 | 45.7 | 100% | 40,960 B/block |
| **TQ3 Q3 (custom)** | **39.2** | **39.9** | **37.7** | **100%** | **17,920 B/block (2.3× smaller)** |

Q3 is only 2-7% slower than FP8 with **2.3× less KV-cache memory**.
Python decode loops — CUDA kernel will close the speed gap.

### Q3 Vectorized Decode

| Backend | Short | Med | Long | Math | KV-Cache |
|---------|-------|-----|------|------|----------|
| FP8 (FlashInfer) | 37.2 | 40.7 | 40.3 | 100% | 1× |
| **TQ3 Q3 (vectorized)** | **45.2** | **47.2** | **42.8** | **100%** | **0.44× (2.3× smaller)** |
| Delta vs FP8 | **+22%** | **+16%** | **+6%** | 0% | **-56% memory** |

### GLM-4.7 Full Comparison (Q3 vectorized decode)

| Backend | Short | Med | Long | Math | Page Size | vs FP8 |
|---------|-------|-----|------|------|-----------|--------|
| FP8     | 37.2  | 40.7| 40.3 | 100% | 40,960 B  | 1×     |
| **TQ3** | **45.2** | **47.2** | **42.8** | **100%** | **17,920 B** | **2.3× smaller** |
| **TQ4** | **40.6** | **42.5** | **45.9** | **100%** | **23,040 B** | **1.8× smaller** |
