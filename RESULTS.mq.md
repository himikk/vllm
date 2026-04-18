# MultiQuant Benchmark Results

Platform: DGX Spark (GB10, SM121, 128 GB Unified Memory)

## GLM-4.7-Flash (31B MoE, D=128, rein Transformer)

Model: `unsloth/GLM-4.7-Flash-FP8-Dynamic` (FP8) / `GLM-4.7-Flash-int4-AutoRound` (INT4)

### Vanilla (kein MTP)

| KV-Dtype | Bits | Packed B/Key | tok/s | Math% | KV B/Block | Notes |
|----------|------|-------------|-------|-------|------------|-------|
| fp8      | 8    | 576         | 45.6/51.0/51.7 | 0%* | — | Baseline, TRITON_MLA |
| tq3      | 3    | ~220        | 30.9/50.2/49.9 | 0%* | — | MLA+MQ, 30064 MiB |
| tq3w     | 3.5  | 112         | 35.3 (avg)     | 100 | — | WHT v2, CUDA Graph, fused kernels |
| tq4      | 4    | ~292        |       |       |            | TurboQuant |
| rq2      | 2    | ~148        |       |       |            | RotorQuant |
| rq3      | 3    | ~220        |       |       |            | RotorQuant |
| rq4      | 4    | ~292        |       |       |            | RotorQuant |

*Math 0%: GLM-4.7 Prompt-Issue (antwortet "1"), nicht KV-bezogen

### +MTP NST=1

| KV-Dtype | tok/s | Math% | Notes |
|----------|-------|-------|-------|
| fp8      |       |       |       |
| tq3      |       |       |       |
| tq4      |       |       |       |

## Qwen3.5-35B-A3B (35B MoE, D=256, Hybrid Mamba+Attention)

Model: `Qwen3.5-35B-A3B-int4-AutoRound`

Flags: `--compilation-config '{"cudagraph_mode":"none"}' -e TORCH_CUDNN_V8_API_DISABLED=1`

### Vanilla

| KV-Dtype | Bits | Packed B/Key | tok/s | Math% | KV B/Block | Notes |
|----------|------|-------------|-------|-------|------------|-------|
| fp8      | 8    | 256         |       |       |            | Baseline |
| tq3      | 3    | 100         |       |       |            | TurboQuant |
| tq4      | 4    | 132         |       |       |            | TurboQuant |
| rq2      | 2    | 68          |       |       |            | RotorQuant |
| rq3      | 3    | 100         |       |       |            | RotorQuant |
| rq4      | 4    | 132         |       |       |            | RotorQuant |

## Compression Ratios

| D | Dtype | Packed | vs FP8 | vs BF16 |
|---|-------|--------|--------|---------|
| 128 | fp8  | 128 B  | 1.0×   | 2.0×    |
| 128 | tq3  | 52 B   | 2.5×   | 4.9×    |
| 128 | tq4  | 68 B   | 1.9×   | 3.8×    |
| 128 | rq2  | 36 B   | 3.6×   | 7.1×    |
| 128 | rq3  | 52 B   | 2.5×   | 4.9×    |
| 128 | rq4  | 68 B   | 1.9×   | 3.8×    |
| 256 | fp8  | 256 B  | 1.0×   | 2.0×    |
| 256 | tq3  | 100 B  | 2.6×   | 5.1×    |
| 256 | tq4  | 132 B  | 1.9×   | 3.9×    |
| 256 | rq2  | 68 B   | 3.8×   | 7.5×    |
| 256 | rq3  | 100 B  | 2.6×   | 5.1×    |
| 256 | rq4  | 132 B  | 1.9×   | 3.9×    |

## WHT TurboQuant v2 (tq3w, CUDA Graphs, GLM-4.7-Flash, D=256)

Offline-Benchmark (kein HTTP/Scheduler-Overhead), 5 Runs avg:

| KV-Dtype | tok/s (avg) | Math | Mozart | KV vs FP8 | Notes |
|----------|------------|------|--------|-----------|-------|
| fp8      | 43.2       | 788✓ | ✓      | 1.0×      | FlashInfer decode, CUDA Graph |
| tq2w     | **41.4**   | 788✓ | ✓      | 3.2× less | 2-bit WHT, Split-KV, fused pack-to-cache |
| tq3w     | **41.2**   | 788✓ | ✓      | 2.3× less | 3-bit WHT, Split-KV, fused pack-to-cache |
| tq3r     | 34.7       | 788✓ | ✓      | 2.3× less | Block-rot 32×32, fused CUDA pack+decode |

tq2w = **96% of FP8** mit **3.2× weniger KV**, tq3w = **95%** mit **2.3× weniger**.

## Serve-Mode Benchmarks (bench.py, HTTP, GLM-4.7-Flash)

| Modell-Quant | Archer Gewichte | KV-Cache | tok/s (s/m/l) | Math |
|-------------|-----------------|----------|---------------|------|
| **INT4 AutoRound** | — | tq3w | **45/51/47** | 70% |
| FP8 | — | tq3w | 34/40/38 | 80% |
| FP8 | — | tq2w | 35/40/38 | 60% |
| BF16 | TQ3 (cos=0.63) | tq3w | ~0.2 (eager) | **Müll** (7×8=10.12) |

Math%: GLM-4.7 Prompt-Varianz bei n=10 (nicht KV-bezogen).
INT4+tq3w ist schnellste Kombi weil INT4 weniger VRAM → mehr KV-Cache-Headroom.

### Archer Weight Quantization — NICHT BRAUCHBAR

| Archer Methode | cos (Unit-Test) | Modell-Test | Status |
|---------------|-----------------|-------------|--------|
| TQ3 | 0.63 | 7×8=10.12 (Müll) | ✗ Qualität zu niedrig |
| RQ3 | 0.92 | CUDA Assert in Clifford | ✗ Dimensions-Bug (D nicht durch 3 teilbar) |

Archer Gewichtskompression funktioniert nicht für GLM-4.7-Flash.
KV-Cache WHT (tq2w/tq3w) bleibt die einzige funktionierende Kompression.

### Kernel-Level Profiling (CUDA Event Timing, B=1, D=256, 40 Layers)

| Kernel | Time/call | Time/token (40L) | Share |
|--------|-----------|-------------------|-------|
| WHT Pack (K+V) | 2.3 us | 0.19 ms | 0.7% |
| WHT Fused Decode (sl=50) | 58.2 us | 2.33 ms | 8.2% |
| **Total Attention** | — | **2.52 ms** | **8.9%** |
| Other (MoE, norms, LM head) | — | ~25.8 ms | 91.1% |

Decode kernel scales linear: 10.3 us pro seq-Position pro Layer.

Gap-Analyse: Die 18% Differenz (5.4 ms/token) kommt NICHT von den
Attention-Kernels (nur 2.5 ms total), sondern vom Graph-Overhead:
tq3w Graphs sind ~4.6 GiB (FP8 ~2 GiB) wegen mehr Kernel-Launches
pro Layer (2× pack + decode + scatter vs 1× FlashInfer + reshape_and_cache).

Features: fused WHT pack kernel (1 launch), in-kernel WHT decode, bf16 zero-copy,
CUDA Graph compatible, warp-shuffle WHT (5 butterfly stages).

## Bisherige TQ-Ergebnisse (Q4, GLM-4.7-Flash, D=128)

Referenz aus früheren Benchmarks (eager, serve-mode):

| KV-Dtype | tok/s (short/med/long) | Math% | KV B/Block |
|----------|----------------------|-------|------------|
| fp8      | 37.2 / 40.7 / 40.3  | 100   | 40,960     |
| tq3      | 45.2 / 47.2 / 42.8  | 100   | 17,920     |
| tq4      | 40.6 / 42.5 / 45.9  | 100   | 23,040     |
