# MultiQuant — Benchmark Results

## Platform: RTX PRO 6000 (SM120, 96 GiB, 1800 GB/s)

### Model: GLM-4.7-Flash INT4 AutoRound (Marlin Kernel)

| KV-Cache | tok/s | Kernel | CG Mode | KV/Token | Status |
|----------|-------|--------|---------|----------|--------|
| **TQ3** | **55.5** | Triton | PIECEWISE | 17,920 B (2.3×) | ✓ |
| **TQ4** | **52.3** | Triton | PIECEWISE | 23,040 B (1.8×) | ✓ |
| RQ3 | TBD | Triton | PIECEWISE | 17,920 B (2.3×) | Testing |
| FP8 (Baseline) | ~137 | FlashInfer | FULL | 40,960 B | ✓ |

---

## Platform: DGX Spark (GB10, SM121, 120 GiB Unified, 273 GB/s)

### Model: GLM-4.7-Flash INT4 AutoRound (Marlin Kernel)

| KV-Cache | tok/s | Kernel | CG Mode | KV/Token | Status |
|----------|-------|--------|---------|----------|--------|
| **FP8** (Baseline) | 37-40 | FlashInfer | FULL | 40,960 B | ✓ |
| **TQ3** | **35-49** | CUDA+Triton | UNIFORM_DECODE | 17,920 B (2.3×) | ✓ |
| **TQ4** | **34.1** | CUDA+Triton | UNIFORM_DECODE | 23,040 B (1.8×) | ✓ |
| RQ3 | TBD | Triton | PIECEWISE | 17,920 B (2.3×) | Rebuild needed |
| RQ4 | TBD | Triton | PIECEWISE | 23,040 B (1.8×) | Rebuild needed |
| RQ2 | TBD | Triton | PIECEWISE | 12,800 B (3.2×) | Rebuild needed |

### DGX TQ3 Detail (bench.multiquant.sh)

| Output Length | tok/s | vs FP8 |
|---------------|-------|--------|
| Short (16 tok) | 12.3 | -67% (prefill overhead) |
| Medium (128 tok) | 38.1 | -6% |
| Long (512 tok) | **49.2** | **+22%** |

---

## Quantizer Quality (cos similarity, D=128, seed=42)

| Variante | Archer Decompress | Attention Decode |
|----------|------------------|-----------------|
| TQ3 | 0.92 | ✓ |
| TQ4 | 0.97 | ✓ |
| RQ3 | 0.92 | TBD |
| RQ4 | 0.97 | TBD |
| RQ2 | 0.80 | TBD |

## Unit Tests

| Count | Status |
|-------|--------|
| 158 | passed |
| 2 | failed (torch.compile — Custom Op WIP) |
| 0 | xfail |

---

## Architecture

```
Prefill:   bmm causal attention (raw K/V)
KV Write:  pack_vectors_batched → slot_mapping → uint8 cache
Decode:    Pre-rotate Q → Fused kernel (score+softmax+V-acc) → Post-GEMV
           TQ: q @ Pi.T / centroid_acc @ Pi (cuBLAS)
           RQ: Clifford rotor sandwich (PIECEWISE)
```

### Kernel Launches per Decode Token

| Path | Launches | CUDA Graph |
|------|----------|------------|
| Python loop (old) | ~100 | ✗ |
| **Triton fused** | **5** | PIECEWISE |
| **CUDA fused (TQ)** | **5** | UNIFORM_DECODE |

### CUDA Kernel Performance (isolated, DGX)

```
tq_fused_decode_kernel: 0.1ms/call = 6,700 tok/s (decode only)
```

---

## Bench Commands

```bash
# Quick bench (short/medium/long tok/s)
./bench.multiquant.sh

# All KV variants (perf + math + memory)
./bench_all_kv.sh

# Weight variants (INT4/FP8/Archer + MTP)
./bench_weights.sh

# Deterministic bench (50 math, context scaling)
python3 bench.py --url http://localhost:8011 --model glm-4.7-flash --label "..." --context
```

---

*Update with `./bench_all_kv.sh` and `./bench_weights.sh`*
