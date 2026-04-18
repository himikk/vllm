# TASK: MultiQuant Benchmark Matrix

## Ziel

Vollständige Benchmark-Matrix für alle KV-Cache-Quantisierungsvarianten auf DGX Spark (SM121).

## Matrix

### Modelle
- GLM-4.7-Flash (31B MoE, rein Transformer, D=128)
- Qwen3.5-35B-A3B (35B MoE, Hybrid Mamba+Attention, D=256)

### KV-Cache Dtypes
| Dtype | Bits | Packed (D=128) | Packed (D=256) | Methode |
|-------|------|----------------|----------------|---------|
| fp8   | 8    | 128 B          | 256 B          | Baseline |
| tq3   | 3    | 52 B           | 100 B          | TurboQuant (Dense Pi) |
| tq4   | 4    | 68 B           | 132 B          | TurboQuant (Dense Pi) |
| rq2   | 2    | 36 B           | 68 B           | RotorQuant (Clifford) |
| rq3   | 3    | 52 B           | 100 B          | RotorQuant (Clifford) |
| rq4   | 4    | 68 B           | 132 B          | RotorQuant (Clifford) |

### Speculative Decoding
- Vanilla (kein MTP)
- +MTP NST=1 (wo verfügbar)

### Metriken
- tok/s (Generation, bench.py)
- Math Accuracy (%)
- KV-Cache Size (B/block)
- GPU Memory (nvidia-smi)

## Benchmarks

### GLM-4.7-Flash (D=128, rein Transformer)
- [ ] FP8 Vanilla
- [ ] FP8 +MTP
- [ ] TQ3 Vanilla
- [ ] TQ3 +MTP
- [ ] TQ4 Vanilla
- [ ] TQ4 +MTP
- [ ] RQ2 Vanilla
- [ ] RQ3 Vanilla
- [ ] RQ4 Vanilla

### Qwen3.5-35B (D=256, Hybrid Mamba+Attention)
- [ ] FP8 Vanilla
- [ ] TQ3 Vanilla
- [ ] TQ4 Vanilla
- [ ] RQ2 Vanilla
- [ ] RQ3 Vanilla
- [ ] RQ4 Vanilla

## Setup

```bash
# Image: vllm-riy-tq-bm (oder neues vllm-mq-bench)
# Port: 8011
# Memory: --gpu-memory-utilization 0.05 --kv-cache-memory-bytes 10G
# Qwen3.5: --compilation-config '{"cudagraph_mode":"none"}' -e TORCH_CUDNN_V8_API_DISABLED=1
# GLM-4.7: CUDA Graphs OK (rein Transformer)
# bench.py: seed=42, deterministic
```

## Hinweise

- bench.py NICHT ändern (seed=42)
- Port IMMER 8011
- Container stoppen+entfernen vor jedem neuen Test
- RQ nutzt aktuell gleichen Attention-Backend-Code wie TQ (identisches Packed Format)
- RQ2 = 1-bit MSE + 1-bit QJL, sehr aggressiv → Qualität prüfen
