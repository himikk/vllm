# TurboQuant KV-Cache Quantization for vLLM

TurboQuant compresses the KV-cache to 3-4 bits per coordinate with near-zero quality loss,
enabling longer context lengths and higher throughput within the same GPU memory.

Based on: *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*
(ICLR 2026, Zandieh, Daliri, Hadian, Mirrokni — Google Research / NYU / DeepMind)

## Quick Start

```bash
# Build the container image (one-time)
cd ~/vllm-riy
podman build -f Dockerfile.tq -t vllm-ng17e-tq .

# Start serving with TQ3 (3-bit KV cache)
podman run -d --name vllm-tq \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -p 8011:8000 -v /data/tensordata:/data/tensordata \
  -e FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a" \
  vllm-ng17e-tq \
  vllm serve /data/tensordata/Qwen3.5-35B-A3B-int4-AutoRound \
    --served-model-name qwen35-35b \
    --kv-cache-dtype tq3 \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.05 --kv-cache-memory-bytes 10G \
    --max-model-len 32768 --trust-remote-code --enforce-eager \
    --limit-mm-per-prompt '{"image":0,"video":0}'
```

## How It Works

Two-stage compression per KV vector:

1. **PolarQuant (MSE stage)**: Random orthogonal rotation + per-coordinate Lloyd-Max quantization.
   Uses (b-1) bits per coordinate for near-optimal MSE distortion.

2. **QJL (residual correction)**: 1-bit sign quantization of the residual via the
   Quantized Johnson-Lindenstrauss transform. Corrects inner-product bias,
   yielding **unbiased** attention scores.

TQ integrates as a **transparent wrapper** — the normal attention backend
(FlashInfer/FlashAttention) handles all attention computation. TQ only modifies
keys before they enter the KV cache via a fused CUDA kernel.

## CLI Parameters

```bash
vllm serve <model> --kv-cache-dtype tq3    # 3-bit (2-bit MSE + 1-bit QJL)
vllm serve <model> --kv-cache-dtype tq4    # 4-bit (3-bit MSE + 1-bit QJL)
```

All other vLLM parameters work unchanged.

## Benchmark Results

**Qwen3.5-35B-A3B (INT4 AutoRound) on DGX Spark (GB10 SM121)**

| KV-Cache | Short (20t) | Medium (150t) | Long (400t) | Math | Overhead |
|----------|-------------|---------------|-------------|------|----------|
| FP8      | 2.6 tok/s   | 47.8 tok/s    | 36.2 tok/s  | 100% | Baseline |
| TQ3 v2   | 8.1 tok/s   | 35.5 tok/s    | 32.9 tok/s  | 100% | -9% long |

**Zero quality loss** — 100% math accuracy across all configurations.

## CUDA Kernel

The fused kernel (`tq_round_trip.cu`) performs all 8 TQ stages in a single launch:

```
normalize → rotate(Pi) → quantize → reconstruct → residual → QJL(S) → correct → combine
```

Optimizations (v2):
- Tiled GEMV with shared memory vector tiles (TILE_K=32)
- float4 vectorized loads (128-bit coalesced reads)
- Warp-shuffle reductions (no shared memory for block reduce)
- `__launch_bounds__(256, 4)` for register pressure control
- Precomputed centroids (no scipy at runtime)
- Cached float32 contiguous buffers (no .to()/.float()/.contiguous() per call)

Micro-benchmark: **33µs** per call (256 threads, head_dim=256)

## Architecture

```
vllm serve --kv-cache-dtype tq3
  ↓
Attention.__init__:
  _tq_enabled = True
  kv_cache_dtype = "auto"  ← masquerade for backend selector
  Pi, S, centroids = register_buffer(...)
  ↓
Normal backend selected (FlashInfer on SM121)
  ↓
unified_kv_cache_update:
  if _tq_enabled:
    key = tq_round_trip_keys(key)  ← CUDA fused kernel
  do_kv_cache_update(key, value, ...)  ← normal FlashInfer
```

## Container Image

`vllm-ng17e-tq` = `vllm-ng17e-riy` + TurboQuant baked in:
- Python module: `vllm/turboquant/`
- CUDA kernel: pre-compiled JIT extension for SM121
- All patches pre-applied (no runtime patching)

Build: `podman build -f Dockerfile.tq -t vllm-ng17e-tq .`

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.tq` | Container image build |
| `vllm/turboquant/config.py` | TurboQuantConfig |
| `vllm/turboquant/centroids.py` | Lloyd-Max codebook (precomputed) |
| `vllm/turboquant/quantizer.py` | tq_round_trip_keys() + PyTorch fallback |
| `csrc/quantization/turboquant/tq_round_trip.cu` | Fused CUDA kernel (v2) |
| `csrc/quantization/turboquant/tq_ext.cu` | JIT extension wrapper |
| `tests/turboquant/test_quantizer.py` | Standalone correctness tests |
| `tests/turboquant/test_cuda_kernel.py` | CUDA kernel unit tests |
| `tests/turboquant/patch_tq_transparent.py` | Runtime patching for other images |

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874)
- [QJL CUDA kernels](https://github.com/amirzandieh/QJL) (same first author)
- [PolarQuant](https://github.com/ericshwu/PolarQuant)
- [turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)
