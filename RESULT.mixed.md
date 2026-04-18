# Mixed Quantization Benchmarks

Benchmark results for mixed weight + KV-cache quantization on GLM-4.7-Flash.

- **Commit**: `ed7520c2a` (tag `tested_mixed`)
- **Platform**: DGX Spark (GB10, SM121, 120 GB Unified)
- **Image**: `localhost/vllm-multiquant` (base `nvcr.io/nvidia/vllm:26.02-py3`)
- **vLLM Version**: 0.15.1+befbc472
- **Model**: `/data/tensordata/GLM-4.7-Flash-int4-AutoRound` (pre-quantized AutoRound INT4)
- **Benchmark**: `bench.py` (deterministic, seed=42, n=5 perf rounds, 50 math problems)
- **Runtime flags**: `--gpu-memory-utilization 0.05 --kv-cache-memory-bytes 5G --max-model-len 4096`

## Baseline (no KV quantization)

| Config                              | Short tok/s | Medium tok/s | Long tok/s | Math   |
|-------------------------------------|-------------|--------------|------------|--------|
| INT4 AutoRound Marlin               |        50.0 |         55.3 |       53.6 | 54 %   |

## KV-Cache Matrix (without MTP)

| KV Cache | Short tok/s | Medium tok/s | Long tok/s | Math   |
|----------|-------------|--------------|------------|--------|
| tq2w     |        46.8 |         50.5 |       46.5 | 52 %   |
| tq3w     |        46.7 |         50.5 |       46.8 | 52 %   |
| tq4w     |        49.2 |         50.4 |       45.9 | **56 %** |

## KV-Cache Matrix (with MTP, ngram nst=1)

| KV Cache | Short tok/s | Medium tok/s | Long tok/s | Math   |
|----------|-------------|--------------|------------|--------|
| tq2w     |         1.6 |         47.1 |       43.3 | 54 %   |
| tq3w     |         1.6 |         46.5 |       45.4 | 52 %   |
| tq4w     |         1.6 |         46.1 |       43.9 | 54 %   |

## Observations

- **TQ accuracy scales with bits**: `tq4w` achieves highest math accuracy (56 % vs 52 %
  for `tq2w`/`tq3w`). Matches the expectation — more bits means less quantization error.
- **TQ throughput is dominated by unified memory bandwidth**: `tq4w` is marginally slower
  on long prompts (45.9 vs 46.5/46.8) because the KV cache packs 18 bytes/block instead
  of 10/14. On short prompts `tq4w` is fastest (49.2) — KV cache hasn't filled yet.
- **MTP (ngram, nst=1) gives ~0 speedup** on math-heavy prompts: ngram draft tokens rarely
  match numerical sequences. The `short` measurement (1.6 tok/s) reflects MTP warmup cost
  amortized over only 20 tokens.
- **Pre-quant ALL INT4 baseline (53.6 tok/s)** is the ceiling; any KV quantization trades
  ~13 % throughput (long) for 2–4× KV cache compression.

## Issues Fixed During Testing

### tq4w was not implemented
The CUDA kernel `tq_wht_pack_to_cache_kernel` in `kernels/turboquant/tq_wht_pack.cu` only
had template instantiations for `MSE_BITS=2` and `MSE_BITS=3`. Any request for `tq4w`
crashed with `RuntimeError: tq_wht_pack_to_cache: unsupported D=256 mse_bits=4`.

Fix (same commit):
1. Added `WHT_THRESHOLDS_4BIT[15]` constant (N(0,1) Lloyd-Max boundaries).
2. Added `threshold_quantize_4bit()` device function.
3. Added `MSE_BITS == 4` branch in the kernel: packs 2 nibbles per byte (16 data bytes +
   2 gamma bytes = 18 bytes/block), matches the decode kernel layout at
   `tq_wht_decode.cu:197`.
4. Added dispatch entries `(D=64/128/256, mse_bits=4)`.

### `"auto"` as dtype value removed
Weight and KV dtypes now reject the string `"auto"`. Either a concrete dtype is given
(`bf16`, `fp8`, `int4`, `tq3w`, …) or `--kv-cache-dtype` is omitted entirely. Removes an
invalid fallback path that silently masked configuration bugs.

### RTN INT4 via real Marlin kernel
`vllm/multiquant/autoround/online_linear.py` now hands INT4 weights to the actual Marlin
kernel (`MarlinLinearKernel` + `PackedvLLMParameter`) instead of the dequant-cache path.
The earlier dequant-cache fallback ran at ~15 tok/s because it defeated the whole point
of keeping weights packed — the fused INT4 GEMM never ran. Throughput after the fix
matches pre-quantized AutoRound Marlin.

## Reproduction

```bash
# Start the server (all flags identical, vary only --kv-cache-dtype and --speculative-config)
podman run -d --replace --name mq-test \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  --ipc=host --network host \
  -v /data/tensordata:/data/tensordata \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v /home/flash/vllm-riy/kernels/turboquant:/opt/tq_build:ro \
  -e VLLM_MLA_DISABLE=1 -e VLLM_WORKER_MULTIPROC_METHOD=fork \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_DISABLE_AUTOTUNER=1 \
  -e VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel \
  -e FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a" \
  localhost/vllm-multiquant \
  vllm serve /data/tensordata/GLM-4.7-Flash-int4-AutoRound \
    --host 0.0.0.0 --port 8011 \
    --served-model-name glm-4.7-flash \
    --gpu-memory-utilization 0.05 --kv-cache-memory-bytes 5G \
    --max-model-len 4096 \
    --kv-cache-dtype tq4w \
    --trust-remote-code
# --speculative-config '{"method":"ngram","num_speculative_tokens":1}'  # add for MTP

python3 bench.py --url http://localhost:8011 --model glm-4.7-flash \
  --label "INT4 + tq4w"
```

The `kernels/turboquant:/opt/tq_build:ro` volume mount picks up the updated `tq_wht_pack.cu`
so the JIT compiler builds the fixed kernel; once the image is rebuilt this mount can be
dropped.
