#!/bin/bash
# TurboQuant E2E test on DGX Spark (GB10, SM121)
#
# Prerequisites:
#   - vllm-riy installed: cd ~/vllm-riy && pip install -e . --break-system-packages
#   - scipy installed: pip install scipy --break-system-packages
#   - GLM-4.7-Flash FP8 model at /data/tensordata/unsloth/GLM-4.7-Flash-FP8-Dynamic
#
# Usage:
#   bash tests/turboquant/run_on_dgx.sh [test|serve]
#
# test:  Run standalone CPU + GPU tests
# serve: Start vLLM server with --kv-cache-dtype tq3

set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-test}"

echo "============================================"
echo "TurboQuant E2E — $(hostname)"
echo "Mode: $MODE"
echo "============================================"

if [ "$MODE" = "test" ]; then
    echo ""
    echo "=== Standalone Tests (CPU) ==="
    python3 tests/turboquant/test_quantizer.py
    echo ""
    echo "=== Cache Pipeline Tests (CPU) ==="
    python3 tests/turboquant/test_cache_pipeline.py
    echo ""
    echo "=== GPU Quantizer Benchmark ==="
    python3 -c "
import torch, time, math
from vllm.turboquant.config import TurboQuantConfig
from vllm.turboquant.quantizer import TurboQuantizer

if not torch.cuda.is_available():
    print('No GPU, skipping')
    exit(0)

device = 'cuda'
d = 128
for bits in [3, 4]:
    config = TurboQuantConfig(head_dim=d, total_bits=bits)
    tq = TurboQuantizer(config, layer_idx=0).to(device)

    # Warmup
    x = torch.randn(1024, d, device=device)
    for _ in range(5):
        tq.quantize(x)
    torch.cuda.synchronize()

    # Benchmark quantize
    N = 4096
    x = torch.randn(N, d, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        compressed = tq.quantize(x)
    torch.cuda.synchronize()
    t_quant = (time.perf_counter() - t0) / 100

    # Benchmark dequant
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        recon = tq.dequantize(compressed)
    torch.cuda.synchronize()
    t_deq = (time.perf_counter() - t0) / 100

    print(f'  tq{bits}: quant {N} vecs: {t_quant*1000:.2f}ms, dequant: {t_deq*1000:.2f}ms')
    print(f'        throughput: {N/t_quant:.0f} vecs/s quant, {N/t_deq:.0f} vecs/s dequant')
"

elif [ "$MODE" = "serve" ]; then
    echo ""
    echo "=== Starting vLLM with TurboQuant KV-cache ==="
    echo "Model: GLM-4.7-Flash FP8"
    echo "KV-Cache: tq3 (3-bit TurboQuant)"
    echo ""

    # Note: This will fail until the MetadataBuilder is fully implemented.
    # For now, it tests that the config path works end-to-end.
    python3 -m vllm.entrypoints.openai.api_server \
        --model /data/tensordata/unsloth/GLM-4.7-Flash-FP8-Dynamic \
        --served-model-name glm-4.7-flash \
        --host 0.0.0.0 \
        --port 8011 \
        --kv-cache-dtype tq3 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --trust-remote-code \
        --enforce-eager

else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [test|serve]"
    exit 1
fi
