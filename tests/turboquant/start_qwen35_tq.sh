#!/bin/bash
# TurboQuant test with Qwen3.5-35B-A3B on DGX Spark
#
# Mounts TurboQuant code into the vllm-ng17e-riy container.
# Phase 1: Run standalone tests inside container to validate GPU.
# Phase 2: Start serving with --kv-cache-dtype tq3.

set -euo pipefail

MODEL=/data/tensordata/Qwen3.5-35B-A3B-int4-AutoRound
NAME=vllm-qwen35-tq
IMAGE=vllm-ng17e-riy
VLLM_SRC=/home/flash/vllm-riy
VLLM_DST=/usr/local/lib/python3.12/dist-packages/vllm

echo "============================================"
echo "TurboQuant + Qwen3.5-35B-A3B — $(hostname)"
echo "Image: $IMAGE"
echo "Model: $MODEL"
echo "============================================"

# Stop existing container
podman stop $NAME 2>/dev/null; podman rm $NAME 2>/dev/null

MODE="${1:-test}"

if [ "$MODE" = "test" ]; then
    echo ""
    echo "=== Running TQ tests inside container ==="
    podman run --rm \
      --device nvidia.com/gpu=all \
      --security-opt=label=disable \
      --hooks-dir=/usr/share/containers/oci/hooks.d \
      -v /data/tensordata:/data/tensordata \
      -v ${VLLM_SRC}/vllm/turboquant:${VLLM_DST}/turboquant:ro \
      -v ${VLLM_SRC}/vllm/config/cache.py:${VLLM_DST}/config/cache.py:ro \
      -v ${VLLM_SRC}/vllm/utils/torch_utils.py:${VLLM_DST}/utils/torch_utils.py:ro \
      -v ${VLLM_SRC}/vllm/v1/attention/backend.py:${VLLM_DST}/v1/attention/backend.py:ro \
      -v ${VLLM_SRC}/vllm/v1/attention/backends/registry.py:${VLLM_DST}/v1/attention/backends/registry.py:ro \
      -v ${VLLM_SRC}/vllm/v1/attention/backends/turboquant_attn.py:${VLLM_DST}/v1/attention/backends/turboquant_attn.py:ro \
      -v ${VLLM_SRC}/vllm/v1/attention/ops/triton_tq_reshape_and_cache.py:${VLLM_DST}/v1/attention/ops/triton_tq_reshape_and_cache.py:ro \
      -v ${VLLM_SRC}/vllm/v1/attention/ops/triton_tq_attention_score.py:${VLLM_DST}/v1/attention/ops/triton_tq_attention_score.py:ro \
      -v ${VLLM_SRC}/vllm/model_executor/layers/attention/attention.py:${VLLM_DST}/model_executor/layers/attention/attention.py:ro \
      -v ${VLLM_SRC}/vllm/platforms/cuda.py:${VLLM_DST}/platforms/cuda.py:ro \
      -v ${VLLM_SRC}/tests/turboquant:/opt/tests/turboquant:ro \
      --shm-size=16g \
      $IMAGE \
      bash -c "
        pip install --break-system-packages scipy 2>/dev/null | tail -1
        echo ''
        echo '=== Standalone Quantizer Tests ==='
        python3 /opt/tests/turboquant/test_quantizer.py
        echo ''
        echo '=== Cache Pipeline Tests ==='
        python3 /opt/tests/turboquant/test_cache_pipeline.py
        echo ''
        echo '=== GPU Benchmark ==='
        python3 -c \"
import torch, time
from vllm.turboquant.config import TurboQuantConfig
from vllm.turboquant.quantizer import TurboQuantizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

d = 128
for bits in [3, 4]:
    config = TurboQuantConfig(head_dim=d, total_bits=bits)
    tq = TurboQuantizer(config, layer_idx=0).to(device)
    x = torch.randn(4096, d, device=device)

    # Warmup
    for _ in range(3): tq.quantize(x)
    if device == 'cuda': torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(50): compressed = tq.quantize(x)
    if device == 'cuda': torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 50

    print(f'  tq{bits}: quantize 4096 vecs: {t*1000:.2f}ms ({4096/t:.0f} vecs/s)')
\"
        echo ''
        echo '=== Config Integration Check ==='
        python3 -c \"
from vllm.config.cache import CacheDType
print(f'tq3 in CacheDType: {\\\"tq3\\\" in str(CacheDType)}')
print(f'tq4 in CacheDType: {\\\"tq4\\\" in str(CacheDType)}')
from vllm.v1.attention.backend import is_quantized_kv_cache
print(f'is_quantized(tq3): {is_quantized_kv_cache(\\\"tq3\\\")}')
print(f'is_quantized(fp8): {is_quantized_kv_cache(\\\"fp8\\\")}')
from vllm.v1.attention.backends.registry import AttentionBackendEnum
print(f'TURBOQUANT backend: {AttentionBackendEnum.TURBOQUANT.value}')
\"
      "

elif [ "$MODE" = "serve" ]; then
    echo ""
    echo "=== Starting vLLM with TurboQuant KV-cache ==="
    echo "NOTE: This is experimental — MetadataBuilder is a stub!"
    echo ""

    podman run -d \
      --name $NAME \
      --device nvidia.com/gpu=all \
      --security-opt=label=disable \
      --hooks-dir=/usr/share/containers/oci/hooks.d \
      -p 8011:8000 \
      -v /data/tensordata:/data/tensordata \
      -v ${VLLM_SRC}/vllm/turboquant:${VLLM_DST}/turboquant:ro \
      -v ${VLLM_SRC}/vllm/config/cache.py:${VLLM_DST}/config/cache.py:ro \
      -v ${VLLM_SRC}/vllm/utils/torch_utils.py:${VLLM_DST}/utils/torch_utils.py:ro \
      -v ${VLLM_SRC}/vllm/v1/attention/backend.py:${VLLM_DST}/v1/attention/backend.py:ro \
      -v ${VLLM_SRC}/vllm/v1/attention/backends/registry.py:${VLLM_DST}/v1/attention/backends/registry.py:ro \
      -v ${VLLM_SRC}/vllm/v1/attention/backends/turboquant_attn.py:${VLLM_DST}/v1/attention/backends/turboquant_attn.py:ro \
      -v ${VLLM_SRC}/vllm/v1/attention/ops/triton_tq_reshape_and_cache.py:${VLLM_DST}/v1/attention/ops/triton_tq_reshape_and_cache.py:ro \
      -v ${VLLM_SRC}/vllm/v1/attention/ops/triton_tq_attention_score.py:${VLLM_DST}/v1/attention/ops/triton_tq_attention_score.py:ro \
      -v ${VLLM_SRC}/vllm/model_executor/layers/attention/attention.py:${VLLM_DST}/model_executor/layers/attention/attention.py:ro \
      -v ${VLLM_SRC}/vllm/platforms/cuda.py:${VLLM_DST}/platforms/cuda.py:ro \
      --shm-size=16g \
      -e VLLM_MLA_DISABLE=1 \
      -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
      $IMAGE \
      bash -c "pip install --break-system-packages scipy 2>/dev/null; \
        vllm serve $MODEL \
          --host 0.0.0.0 \
          --port 8000 \
          --served-model-name qwen35-35b-tq3 \
          --gpu-memory-utilization 0.05 \
          --kv-cache-memory-bytes 10G \
          --max-model-len 32768 \
          --trust-remote-code \
          --quantization auto_round \
          --kv-cache-dtype tq3 \
          --enforce-eager"

    echo "Container $NAME started on port 8011"
    echo "Logs: podman logs -f $NAME"

else
    echo "Usage: $0 [test|serve]"
    exit 1
fi
