#!/bin/bash
# MultiQuant Benchmark Matrix
# BF16 Models + Archer Weight Quant + MQ KV-Cache
#
# Matrix:
#   GLM-4.7-Flash BF16 + TQ3 (weights+KV)
#   GLM-4.7-Flash BF16 + RQ3 (weights+KV)
#   Qwen3.5-35B BF16 + TQ3 (weights+KV)
#   Qwen3.5-35B BF16 + RQ3 (weights+KV)

set -euo pipefail

IMAGE="localhost/vllm-mq-bench"
BENCH="/home/flash/vllm-marlin-sm12x/bench.py"
NAME="mq-bench"
PORT=8011

run_bench() {
    local model_path="$1"
    local model_name="$2"
    local kv_dtype="$3"
    local label="$4"
    local extra_flags="${5:-}"

    echo ""
    echo "================================================================"
    echo "  $label"
    echo "  Model: $model_path"
    echo "  KV: $kv_dtype  Weights: multiquant ($kv_dtype)"
    echo "================================================================"

    podman stop "$NAME" 2>/dev/null || true
    podman rm "$NAME" 2>/dev/null || true
    sleep 2

    podman run -d --name "$NAME" \
        --device nvidia.com/gpu=all \
        --security-opt=label=disable \
        --hooks-dir=/usr/share/containers/oci/hooks.d \
        -p ${PORT}:8000 \
        -v /data/tensordata:/data/tensordata \
        -e TORCH_CUDNN_V8_API_DISABLED=1 \
        "$IMAGE" \
        vllm serve "$model_path" \
            --host 0.0.0.0 --port 8000 \
            --served-model-name "$model_name" \
            --gpu-memory-utilization 0.05 \
            --kv-cache-memory-bytes 10G \
            --kv-cache-dtype "$kv_dtype" \
            --quantization multiquant \
            --max-model-len 32768 \
            --compilation-config '{"cudagraph_mode":"none"}' \
            --trust-remote-code \
            $extra_flags

    # Wait for startup
    echo "Waiting for startup..."
    for i in $(seq 1 120); do
        if podman logs "$NAME" 2>&1 | grep -q "Application startup complete"; then
            echo "  Ready after ${i}0s"
            break
        fi
        if podman logs "$NAME" 2>&1 | grep -q "EngineCore failed"; then
            echo "  FAILED!"
            podman logs "$NAME" 2>&1 | grep "ERROR" | tail -3
            return 1
        fi
        sleep 10
    done

    # Memory
    GPU_MEM=$(nvidia-smi 2>&1 | grep MiB | tail -1 | awk '{print $9}')
    KV_TOKENS=$(podman logs "$NAME" 2>&1 | grep "GPU KV cache size" | tail -1 | grep -oP '[\d,]+' | head -1)
    echo "  GPU Memory: $GPU_MEM"
    echo "  KV Tokens: $KV_TOKENS"

    # Bench
    echo "  Running bench.py..."
    python3 "$BENCH" --url "http://localhost:$PORT" --model "$model_name" --label "$label"

    echo ""
}

echo "=========================================="
echo "  MultiQuant Benchmark Matrix"
echo "  $(date)"
echo "=========================================="

# 1. GLM-4.7-Flash BF16 + TQ3
run_bench "/data/tensordata/GLM-4.7-Flash" "glm47-tq3" "tq3" \
    "GLM-4.7 BF16→TQ3 (W+KV)"

# 2. GLM-4.7-Flash BF16 + RQ3
run_bench "/data/tensordata/GLM-4.7-Flash" "glm47-rq3" "rq3" \
    "GLM-4.7 BF16→RQ3 (W+KV)"

# 3. Qwen3.5-35B BF16 + TQ3
run_bench "/data/tensordata/Qwen3.5-35B-A3B" "qwen35-tq3" "tq3" \
    "Qwen3.5-35B BF16→TQ3 (W+KV)"

# 4. Qwen3.5-35B BF16 + RQ3
run_bench "/data/tensordata/Qwen3.5-35B-A3B" "qwen35-rq3" "rq3" \
    "Qwen3.5-35B BF16→RQ3 (W+KV)"

echo ""
echo "=========================================="
echo "  Done! $(date)"
echo "=========================================="
