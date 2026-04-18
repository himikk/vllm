#!/bin/bash
# ============================================================
# MultiQuant Quant-on-Load Benchmark Matrix
#
# Tests: 1:1 BF16 baseline, RTN INT3/INT4, pre-quantized INT4,
#        with and without KV-cache compression.
#
# Misst: tok/s, Math Accuracy, Memory (GPU + KV-Cache)
# bench.py NICHT AENDERN (deterministisch, seed=42)
# ============================================================

set -euo pipefail

IMAGE="localhost/vllm-multiquant"
BENCH="$HOME/vllm-riy/bench.py"
NAME="mq-bench"
PORT=8011
MODEL_DIR="/data/tensordata"
MODEL_BF16="GLM-4.7-Flash"
MODEL_INT4="GLM-4.7-Flash-int4-AutoRound"
SERVED="glm-4.7-flash"
RESULTS_DIR="$HOME/vllm-riy/results"
TIMESTAMP=$(date +%Y%m%d_%H%M)

mkdir -p "$RESULTS_DIR"

# Auto-detect GB10 unified memory
if nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | grep -q "N/A\|Not Supported"; then
    GPU_UTIL=0.33
    KV_MEM="--kv-cache-memory-bytes 10G"
else
    GPU_UTIL=0.95
    KV_MEM=""
fi

stop_server() {
    podman stop "$NAME" 2>/dev/null || true
    podman rm "$NAME" 2>/dev/null || true
    sleep 3
}

start_server() {
    local model_path="$1"
    local label="$2"
    shift 2
    local extra_args=("$@")

    echo ""
    echo "================================================================"
    echo "  $label"
    echo "  Model: $model_path"
    echo "  Args:  ${extra_args[*]:-none}"
    echo "================================================================"

    stop_server

    podman run -d --name "$NAME" \
        --device nvidia.com/gpu=all \
        --security-opt=label=disable \
        --hooks-dir=/usr/share/containers/oci/hooks.d \
        --network host \
        -v "$MODEL_DIR:$MODEL_DIR" \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -e VLLM_MLA_DISABLE=1 \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a" \
        -e VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel \
        "$IMAGE" \
        vllm serve "$model_path" \
            --host 0.0.0.0 --port "$PORT" \
            --served-model-name "$SERVED" \
            --gpu-memory-utilization "$GPU_UTIL" \
            --max-model-len 32768 \
            --trust-remote-code \
            $KV_MEM \
            "${extra_args[@]}"

    # Wait for startup
    echo -n "  Waiting... "
    for i in $(seq 1 180); do
        if podman logs "$NAME" 2>&1 | grep -q "Application startup complete"; then
            echo "ready (${i}s)"
            break
        fi
        if podman logs "$NAME" 2>&1 | grep -qE "EngineCore failed|Error|FAILED"; then
            echo "FAILED!"
            podman logs "$NAME" 2>&1 | grep -iE "error|failed|exception" | tail -5
            return 1
        fi
        sleep 1
    done
}

run_bench() {
    local label="$1"
    local outfile="$RESULTS_DIR/${TIMESTAMP}_$(echo "$label" | tr ' /' '_').json"

    echo "  Running bench.py → $outfile"
    python3 "$BENCH" \
        --url "http://localhost:$PORT" \
        --model "$SERVED" \
        --label "$label" \
        --perf-rounds 3 \
        --math-count 50 \
        2>&1 | tee -a "$RESULTS_DIR/${TIMESTAMP}_log.txt"

    # Capture GPU memory
    echo ""
    echo "  GPU Memory:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    echo ""
}

# ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "  MultiQuant Benchmark Matrix"
echo "  $(date)"
echo "  Image: $IMAGE"
echo "=========================================="

# ── 1. BF16 Baseline (1:1, no KV compression) ────────────────
start_server "$MODEL_DIR/$MODEL_BF16" "BF16 Baseline (1:1)" \
    --kv-cache-dtype auto
run_bench "BF16 1:1"

# ── 2. BF16 + TQ3 KV-Cache ───────────────────────────────────
start_server "$MODEL_DIR/$MODEL_BF16" "BF16 + TQ3w KV-Cache" \
    --kv-cache-dtype tq3w
run_bench "BF16 + TQ3w KV"

# ── 3. RTN INT4 Routed + BF16 Shared + TQ3 KV ───────────────
start_server "$MODEL_DIR/$MODEL_BF16" "RTN INT4 Routed + TQ3w KV" \
    --quantization autoround_rtn \
    --weight-dtype-routed int4 \
    --kv-cache-dtype tq3w
run_bench "RTN INT4 Routed + TQ3w KV"

# ── 4. RTN INT3 Routed + BF16 Shared + TQ3 KV ───────────────
start_server "$MODEL_DIR/$MODEL_BF16" "RTN INT3 Routed + TQ3w KV" \
    --quantization autoround_rtn \
    --weight-dtype-routed int3 \
    --kv-cache-dtype tq3w
run_bench "RTN INT3 Routed + TQ3w KV"

# ── 5. RTN INT4 Routed + INT4 Attn + TQ3 KV ─────────────────
start_server "$MODEL_DIR/$MODEL_BF16" "RTN INT4 Routed+Attn + TQ3w KV" \
    --quantization autoround_rtn \
    --weight-dtype-routed int4 \
    --weight-dtype-attn int4 \
    --kv-cache-dtype tq3w
run_bench "RTN INT4 Routed+Attn + TQ3w KV"

# ── 6. Pre-quantized INT4 AutoRound (1:1 load) ──────────────
if [ -d "$MODEL_DIR/$MODEL_INT4" ]; then
    start_server "$MODEL_DIR/$MODEL_INT4" "INT4 AutoRound (pre-quant)" \
        --kv-cache-dtype tq3w
    run_bench "INT4 AutoRound pre-quant + TQ3w KV"
fi

# ── Cleanup ──────────────────────────────────────────────────
stop_server

echo ""
echo "=========================================="
echo "  Done! $(date)"
echo "  Results: $RESULTS_DIR/${TIMESTAMP}_*"
echo "=========================================="
