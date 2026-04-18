#!/bin/bash
# Bench all KV cache variants sequentially
# Usage: ./bench_all_kv.sh [model_name]
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-GLM-4.7-Flash-int4-AutoRound}"
RESULTS=()

test_kv() {
    local KV=$1
    local MDL=${2:-$MODEL}
    local LABEL="${MDL##*/} / $KV"
    echo ""
    echo "══════════ $LABEL ══════════"

    # Stop any running serve containers
    podman stop mq-serve 2>/dev/null || true
    podman rm mq-serve 2>/dev/null || true

    # Check for competing GPU processes
    local GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | grep -v "^$" | wc -l)
    if [ "$GPU_PROCS" -gt 0 ]; then
        echo "  WARNING: $GPU_PROCS GPU process(es) still running:"
        nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        echo "  Waiting 10s for cleanup..."
        sleep 10
    fi
    sleep 2

    ./start.multiquant --model "$MDL" --kv "$KV"

    echo -n "  Warte... "
    for i in $(seq 1 600); do
        if curl -s http://localhost:8011/v1/models 2>/dev/null | grep -q "glm\|qwen\|llama"; then
            echo "ready (${i}x5s)"
            break
        fi
        if [ $i -gt 3 ] && podman logs mq-serve 2>&1 | tail -5 | grep -q "RuntimeError"; then
            if podman logs mq-serve 2>&1 | grep -q "out of memory\|OOM\|less than desired.*memory"; then
                echo "OOM"
                RESULTS+=("$LABEL: OOM")
            else
                echo "CRASHED"
                podman logs mq-serve 2>&1 | grep "Error:" | tail -1
                RESULTS+=("$LABEL: CRASHED")
            fi
            return
        fi
        if [ $i -eq 600 ]; then
            echo "TIMEOUT"
            RESULTS+=("$LABEL: TIMEOUT")
            return
        fi
        sleep 5
    done

    # Run bench.py (perf + math + optional context)
    echo "  Benchmarking..."
    python3 "$(dirname "$0")/bench.py" \
        --url http://localhost:8011 \
        --model glm-4.7-flash \
        --label "$LABEL" \
        --perf-rounds 3 \
        --math-count 10 2>&1 | tee "/tmp/bench_kv_${KV}_${MDL##*/}.log" | grep -E "^\s+(short|medium|long|Math):"

    # Extract summary for results table
    local TPS=$(grep "short" "/tmp/bench_kv_${KV}_${MDL##*/}.log" 2>/dev/null | awk '{print $2}' | head -1)
    local MATH=$(grep "Math:" "/tmp/bench_kv_${KV}_${MDL##*/}.log" 2>/dev/null | head -1 | sed 's/.*Math: //')
    RESULTS+=("$LABEL: short=${TPS:-?} tok/s  $MATH")
}

echo "MultiQuant KV Bench — Model: $MODEL"
echo "════════════════════════════════════════"

# INT4 model + all KV variants
for KV in auto fp8 tq3 tq4 rq3 rq4 rq2; do
    test_kv "$KV" "$MODEL"
done

# BF16 model baselines (same KV variants, different weight format)
BF16_MODEL="GLM-4.7-Flash"
if [ -d "/data/tensordata/$BF16_MODEL" ]; then
    for KV in auto fp8 tq3 tq4; do
        test_kv "$KV" "$BF16_MODEL"
    done
fi

echo ""
echo "════════════════════════════════════════"
echo "  ERGEBNISSE"
echo "════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    printf "  %-6s %s\n" "$(echo "$r" | cut -d: -f1):" "$(echo "$r" | cut -d: -f2-)"
done
echo ""
echo "  auto=BF16 KV, fp8=FP8 KV"
echo "════════════════════════════════════════"

podman stop mq-serve 2>/dev/null || true
