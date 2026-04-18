#!/bin/bash
# Bench weight quantization variants
# Tests: native weights (INT4/FP8/BF16) vs Archer online-quant (TQ3/RQ3)
# Usage: ./bench_weights.sh
set -euo pipefail
cd "$(dirname "$0")"

KV=tq3w
RESULTS=()

test_variant() {
    local LABEL=$1
    local MODEL=$2
    local WEIGHTS=$3  # "" = native, "tq3"/"rq3" = Archer

    echo ""
    echo "══════════ $LABEL ══════════"
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

    local MTP=${4:-false}
    local KV_OVERRIDE=${5:-""}

    # KV dtype: default from global KV, override per-test if specified
    local KV_USE="$KV"
    if [ -n "$KV_OVERRIDE" ]; then
        KV_USE="$KV_OVERRIDE"
    fi

    local ARGS=(--model "$MODEL" --kv "$KV_USE")
    if [ -n "$WEIGHTS" ]; then
        ARGS+=(--weights "$WEIGHTS")
    fi
    if [ "$MTP" = "true" ]; then
        ARGS+=(--mtp)
    fi
    ./start.multiquant "${ARGS[@]}"

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

    # Run bench.py (perf + math)
    local SAFE_LABEL=$(echo "$LABEL" | tr '/ ' '__')
    echo "  Benchmarking..."
    python3 "$(dirname "$0")/bench.py" \
        --url http://localhost:8011 \
        --model glm-4.7-flash \
        --label "$LABEL" \
        --perf-rounds 3 \
        --math-count 10 2>&1 | tee /tmp/bench_${SAFE_LABEL}.log | grep -E "^\s+(short|medium|long|Math):"

    local TPS=$(grep "short" /tmp/bench_${SAFE_LABEL}.log 2>/dev/null | awk '{print $2}' | head -1)
    local MATH=$(grep "Math:" /tmp/bench_${SAFE_LABEL}.log 2>/dev/null | head -1 | sed 's/.*Math: //')
    RESULTS+=("$LABEL: short=${TPS:-?} tok/s  $MATH")
}

echo "MultiQuant Weight Bench"
echo "═══════════════════════════════════════════════════════════════════"
printf "  %-4s  %-12s  %-10s  %-6s  %-3s\n" "#" "Modell-Quant" "Archer" "KV" "MTP"
echo "───────────────────────────────────────────────────────────────────"

#                    Label                               Model                            Archer  MTP
#       Modell-Quant | Archer-Gewichte | KV-Cache | MTP
test_variant "INT4  / —    / tq3w / —"    "GLM-4.7-Flash-int4-AutoRound"  ""      ""
test_variant "FP8   / —    / tq3w / —"    "GLM-4.7-Flash-FP8"            ""      ""
test_variant "FP8   / —    / tq2w / —"    "GLM-4.7-Flash-FP8"            ""      ""     "tq2w"
test_variant "BF16  / tq3  / tq3w / —"    "GLM-4.7-Flash"               "tq3"   ""
test_variant "BF16  / rq3  / tq3w / —"    "GLM-4.7-Flash"               "rq3"   ""

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  ERGEBNISSE"
echo "───────────────────────────────────────────────────────────────────"
printf "  %-30s  %s\n" "Modell / Archer / KV / MTP" "tok/s     mem"
echo "───────────────────────────────────────────────────────────────────"
for r in "${RESULTS[@]}"; do
    printf "  %-30s  %s\n" "$(echo "$r" | cut -d: -f1)" "$(echo "$r" | cut -d: -f2-)"
done
echo "───────────────────────────────────────────────────────────────────"
echo "  FP8 Baseline (ohne MQ):       37-40 tok/s"
echo "═══════════════════════════════════════════════════════════════════"

podman stop mq-serve 2>/dev/null || true
