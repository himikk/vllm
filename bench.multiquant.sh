#!/bin/bash
# Quick benchmark: 3 Requests mit unterschiedlicher Länge
# Misst tok/s für Short/Medium/Long Output

set -euo pipefail

PORT="${1:-8011}"
URL="http://localhost:$PORT/v1/chat/completions"

# Auto-detect model name from server
MODEL=$(curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | python3 -c "
import sys,json
try: print(json.load(sys.stdin)['data'][0]['id'])
except: print('glm-4.7-flash')" 2>/dev/null || echo "glm-4.7-flash")

echo "MultiQuant Bench — $URL"
echo "================================"

# Warten bis Server bereit
echo -n "Warte auf Server... "
for i in $(seq 1 120); do
    if curl -s "$URL" -H "Content-Type: application/json" \
       -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
       2>/dev/null | grep -q "content"; then
        echo "OK"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

bench() {
    local label="$1"
    local prompt="$2"
    local max_tokens="$3"

    local start=$(date +%s%N)
    local response=$(curl -s "$URL" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'$MODEL'",
            "messages": [{"role": "user", "content": "'"$prompt"'"}],
            "max_tokens": '$max_tokens',
            "temperature": 0
        }')
    local end=$(date +%s%N)

    local tokens=$(echo "$response" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    print(r['usage']['completion_tokens'])
except: print(0)")

    local elapsed_ms=$(( (end - start) / 1000000 ))
    local elapsed_s=$(python3 -c "print(f'{$elapsed_ms/1000:.2f}')")

    if [ "$tokens" -gt 0 ]; then
        local tps=$(python3 -c "print(f'{$tokens / ($elapsed_ms / 1000):.1f}')")
        printf "  %-12s %4d tok / %5ss = %6s tok/s\n" "$label" "$tokens" "$elapsed_s" "$tps"
    else
        printf "  %-12s FEHLER (keine Tokens)\n" "$label"
    fi
}

# Warmup
echo "Warmup..."
curl -s "$URL" -H "Content-Type: application/json" \
    -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"Zähle bis 10"}],"max_tokens":50}' > /dev/null

echo ""
echo "Benchmark:"
bench "Short"  "Was ist 2+2? Antworte nur mit der Zahl." 16
bench "Medium" "Erkläre was ein Transformer ist in 3 Sätzen." 128
bench "Long"   "Schreibe eine ausführliche Erklärung von Attention in neuronalen Netzen." 512

# Memory usage (via container python)
echo ""
echo "Speicher:"
CONTAINER=$(podman ps --filter "publish=8011" --format "{{.Names}}" | head -1)
if [ -n "$CONTAINER" ]; then
    podman exec "$CONTAINER" python3 -c "
import torch
alloc = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
max_alloc = torch.cuda.max_memory_allocated() / 1024**3
free, total = torch.cuda.mem_get_info()
print(f'  Allocated:     {alloc:.2f} GiB')
print(f'  Reserved:      {reserved:.2f} GiB')
print(f'  Peak:          {max_alloc:.2f} GiB')
print(f'  GPU Free:      {free/1024**3:.2f} / {total/1024**3:.2f} GiB')
" 2>/dev/null || echo "  (Container nicht erreichbar)"
fi

echo ""
echo "Vergleich FP8 (Baseline):"
echo "  Short:  37.2 tok/s"
echo "  Medium: 40.7 tok/s"
echo "  Long:   40.3 tok/s"
