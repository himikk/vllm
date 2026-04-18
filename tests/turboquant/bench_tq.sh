#!/bin/bash
# TurboQuant Benchmark: FP8 vs TQ3 on Qwen3.5 35B and 122B
#
# Usage: bash tests/turboquant/bench_tq.sh [35b|122b|all]
set -euo pipefail

VLLM_SRC=/home/flash/vllm-riy
VLLM_DST=/usr/local/lib/python3.12/dist-packages/vllm
IMAGE=localhost/vllm-ng17e-riy
BENCH=/home/flash/vllm-marlin-sm12x/bench.py
NAME=ng17e-head

MODEL_35B=/data/tensordata/Qwen3.5-35B-A3B-int4-AutoRound
MODEL_122B=/data/tensordata/Qwen3.5-122B-A10B-int4-AutoRound

TARGET="${1:-35b}"

start_container() {
  echo "--- Starting container ---"
  podman stop $NAME 2>/dev/null || true
  podman rm $NAME 2>/dev/null || true

  podman run -d --name $NAME \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    --hooks-dir=/usr/share/containers/oci/hooks.d \
    --network host --ipc host \
    -v /data/tensordata:/data/tensordata \
    -v ${VLLM_SRC}/vllm/turboquant:${VLLM_DST}/turboquant:ro \
    -v ${VLLM_SRC}/vllm/v1/attention/backends/turboquant_attn.py:${VLLM_DST}/v1/attention/backends/turboquant_attn.py:ro \
    -e VLLM_MLA_DISABLE=1 \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a" \
    $IMAGE sleep infinity
  sleep 2

  # Patch _layer_idx bug
  podman exec $NAME sed -i \
    's/^import vllm.envs as envs/import os as _os\nimport vllm.envs as envs/' \
    ${VLLM_DST}/model_executor/layers/fused_moe/layer.py

  podman exec $NAME sed -i \
    '/self._prune_logit_mask = None.*_layer_idx/i\        import re as _re_moe; _m = _re_moe.search(r"layers\\\\.(\\\\d+)", prefix); _layer_idx = int(_m.group(1)) if _m else -1' \
    ${VLLM_DST}/model_executor/layers/fused_moe/layer.py

  # Patch container for TQ
  podman exec $NAME python3 -c "
import sys; sys.path.insert(0, '${VLLM_DST}')
F = '${VLLM_DST}/config/cache.py'
with open(F) as f: c = f.read()
if 'tq3' not in c:
    c = c.replace('\"fp8_ds_mla\",', '\"fp8_ds_mla\",\n    \"tq3\",\n    \"tq4\",')
    with open(F,'w') as f: f.write(c)
    print('Patched cache.py')

F = '${VLLM_DST}/utils/torch_utils.py'
with open(F) as f: c = f.read()
if 'tq3' not in c:
    c = c.replace('\"fp8_ds_mla\": torch.uint8,', '\"fp8_ds_mla\": torch.uint8,\n    \"tq3\": torch.uint8,\n    \"tq4\": torch.uint8,')
    # Fix kv_cache_dtype_str_to_dtype for TQ
    c = c.replace(
        'if kv_cache_dtype == \"auto\":',
        'if kv_cache_dtype.startswith(\"tq\"):\n        return model_config.dtype if model_config else torch.half\n    if kv_cache_dtype == \"auto\":')
    with open(F,'w') as f: f.write(c)
    print('Patched torch_utils.py')

F = '${VLLM_DST}/v1/attention/backend.py'
with open(F) as f: c = f.read()
if 'startswith(\"tq\")' not in c:
    c = c.replace('return kv_cache_dtype.startswith(\"fp8\")', 'return kv_cache_dtype.startswith(\"fp8\") or kv_cache_dtype.startswith(\"tq\")')
    with open(F,'w') as f: f.write(c)
    print('Patched backend.py')

F = '${VLLM_DST}/v1/attention/backends/registry.py'
with open(F) as f: c = f.read()
if 'TURBOQUANT' not in c:
    c = c.replace('# Placeholder for third-party',
        'TURBOQUANT = (\n        \"vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend\"\n    )\n    # Placeholder for third-party')
    with open(F,'w') as f: f.write(c)
    print('Patched registry.py')

F = '${VLLM_DST}/platforms/cuda.py'
with open(F) as f: c = f.read()
if 'startswith(\"tq\")' not in c:
    c = c.replace('if use_mla:', 'if kv_cache_dtype is not None and kv_cache_dtype.startswith(\"tq\"):\n        return [AttentionBackendEnum.TURBOQUANT]\n\n    if use_mla:')
    with open(F,'w') as f: f.write(c)
    print('Patched cuda.py')

# Patch attention.py for TQ buffers
F = '${VLLM_DST}/model_executor/layers/attention/attention.py'
with open(F) as f: c = f.read()
if '_init_turboquant_buffers' not in c:
    c = c.replace(
        '        _init_kv_cache_quant(self, quant_config, prefix)\n',
        '        _init_kv_cache_quant(self, quant_config, prefix)\n\n'
        '        if kv_cache_dtype.startswith(\"tq\"):\n'
        '            self._init_turboquant_buffers(kv_cache_dtype, head_size, prefix)\n')
    c = c.replace(
        '    def forward(\n        self,\n        query: torch.Tensor,',
        '''    def _init_turboquant_buffers(self, cache_dtype, head_size, prefix):
        import re as _re
        from vllm.turboquant.config import TurboQuantConfig
        from vllm.turboquant.quantizer import generate_rotation_matrix, generate_qjl_matrix
        from vllm.turboquant.centroids import get_centroids
        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)
        match = _re.search(r\"layers\\\\.(\\\\d+)\", prefix)
        layer_idx = int(match.group(1)) if match else 0
        seed = tq_config.seed + layer_idx * 1337
        self.register_buffer(\"_tq_Pi\", generate_rotation_matrix(head_size, seed=seed))
        self.register_buffer(\"_tq_S\", generate_qjl_matrix(head_size, seed=seed + 1))
        self.register_buffer(\"_tq_centroids\", get_centroids(head_size, tq_config.mse_bits))
        self._tq_config = tq_config

    def forward(
        self,
        query: torch.Tensor,''')
    with open(F,'w') as f: f.write(c)
    print('Patched attention.py')
print('All patches done')
"

  # Install scipy for centroids
  podman exec $NAME pip install --break-system-packages -q scipy 2>&1 | tail -1

  # Clear pycache
  podman exec $NAME find /usr/local/lib/python3.12 -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

  echo "Container ready"
}

serve_model() {
  local model=$1
  local model_name=$2
  local kv_dtype=$3
  local extra_args="${4:-}"

  echo ""
  echo "--- Serving $model_name with --kv-cache-dtype $kv_dtype ---"

  podman exec $NAME bash -c "pkill -f 'vllm serve' 2>/dev/null" || true
  sleep 3

  podman exec -d $NAME bash -c "
    vllm serve $model \
      --served-model-name $model_name \
      --host 0.0.0.0 --port 8011 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.05 \
      --kv-cache-memory-bytes 10G \
      --kv-cache-dtype $kv_dtype \
      --trust-remote-code \
      --enforce-eager \
      --limit-mm-per-prompt '{\"image\":0,\"video\":0}' \
      $extra_args \
      2>&1 | tee /tmp/serve.log
  "

  # Wait for ready
  for i in $(seq 1 60); do
    if podman exec $NAME grep -q "Application startup complete\|Uvicorn running\|Started server" /tmp/serve.log 2>/dev/null; then
      echo "SERVER READY (${i}0s)"
      return 0
    fi
    if podman exec $NAME grep -q "RuntimeError\|NameError\|Error.*engine" /tmp/serve.log 2>/dev/null; then
      echo "FAILED:"
      podman exec $NAME grep "Error" /tmp/serve.log | tail -3
      return 1
    fi
    sleep 10
  done
  echo "TIMEOUT"
  return 1
}

run_bench() {
  local model_name=$1
  local label=$2
  local logfile=$3

  echo "--- Benchmarking: $label ---"
  python3 $BENCH --url http://localhost:8011 --model "$model_name" --label "$label" 2>&1 | tee "$logfile"
}

# === Main ===
echo "============================================"
echo "TurboQuant Benchmark — $(hostname)"
echo "Target: $TARGET"
echo "============================================"

start_container

if [ "$TARGET" = "35b" ] || [ "$TARGET" = "all" ]; then
  echo ""
  echo "======== Qwen3.5-35B ========"

  # FP8 Baseline
  if serve_model "$MODEL_35B" "qwen35-35b" "fp8"; then
    run_bench "qwen35-35b" "Qwen3.5-35B FP8 (DGX)" /tmp/bench_35b_fp8.log
  fi

  # TQ3
  if serve_model "$MODEL_35B" "qwen35-35b-tq3" "tq3"; then
    run_bench "qwen35-35b-tq3" "Qwen3.5-35B TQ3 (DGX)" /tmp/bench_35b_tq3.log
  fi
fi

if [ "$TARGET" = "122b" ] || [ "$TARGET" = "all" ]; then
  echo ""
  echo "======== Qwen3.5-122B ========"

  # FP8 Baseline
  if serve_model "$MODEL_122B" "qwen35-122b" "fp8"; then
    run_bench "qwen35-122b" "Qwen3.5-122B FP8 (DGX)" /tmp/bench_122b_fp8.log
  fi

  # TQ3
  if serve_model "$MODEL_122B" "qwen35-122b-tq3" "tq3"; then
    run_bench "qwen35-122b-tq3" "Qwen3.5-122B TQ3 (DGX)" /tmp/bench_122b_tq3.log
  fi
fi

echo ""
echo "============================================"
echo "All benchmarks complete"
echo "Results in /tmp/bench_*.log"
echo "============================================"
