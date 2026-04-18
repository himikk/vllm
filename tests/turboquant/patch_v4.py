#!/usr/bin/env python3
"""Patch vLLM container for TurboQuant v4 (compressed KV-cache)."""
import re
import torch

V = "/usr/local/lib/python3.12/dist-packages/vllm"

def p(path, old, new, label=""):
    with open(path) as f: c = f.read()
    if new in c: print(f"  [skip] {label}"); return c
    if old not in c: print(f"  [WARN] {label} — not found"); return c
    c = c.replace(old, new)
    with open(path, "w") as f: f.write(c)
    print(f"  [ok] {label}")
    return c

print("Patching for TurboQuant v4...")

# 1. _layer_idx fix
F = f"{V}/model_executor/layers/fused_moe/layer.py"
with open(F) as f: c = f.read()
if "_layer_idx = int" not in c:
    c = c.replace("import vllm.envs as envs", "import os as _os\nimport vllm.envs as envs")
    c = c.replace(
        "        self._prune_logit_mask = None  # set below after _layer_idx is known",
        "        import re as _re_moe\n"
        '        _m = _re_moe.search(r"layers\\.(\\d+)", prefix)\n'
        "        _layer_idx = int(_m.group(1)) if _m else -1\n\n"
        "        self._prune_logit_mask = None  # set below after _layer_idx is known")
    with open(F, "w") as f: f.write(c)
    print("  [ok] layer.py")

# 2. CacheDType
p(f"{V}/config/cache.py",
  '    "fp8_ds_mla",\n]',
  '    "fp8_ds_mla",\n    "tq3",\n    "tq4",\n]', "cache.py")

# 3. dtype map → uint8
p(f"{V}/utils/torch_utils.py",
  '    "fp8_ds_mla": torch.uint8,\n}',
  '    "fp8_ds_mla": torch.uint8,\n    "tq3": torch.uint8,\n    "tq4": torch.uint8,\n}',
  "torch_utils.py")

# 4. is_quantized
p(f"{V}/v1/attention/backend.py",
  'return kv_cache_dtype.startswith("fp8")',
  'return kv_cache_dtype.startswith("fp8") or kv_cache_dtype.startswith("tq")',
  "backend.py")

# 5. Registry
p(f"{V}/v1/attention/backends/registry.py",
  "    # Placeholder for third-party",
  '    TURBOQUANT = (\n'
  '        "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"\n'
  '    )\n'
  "    # Placeholder for third-party",
  "registry.py")

# 6. cuda.py — TQ priority + kv_cache_dtype param
F = f"{V}/platforms/cuda.py"
with open(F) as f: c = f.read()
if 'startswith("tq")' not in c:
    c = c.replace("    if use_mla:",
        '    if kv_cache_dtype is not None and kv_cache_dtype.startswith("tq"):\n'
        "        return [AttentionBackendEnum.TURBOQUANT]\n\n"
        "    if use_mla:")
    # Add kv_cache_dtype parameter
    c = c.replace(
        "    num_heads: int | None = None,\n) -> list[AttentionBackendEnum]:",
        "    num_heads: int | None = None,\n"
        "    kv_cache_dtype: str | None = None,\n"
        ") -> list[AttentionBackendEnum]:")
    # Pass it from caller
    c = c.replace(
        "            num_heads,\n        )\n        for priority",
        "            num_heads,\n            attn_selector_config.kv_cache_dtype,\n"
        "        )\n        for priority")
    with open(F, "w") as f: f.write(c)
    print("  [ok] cuda.py")
else:
    print("  [skip] cuda.py")

# 7. attention.py — TQ init + get_kv_cache_spec
F = f"{V}/model_executor/layers/attention/attention.py"
with open(F) as f: c = f.read()
if "_tq_enabled" not in c:
    # TQ init before kv_cache_torch_dtype
    c = c.replace(
        "        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(",
        '        import os as _tq_os\n'
        '        self._tq_enabled = (kv_cache_dtype.startswith("tq") or\n'
        '            _tq_os.environ.get("VLLM_TQ_ROUNDTRIP", "0") == "1")\n'
        '        if self._tq_enabled:\n'
        '            tq_bits = kv_cache_dtype if kv_cache_dtype.startswith("tq") else "tq3"\n'
        '            self._init_turboquant_buffers(tq_bits, head_size, prefix)\n'
        '            self._kv_storage_dtype = kv_cache_dtype\n'
        '            if kv_cache_dtype.startswith("tq"):\n'
        '                kv_cache_dtype = "fp8"\n'
        '                if cache_config is not None:\n'
        '                    cache_config.cache_dtype = "fp8"\n\n'
        "        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(")

    # _init_turboquant_buffers method
    method = '''
    def _init_turboquant_buffers(self, cache_dtype, head_size, prefix):
        import re as _re
        from vllm.turboquant.config import TurboQuantConfig
        from vllm.turboquant.quantizer import generate_rotation_matrix, generate_qjl_matrix
        from vllm.turboquant.centroids import get_centroids
        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)
        match = _re.search(r"layers\\.(\\d+)", prefix)
        layer_idx = int(match.group(1)) if match else 0
        seed = tq_config.seed + layer_idx * 1337
        self.register_buffer("_tq_Pi",
            generate_rotation_matrix(head_size, seed=seed))
        self.register_buffer("_tq_S",
            generate_qjl_matrix(head_size, seed=seed + 1))
        self.register_buffer("_tq_centroids",
            get_centroids(head_size, tq_config.mse_bits))
        self._tq_config = tq_config

'''
    c = c.replace(
        "    def forward(\n        self,\n        query: torch.Tensor,",
        method + "    def forward(\n        self,\n        query: torch.Tensor,")

    # No get_kv_cache_spec override — uses standard FP8 spec
    # (tq3 maps to fp8 in __init__, so get_kv_cache_spec returns fp8 spec)

    with open(F, "w") as f: f.write(c)
    print("  [ok] attention.py")
else:
    print("  [skip] attention.py")

print("\nDone!")
