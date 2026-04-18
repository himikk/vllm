#!/usr/bin/env python3
"""Patch container vLLM for TurboQuant transparent wrapper approach.
TQ injects into the normal attention pipeline (FlashInfer/FlashAttn).
"""
import re

V = "/usr/local/lib/python3.12/dist-packages/vllm"


def patch(path, old, new, label=""):
    with open(path) as f:
        c = f.read()
    if new in c:
        print(f"  [skip] {label or path}")
        return
    if old not in c:
        print(f"  [WARN] {label or path} — pattern not found")
        return
    c = c.replace(old, new)
    with open(path, "w") as f:
        f.write(c)
    print(f"  [ok] {label or path}")


print("Patching for TurboQuant transparent wrapper...")

# 1. _layer_idx fix (RIY bug)
F = f"{V}/model_executor/layers/fused_moe/layer.py"
with open(F) as f:
    c = f.read()
if "_layer_idx = int" not in c:
    c = c.replace(
        "import vllm.envs as envs",
        "import os as _os\nimport vllm.envs as envs",
    )
    # Insert _layer_idx extraction BEFORE the line that uses it
    # The line has 8 spaces indent (inside __init__ method body)
    c = c.replace(
        "        self._prune_logit_mask = None  # set below after _layer_idx is known",
        "        import re as _re_moe\n"
        '        _m = _re_moe.search(r"layers\\.(\\d+)", prefix)\n'
        "        _layer_idx = int(_m.group(1)) if _m else -1\n\n"
        "        self._prune_logit_mask = None  # set below after _layer_idx is known",
    )
    with open(F, "w") as f:
        f.write(c)
    print("  [ok] layer.py (_layer_idx)")
else:
    print("  [skip] layer.py")

# 2. CacheDType tq3/tq4
patch(
    f"{V}/config/cache.py",
    '    "fp8_ds_mla",\n]',
    '    "fp8_ds_mla",\n    "tq3",\n    "tq4",\n]',
    "cache.py (CacheDType)",
)

# 3. STR_DTYPE → float16 for TQ
patch(
    f"{V}/utils/torch_utils.py",
    '    "fp8_ds_mla": torch.uint8,\n}',
    '    "fp8_ds_mla": torch.uint8,\n    "tq3": torch.float16,\n    "tq4": torch.float16,\n}',
    "torch_utils.py (dtype map)",
)

# 4. kv_cache_dtype_str_to_dtype: TQ → model dtype
patch(
    f"{V}/utils/torch_utils.py",
    '    if kv_cache_dtype == "auto":',
    '    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("tq"):\n'
    "        return model_config.dtype if model_config else torch.half\n"
    '    if kv_cache_dtype == "auto":',
    "torch_utils.py (kv_cache_dtype_str_to_dtype)",
)

# 5. attention.py: TQ init + masquerade as auto + round-trip hook
F = f"{V}/model_executor/layers/attention/attention.py"
with open(F) as f:
    c = f.read()

if "_tq_enabled" not in c:
    # TQ init before kv_cache_torch_dtype
    c = c.replace(
        "        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(",
        "        # TurboQuant: init buffers, then masquerade as auto\n"
        '        self._tq_enabled = kv_cache_dtype.startswith("tq")\n'
        "        if self._tq_enabled:\n"
        "            self._tq_original_dtype = kv_cache_dtype\n"
        "            self._init_turboquant_buffers(kv_cache_dtype, head_size, prefix)\n"
        '            kv_cache_dtype = "auto"\n\n'
        "        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(",
    )

    # Add _init_turboquant_buffers method before forward()
    method = """
    def _init_turboquant_buffers(self, cache_dtype, head_size, prefix):
        import re as _re
        from vllm.turboquant.config import TurboQuantConfig
        from vllm.turboquant.quantizer import (
            generate_rotation_matrix, generate_qjl_matrix,
        )
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

"""
    c = c.replace(
        "    def forward(\n        self,\n        query: torch.Tensor,",
        method + "    def forward(\n        self,\n        query: torch.Tensor,",
    )

    # TQ round-trip hook in unified_kv_cache_update
    c = c.replace(
        "    if layer_slot_mapping is not None:\n"
        '        assert hasattr(attn_layer.impl, "do_kv_cache_update")',
        "    if layer_slot_mapping is not None:\n"
        "        # TurboQuant round-trip on keys\n"
        "        if getattr(attn_layer, '_tq_enabled', False):\n"
        "            from vllm.turboquant.quantizer import tq_round_trip_keys\n"
        "            key = tq_round_trip_keys(key, attn_layer)\n\n"
        '        assert hasattr(attn_layer.impl, "do_kv_cache_update")',
    )

    with open(F, "w") as f:
        f.write(c)
    print("  [ok] attention.py (TQ init + hook)")
else:
    print("  [skip] attention.py")

print("\nDone!")
