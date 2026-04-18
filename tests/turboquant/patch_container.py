#!/usr/bin/env python3
"""Patch a vLLM container installation to add TurboQuant support.

Run inside the container:
  python3 /opt/patch_container.py

Patches:
  1. Adds tq3/tq4 to CacheDType
  2. Adds tq3/tq4 to STR_DTYPE_TO_TORCH_DTYPE
  3. Extends is_quantized_kv_cache() for tq*
  4. Adds TURBOQUANT to AttentionBackendEnum
  5. Adds TQ backend priority in cuda platform
"""

import os
import re

VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"


def patch_file(path, old, new):
    with open(path) as f:
        content = f.read()
    if new in content:
        print(f"  [skip] {os.path.basename(path)} — already patched")
        return
    if old not in content:
        print(f"  [WARN] {os.path.basename(path)} — pattern not found: {old[:60]}...")
        return
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [ok] {os.path.basename(path)}")


def main():
    print("Patching vLLM for TurboQuant...")

    # 1. CacheDType
    patch_file(
        f"{VLLM}/config/cache.py",
        '    "fp8_ds_mla",\n]',
        '    "fp8_ds_mla",\n    "tq3",\n    "tq4",\n]',
    )

    # 2. STR_DTYPE_TO_TORCH_DTYPE
    patch_file(
        f"{VLLM}/utils/torch_utils.py",
        '    "fp8_ds_mla": torch.uint8,\n}',
        '    "fp8_ds_mla": torch.uint8,\n    "tq3": torch.uint8,\n    "tq4": torch.uint8,\n}',
    )

    # 3. is_quantized_kv_cache
    patch_file(
        f"{VLLM}/v1/attention/backend.py",
        'return kv_cache_dtype.startswith("fp8")',
        'return kv_cache_dtype.startswith("fp8") or kv_cache_dtype.startswith("tq")',
    )

    # 4. AttentionBackendEnum — add TURBOQUANT before CUSTOM
    patch_file(
        f"{VLLM}/v1/attention/backends/registry.py",
        '    # Placeholder for third-party',
        '    TURBOQUANT = (\n'
        '        "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"\n'
        '    )\n'
        '    # Placeholder for third-party',
    )

    # 5. cuda.py — TQ backend priority
    # Find the _get_backend_priorities function and add TQ check
    cuda_path = f"{VLLM}/platforms/cuda.py"
    with open(cuda_path) as f:
        content = f.read()
    if "tq" not in content:
        content = content.replace(
            "    if use_mla:",
            "    # TurboQuant: exclusive backend for tq3/tq4 cache dtypes\n"
            "    if kv_cache_dtype is not None and kv_cache_dtype.startswith('tq'):\n"
            "        return [AttentionBackendEnum.TURBOQUANT]\n\n"
            "    if use_mla:",
        )
        with open(cuda_path, "w") as f:
            f.write(content)
        print(f"  [ok] cuda.py")
    else:
        print(f"  [skip] cuda.py — already patched")

    # 6. attention.py — TQ buffer init
    attn_path = f"{VLLM}/model_executor/layers/attention/attention.py"
    with open(attn_path) as f:
        content = f.read()
    if "_init_turboquant_buffers" not in content:
        # Add TQ buffer init call after _init_kv_cache_quant
        content = content.replace(
            "        _init_kv_cache_quant(self, quant_config, prefix)\n",
            "        _init_kv_cache_quant(self, quant_config, prefix)\n\n"
            "        # Initialize TurboQuant buffers if tq cache dtype\n"
            "        if kv_cache_dtype.startswith('tq'):\n"
            "            self._init_turboquant_buffers(kv_cache_dtype, head_size, prefix)\n",
        )
        # Add the method before forward()
        content = content.replace(
            "    def forward(\n        self,\n        query: torch.Tensor,",
            '''    def _init_turboquant_buffers(self, cache_dtype, head_size, prefix):
        """Initialize TurboQuant rotation/projection matrices."""
        import re as _re
        from vllm.turboquant.config import TurboQuantConfig
        from vllm.turboquant.quantizer import generate_rotation_matrix, generate_qjl_matrix
        from vllm.turboquant.centroids import get_centroids
        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)
        match = _re.search(r"layers\\.(\\d+)", prefix)
        layer_idx = int(match.group(1)) if match else 0
        seed = tq_config.seed + layer_idx * 1337
        self.register_buffer("_tq_Pi", generate_rotation_matrix(head_size, seed=seed))
        self.register_buffer("_tq_S", generate_qjl_matrix(head_size, seed=seed + 1))
        self.register_buffer("_tq_centroids", get_centroids(head_size, tq_config.mse_bits))
        self._tq_config = tq_config

    def forward(
        self,
        query: torch.Tensor,''',
        )
        with open(attn_path, "w") as f:
            f.write(content)
        print(f"  [ok] attention.py")
    else:
        print(f"  [skip] attention.py — already patched")

    # 7. Fix _layer_idx + _os + _riy NameErrors in FusedMoE (RIY bug)
    moe_path = f"{VLLM}/model_executor/layers/fused_moe/layer.py"
    with open(moe_path) as f:
        content = f.read()
    if "# TQ-fix: extract _layer_idx" not in content:
        # Add missing imports at top
        content = content.replace(
            "import vllm.envs as envs",
            "import os as _os\nimport vllm.envs as envs",
        )
        # Add _riy import (lazy, may not exist)
        content = content.replace(
            "from vllm.logger import init_logger",
            "from vllm.logger import init_logger\n"
            "try:\n"
            "    from vllm.model_executor.layers.fused_moe import riy as _riy\n"
            "except ImportError:\n"
            "    _riy = None",
        )
        # Extract _layer_idx from prefix before it's used
        content = content.replace(
            "self._prune_logit_mask = None  # set below after _layer_idx is known",
            "# TQ-fix: extract _layer_idx from prefix (e.g. 'model.layers.5.mlp')\n"
            "        import re as _re_moe\n"
            "        _m = _re_moe.search(r'layers\\.(\\d+)', prefix)\n"
            "        _layer_idx = int(_m.group(1)) if _m else -1\n\n"
            "        self._prune_logit_mask = None  # set below after _layer_idx is known",
        )
        # Guard _riy calls
        content = content.replace(
            "if _layer_idx >= 0 and _os.environ.get(\"VLLM_RIY_MONITOR\", \"0\") == \"1\":",
            "if _layer_idx >= 0 and _riy is not None and _os.environ.get(\"VLLM_RIY_MONITOR\", \"0\") == \"1\":",
        )
        with open(moe_path, "w") as f:
            f.write(content)
        print(f"  [ok] layer.py (_layer_idx + _os + _riy fix)")
    else:
        print(f"  [skip] layer.py — already patched")

    # 8. Fix kv_cache_dtype_str_to_dtype for TQ (maps to model dtype, not uint8)
    tu_path = f"{VLLM}/utils/torch_utils.py"
    with open(tu_path) as f:
        content = f.read()
    if "def kv_cache_dtype_str_to_dtype" in content and "tq3" not in content.split("kv_cache_dtype_str_to_dtype")[1][:200]:
        content = content.replace(
            '    if kv_cache_dtype == "auto":',
            '    if kv_cache_dtype.startswith("tq"):\n'
            '        return model_config.dtype if model_config else torch.half\n'
            '    if kv_cache_dtype == "auto":',
        )
        with open(tu_path, "w") as f:
            f.write(content)
        print(f"  [ok] torch_utils.py (kv_cache_dtype_str_to_dtype TQ fix)")
    else:
        print(f"  [skip] torch_utils.py — already patched or not found")

    print("\nDone! TurboQuant patches applied.")


if __name__ == "__main__":
    main()
