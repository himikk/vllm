# SPDX-License-Identifier: Apache-2.0
"""Generic disk cache for MultiQuant quantized weights.

Any quant method in the MultiQuant ecosystem (XFP, TurboQuant, RotorQuant,
…) can persist its per-layer artefacts here and skip the expensive
quant pipeline on subsequent starts. The cache is keyed by a hash that
includes the MultiQuantPolicyRegistry state — so changing *what* is
quantized *how* automatically invalidates the cache.

Design:
  - This module owns the plumbing only: paths, safetensors I/O, manifest,
    hash over policy + source code, stats, singleton.
  - Per-method serialization (which layer attributes to save and load)
    lives in the method's own module (e.g.
    ``vllm.multiquant.xfp.xfp_weight_cache``). Those modules are thin
    adapters over ``MultiQuantWeightCache.save`` / ``.load``.

On-disk layout::

    $MULTIQUANT_CACHE_DIR/<model_basename>/<hash16>/
        manifest.json
        <layer_prefix>.safetensors      (metadata['method'] == "xfp_linear", ...)
        …

Enable via env ``MULTIQUANT_CACHE_DIR`` (``XFP_CACHE_DIR`` kept as an
alias for backwards compat). Set ``MULTIQUANT_CACHE_READ_ONLY=1`` for
eval runs that must never write.

Every fallback path emits a log line — silent misses defeat the purpose.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# Constants that affect the bytes written to cache. Keep in sync with the
# defaults in the quant methods (xfp_pack etc.). Anything listed here is
# part of the cache key.
_DEFAULT_OUTLIER_SIGMA = 4.0
_DEFAULT_OUTLIER_MAX_FRACTION = 0.02
_LLOYD_ITERS_LINEAR = 20
_MANIFEST_SCHEMA_VERSION = 1


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _file_sha(path: str | Path) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        logger.warning(
            "MultiQuant cache: cannot hash %s (%s) — treating as dirty",
            path, e,
        )
        return "unreadable"


def _tensor_cpu_contig(t: torch.Tensor) -> torch.Tensor:
    if t.device.type != "cpu":
        t = t.detach().cpu()
    return t.contiguous()


class MultiQuantWeightCache:
    """Per-process disk cache. Singleton via ``get_active()``."""

    _active: Optional["MultiQuantWeightCache"] = None

    def __init__(
        self,
        cache_root: str,
        model_basename: str,
        cache_key: str,
        read_only: bool = False,
    ):
        self.cache_root = Path(cache_root)
        self.model_basename = model_basename
        self.cache_key = cache_key
        self.read_only = read_only
        self.cache_dir = self.cache_root / model_basename / cache_key
        # Stats for the end-of-load summary.
        self._hits: dict[str, int] = {}     # by method name
        self._misses: dict[str, int] = {}
        self._saves = 0
        self._save_fail = 0
        self._load_fail = 0
        self._t_loaded_s = 0.0
        self._t_saved_s = 0.0

    # ─── Singleton ───────────────────────────────────────────────────

    @classmethod
    def get_active(cls) -> Optional["MultiQuantWeightCache"]:
        return cls._active

    @classmethod
    def set_active(cls, cache: Optional["MultiQuantWeightCache"]) -> None:
        cls._active = cache

    @classmethod
    def from_env(
        cls,
        model_path: str,
        registry,
        hf_config,
        tp_size: int = 1,
        ep_size: int = 1,
    ) -> Optional["MultiQuantWeightCache"]:
        """Construct from env. Returns None when cache disabled."""
        # Primary var + XFP-era alias.
        cache_root = (
            os.environ.get("MULTIQUANT_CACHE_DIR", "").strip()
            or os.environ.get("XFP_CACHE_DIR", "").strip()
        )
        if not cache_root:
            logger.info(
                "MultiQuant cache: disabled "
                "(set MULTIQUANT_CACHE_DIR to enable)"
            )
            return None
        read_only = (
            _env_truthy("MULTIQUANT_CACHE_READ_ONLY")
            or _env_truthy("XFP_CACHE_READ_ONLY")
        )
        model_basename = os.path.basename(
            os.path.normpath(model_path)
        ) or "model"
        cache_key = cls.compute_cache_key(
            model_path, registry, hf_config, tp_size, ep_size,
        )
        inst = cls(cache_root, model_basename, cache_key, read_only)
        try:
            inst.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "MultiQuant cache: cannot mkdir %s (%s) — disabled",
                inst.cache_dir, e,
            )
            return None
        logger.info(
            "MultiQuant cache: root=%s key=%s%s (dir=%s)",
            cache_root, cache_key,
            " (read-only)" if read_only else "",
            inst.cache_dir,
        )
        return inst

    # ─── Hash / key ───────────────────────────────────────────────────

    @staticmethod
    def compute_cache_key(
        model_path: str,
        registry,
        hf_config,
        tp_size: int = 1,
        ep_size: int = 1,
    ) -> str:
        """Hash over everything that changes the quantized bytes."""
        h = hashlib.sha256()

        # Model identity
        cfg_path = os.path.join(model_path, "config.json")
        if os.path.exists(cfg_path):
            h.update(b"config_json|")
            h.update(_file_sha(cfg_path).encode())
        idx_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            h.update(b"|weights_index|")
            h.update(_file_sha(idx_path).encode())
        else:
            try:
                entries = sorted(
                    f"{f.name}:{f.stat().st_size}"
                    for f in Path(model_path).iterdir()
                    if f.is_file() and f.name.endswith(".safetensors")
                )
                h.update(b"|shard_listing|")
                h.update(json.dumps(entries).encode())
            except OSError as e:
                logger.warning(
                    "MultiQuant cache: cannot list shards in %s (%s)",
                    model_path, e,
                )

        # MultiQuant policy (WAS wird wie quantisiert)
        h.update(b"|policy|")
        try:
            pol = registry.to_dict() if registry is not None else {}
        except Exception as e:
            logger.warning(
                "MultiQuant cache: registry.to_dict failed (%s)", e
            )
            pol = {}
        h.update(json.dumps(pol, sort_keys=True).encode())

        # Quant pipeline source code — catches semantic changes to packers.
        for mod_name in (
            "vllm.multiquant.policy",
            "vllm.multiquant.xfp.xfp_pack",
        ):
            try:
                mod = __import__(mod_name, fromlist=["*"])
                path = getattr(mod, "__file__", None)
                if path:
                    h.update(f"|{mod_name}_sha|".encode())
                    h.update(_file_sha(path).encode())
            except ImportError:
                # Module absent → skip in hash; no effect on cache key
                continue

        # Quant-tuning knobs not in the registry (env-configurable).
        moe_lloyd = int(os.environ.get("XFP_MOE_LLOYD_ITERS", "5"))
        auto_min_cos = float(os.environ.get("XFP_MIN_COS", "0.98"))
        h.update(
            f"|sigma={_DEFAULT_OUTLIER_SIGMA}"
            f"|maxf={_DEFAULT_OUTLIER_MAX_FRACTION}"
            f"|lloyd_lin={_LLOYD_ITERS_LINEAR}"
            f"|moe_lloyd={moe_lloyd}"
            f"|min_cos={auto_min_cos}"
            f"|tp={tp_size}|ep={ep_size}"
            f"|schema={_MANIFEST_SCHEMA_VERSION}".encode()
        )
        return h.hexdigest()[:16]

    # ─── Path resolution ──────────────────────────────────────────────

    def layer_path(self, layer_prefix: str) -> Path:
        safe = layer_prefix.replace("/", "__").replace(":", "_")
        if not safe:
            safe = "__root__"
        return self.cache_dir / f"{safe}.safetensors"

    # ─── Generic save / load (used by per-method adapters) ───────────

    def save(
        self,
        layer_prefix: str,
        method: str,
        tensors: dict[str, torch.Tensor],
        metadata: Optional[dict[str, str]] = None,
    ) -> bool:
        """Persist a per-layer artefact bundle. Returns True on success.

        The method name (e.g. "xfp_linear", "xfp_moe", "tq_wht_linear") is
        stored in the safetensors header so loads can verify the caller's
        expected format matches the on-disk kind.
        """
        if self.read_only or not layer_prefix:
            return False
        try:
            from safetensors.torch import save_file
        except ImportError:
            logger.warning(
                "MultiQuant cache: safetensors not available — "
                "cache save disabled",
            )
            return False
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        meta: dict[str, str] = {"method": method, "cache_key": self.cache_key}
        if metadata:
            for k, v in metadata.items():
                # Safetensors metadata values MUST be strings.
                meta[str(k)] = str(v)

        contig = {k: _tensor_cpu_contig(v) for k, v in tensors.items()}

        path = self.layer_path(layer_prefix)
        tmp = path.with_suffix(".safetensors.tmp")
        t0 = time.perf_counter()
        try:
            save_file(contig, str(tmp), metadata=meta)
            tmp.replace(path)
            self._saves += 1
            self._t_saved_s += time.perf_counter() - t0
            return True
        except (OSError, RuntimeError) as e:
            self._save_fail += 1
            logger.warning(
                "MultiQuant cache: save failed for %s/%s (%s)",
                method, layer_prefix, e,
            )
            return False

    def load(
        self,
        layer_prefix: str,
        expected_method: str,
        device: torch.device,
    ) -> Optional[tuple[dict[str, torch.Tensor], dict[str, str]]]:
        """Return (tensors_on_device, metadata) or None on miss/mismatch.

        On a hit with a different ``method`` tag, returns None and logs.
        """
        if not layer_prefix:
            return None
        path = self.layer_path(layer_prefix)
        if not path.exists():
            self._misses[expected_method] = \
                self._misses.get(expected_method, 0) + 1
            return None
        try:
            from safetensors import safe_open
            t0 = time.perf_counter()
            tensors: dict[str, torch.Tensor] = {}
            with safe_open(str(path), framework="pt") as f:
                meta = f.metadata() or {}
                on_disk_method = meta.get("method")
                if on_disk_method != expected_method:
                    logger.warning(
                        "MultiQuant cache: %s method=%r != expected %r "
                        "→ treating as miss",
                        path, on_disk_method, expected_method,
                    )
                    self._load_fail += 1
                    return None
                for k in f.keys():  # noqa: SIM118
                    tensors[k] = f.get_tensor(k).to(device)
            self._hits[expected_method] = \
                self._hits.get(expected_method, 0) + 1
            self._t_loaded_s += time.perf_counter() - t0
            return tensors, dict(meta)
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            self._load_fail += 1
            logger.warning(
                "MultiQuant cache: load failed for %s/%s (%s)",
                expected_method, layer_prefix, e,
            )
            return None

    # ─── Manifest ─────────────────────────────────────────────────────

    def write_manifest(self, registry, inventory: list[str]) -> None:
        if self.read_only:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        mf = {
            "schema": _MANIFEST_SCHEMA_VERSION,
            "cache_key": self.cache_key,
            "model_basename": self.model_basename,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "policy": registry.to_dict() if registry is not None else {},
            "inventory": sorted(inventory),
            "vllm_version": _best_effort_vllm_version(),
            "env": {
                "XFP_MOE_LLOYD_ITERS":
                    os.environ.get("XFP_MOE_LLOYD_ITERS", "5"),
                "XFP_MIN_COS": os.environ.get("XFP_MIN_COS", "0.98"),
            },
        }
        path = self.cache_dir / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(mf, indent=2, sort_keys=True))
            tmp.replace(path)
        except OSError as e:
            logger.warning(
                "MultiQuant cache: manifest write failed (%s): %s",
                path, e,
            )

    def verify_manifest(self) -> bool:
        path = self.cache_dir / "manifest.json"
        if not path.exists():
            logger.info(
                "MultiQuant cache: no manifest at %s — cold start, "
                "cache will populate",
                path,
            )
            return False
        try:
            mf = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "MultiQuant cache: manifest unreadable (%s): %s → re-quant",
                path, e,
            )
            return False
        if mf.get("schema") != _MANIFEST_SCHEMA_VERSION:
            logger.warning(
                "MultiQuant cache: manifest schema=%s (want %d) → re-quant",
                mf.get("schema"), _MANIFEST_SCHEMA_VERSION,
            )
            return False
        if mf.get("cache_key") != self.cache_key:
            logger.warning(
                "MultiQuant cache: manifest cache_key=%s but "
                "computed=%s → re-quant",
                mf.get("cache_key"), self.cache_key,
            )
            return False
        return True

    # ─── Summary ──────────────────────────────────────────────────────

    def log_summary(self) -> None:
        total_hits = sum(self._hits.values())
        total_miss = sum(self._misses.values())
        hit_breakdown = ", ".join(
            f"{m}={n}" for m, n in sorted(self._hits.items())
        ) or "-"
        miss_breakdown = ", ".join(
            f"{m}={n}" for m, n in sorted(self._misses.items())
        ) or "-"
        logger.info(
            "MultiQuant cache summary: hits={%s} misses={%s} "
            "| saves=%d (%.1fs) save_fail=%d load_fail=%d "
            "| cache-io-time=%.1fs",
            hit_breakdown, miss_breakdown,
            self._saves, self._t_saved_s, self._save_fail, self._load_fail,
            self._t_loaded_s,
        )
        if total_hits > 0 and total_miss > 0:
            logger.warning(
                "MultiQuant cache: %d hits + %d misses — cache is PARTIAL. "
                "Some layers re-quantized. Check prefix stability or "
                "missing files.",
                total_hits, total_miss,
            )


def _best_effort_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"
