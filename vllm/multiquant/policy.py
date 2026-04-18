# SPDX-License-Identifier: Apache-2.0
"""MultiQuant Policy Registry — per-class quantization configuration.

Defines what quantization scheme is used for each component:
- K-Cache, V-Cache (separate or joint)
- Weights: shared experts, routed experts, attention, dense MLP
- MTP, DeltaNet

Built from CLI args + model quantization config at startup.
Logged as banner showing current state, target, and model structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class QuantPolicy:
    """Quantization policy for a single component class."""
    dtype: str = "bf16"       # "bf16", "fp8", "fp16", "int2", "int3", "int4",
                               # "tq2w", "tq3w", "tq4w", "tq3r", etc.
    bits: int = 16             # effective bit width (16=bf16, 8=fp8, 4=int4, etc.)
    source: str = "default"    # "cli", "model", "default"
    group_size: int = 0        # 0 = use model default or N/A
    description: str = ""      # human-readable, for startup log

    @property
    def is_quantized(self) -> bool:
        return self.bits < 16

    @property
    def is_multiquant(self) -> bool:
        """True if this dtype is handled by MultiQuant (tqXw, rqX, etc.)."""
        from vllm.multiquant.registry import is_multiquant_dtype
        return is_multiquant_dtype(self.dtype)


@dataclass
class ComponentInfo:
    """What model introspection found for one component class."""
    exists: bool = False
    count: int = 0
    current_dtype: str = "n/a"
    current_bits: int = 0
    detail: str = ""


# Component class names (keys for the registry)
K_CACHE = "k_cache"
V_CACHE = "v_cache"
WEIGHTS_SHARED = "weights_shared"
WEIGHTS_ROUTED = "weights_routed"
WEIGHTS_ATTN = "weights_attn"
WEIGHTS_DENSE = "weights_dense"
WEIGHTS_ALL = "weights"  # shorthand for all weight classes
LM_HEAD = "lm_head"
MTP = "mtp"
DELTANET = "deltanet"

# All component classes
ALL_CLASSES = [
    K_CACHE, V_CACHE,
    WEIGHTS_SHARED, WEIGHTS_ROUTED, WEIGHTS_ATTN, WEIGHTS_DENSE,
    LM_HEAD, MTP, DELTANET,
]

# Map dtype strings to bit widths
DTYPE_BITS = {
    "bf16": 16, "fp16": 16, "fp32": 32,
    "fp8": 8, "fp8_e4m3": 8, "fp8_e5m2": 8,
    "int8": 8, "int4": 4, "int3": 3, "int2": 2,
    "tq2w": 2, "tq3w": 3, "tq4w": 4,
    "tq3r": 3, "tq4r": 4,
    "tq3": 3, "tq4": 4,
    "rq2": 2, "rq3": 3, "rq4": 4,
    # XFP family (learned per-channel codebook, XFP.PAPER.md)
    "xfp": 0,  # auto-select bit width per layer via MSE/cos analysis
    "xfp2": 2, "xfp3": 3, "xfp4": 4, "xfp5": 5, "xfp6": 6,
}


def _bits_for_dtype(dtype: str) -> int:
    """Get effective bit width for a dtype string."""
    return DTYPE_BITS.get(dtype, 16)


def classify_layer(prefix: str) -> str:
    """Classify a layer by its parameter name prefix.

    Returns one of: 'routed_expert', 'shared_expert', 'attn', 'mtp',
    'dense_mlp', 'deltanet', or 'other'.
    """
    p = prefix.lower()
    # Vision / ViT encoders in multimodal models — skip quantization.
    # XFP kernels are tuned for LLM shapes; ViT shapes (small patch count,
    # different head layout) trigger CUDA invalid-argument errors.
    if "visual." in p or "vision." in p or "vit." in p or p.startswith("visual") \
            or p.startswith("vision") or p.startswith("vit"):
        return "other"
    # LM Head (output projection to vocab — typically the largest single layer)
    if "lm_head" in p or (("output" in p or "head" in p) and "embed" not in p
                          and "attn" not in p and "expert" not in p):
        return "lm_head"
    # MTP layers
    if "mtp_" in p or "multi_token_prediction" in p or ".mtp." in p:
        return "mtp"
    # DeltaNet
    if "deltanet" in p:
        return "deltanet"
    # Attention
    if "self_attn" in p or "attention" in p:
        return "attn"
    # Shared experts (check before routed — "shared" substring match)
    if "shared_expert" in p:
        return "shared_expert"
    # Routed experts (check BEFORE dense_mlp — ".mlp.experts" contains ".mlp.")
    if ".experts" in p or "routed_expert" in p:
        return "routed_expert"
    # Dense MLP (layer 0 in MoE models, or generic MLP)
    if ".mlp." in p:
        return "dense_mlp"
    return "other"


def _desc_for_dtype(dtype: str) -> str:
    """Human-readable description for startup log."""
    descs = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp8": "FP8 E4M3",
        "int4": "INT4 (AutoRound/GPTQ)",
        "int3": "INT3 (AutoRound/GPTQ, 3-in-4)",
        "int2": "INT2 (AutoRound/GPTQ)",
        "tq2w": "WHT 2-bit (Lloyd-Max, block-32)",
        "tq3w": "WHT 3-bit (Lloyd-Max, block-32)",
        "tq4w": "WHT 4-bit (Lloyd-Max, block-32)",
        "tq3r": "Block-rot 3-bit (random orthogonal)",
        "xfp": "XFP auto (MSE-selected codebook size per layer)",
        "xfp2": "XFP2 (learned codebook, 4 entries)",
        "xfp3": "XFP3 (learned codebook, 8 entries)",
        "xfp4": "XFP4 (learned codebook, 16 entries)",
        "xfp5": "XFP5 (learned codebook, 32 entries) [v3]",
        "xfp6": "XFP6 (learned codebook, 64 entries) [v3]",
        # no "auto" — always explicit
    }
    return descs.get(dtype, dtype)


class MultiQuantPolicyRegistry:
    """Registry of quantization policies per component class.

    Built at startup from CLI args + model config.
    Immutable after construction.

    Singleton: the active instance is accessible via ``get_active()``.
    """

    _active: Optional["MultiQuantPolicyRegistry"] = None

    @classmethod
    def get_active(cls) -> Optional["MultiQuantPolicyRegistry"]:
        """Return the active policy registry (set by from_cli)."""
        return cls._active

    def __init__(self):
        # Default: everything bf16
        self._policies: dict[str, QuantPolicy] = {
            K_CACHE: QuantPolicy("bf16", 16, "default", 0, "bfloat16"),
            V_CACHE: QuantPolicy("bf16", 16, "default", 0, "bfloat16"),
            WEIGHTS_SHARED: QuantPolicy("bf16", 16, "default", 0,
                                        "bfloat16"),
            WEIGHTS_ROUTED: QuantPolicy("bf16", 16, "default", 0,
                                        "bfloat16"),
            WEIGHTS_ATTN: QuantPolicy("bf16", 16, "default", 0,
                                      "bfloat16"),
            WEIGHTS_DENSE: QuantPolicy("bf16", 16, "default", 0,
                                       "bfloat16"),
            LM_HEAD: QuantPolicy("bf16", 16, "default", 0, "bfloat16"),
            MTP: QuantPolicy("bf16", 16, "default", 0, "bfloat16"),
            DELTANET: QuantPolicy("bf16", 16, "default", 0, "bfloat16"),
        }
        self._components: dict[str, ComponentInfo] = {}

    def set(self, cls: str, dtype: str, source: str = "cli",
            group_size: int = 0) -> None:
        """Set policy for a component class."""
        self._policies[cls] = QuantPolicy(
            dtype=dtype,
            bits=_bits_for_dtype(dtype),
            source=source,
            group_size=group_size,
            description=_desc_for_dtype(dtype),
        )

    def get(self, cls: str) -> QuantPolicy:
        """Get policy for a component class."""
        return self._policies.get(cls, QuantPolicy())

    def to_dict(self) -> dict:
        """Stable, deterministic serialization of the weight-quant state.

        Used as input to the XFP weight-cache hash. Only fields that change
        the packed/codebook bytes are included — `description`/`source` are
        pure metadata. KV-cache entries are excluded because they only
        affect runtime inference, not the saved weights.

        Keys are sorted component names; inner dicts are sorted by key.
        Safe to feed to ``json.dumps(..., sort_keys=True)``.
        """
        weight_classes = (
            WEIGHTS_SHARED, WEIGHTS_ROUTED, WEIGHTS_ATTN,
            WEIGHTS_DENSE, LM_HEAD, MTP, DELTANET,
        )
        out: dict[str, dict] = {}
        for cls_name in sorted(weight_classes):
            p = self._policies.get(cls_name)
            if p is None:
                continue
            out[cls_name] = {
                "dtype": p.dtype,
                "bits": p.bits,
                "group_size": p.group_size,
            }
        return out

    @property
    def k_cache(self) -> QuantPolicy:
        return self._policies[K_CACHE]

    @property
    def v_cache(self) -> QuantPolicy:
        return self._policies[V_CACHE]

    @property
    def kv_cache_dtype(self) -> str:
        """Legacy: single KV dtype (K and V same). Returns K dtype."""
        return self._policies[K_CACHE].dtype

    def get_weight_policy(self, layer_type: str) -> QuantPolicy:
        """Get weight policy for a specific layer type.

        layer_type: 'shared_expert', 'routed_expert', 'attn', 'dense_mlp',
                    'mtp', 'deltanet', or generic
        """
        mapping = {
            "shared_expert": WEIGHTS_SHARED,
            "routed_expert": WEIGHTS_ROUTED,
            "attn": WEIGHTS_ATTN,
            "dense_mlp": WEIGHTS_DENSE,
            "lm_head": LM_HEAD,
            "mtp": MTP,
            "deltanet": DELTANET,
        }
        cls = mapping.get(layer_type, WEIGHTS_ROUTED)
        return self._policies[cls]

    @staticmethod
    def analyze_model(
        hf_config: object,
        model_quant_config: Optional[dict] = None,
    ) -> dict[str, ComponentInfo]:
        """Introspect HF model config to discover component classes.

        Returns a dict of component class → ComponentInfo describing
        what the model contains and its current quantization state.
        """
        components: dict[str, ComponentInfo] = {}

        num_layers = getattr(hf_config, "num_hidden_layers", 0)

        # Current dtype from quantization config
        if model_quant_config:
            qm = model_quant_config.get("quant_method", "")
            qb = model_quant_config.get("bits", 16)
            extra = model_quant_config.get("extra_config", {})
            if qm in ("gptq", "awq", "auto-round"):
                cur_dtype = f"int{qb}"
                cur_bits = qb
            elif qm in ("fp8", "compressed-tensors"):
                cur_dtype, cur_bits = "fp8", 8
            else:
                cur_dtype, cur_bits = "bf16", 16
        else:
            cur_dtype, cur_bits = "bf16", 16
            extra = {}

        # Shared experts BF16?
        has_bf16_shared = any(
            "shared_expert" in k and v.get("bits", 0) >= 16
            for k, v in extra.items()
        ) if extra else False

        # Dense MLP layer 0 BF16?
        has_bf16_dense = any(
            "layers.0.mlp" in k and "expert" not in k and v.get("bits", 0) >= 16
            for k, v in extra.items()
        ) if extra else False

        # --- Routed Experts ---
        n_experts = (getattr(hf_config, "n_routed_experts", 0)
                     or getattr(hf_config, "num_local_experts", 0)
                     or getattr(hf_config, "num_experts", 0))
        first_dense = getattr(hf_config, "first_k_dense_replace", 1)
        moe_freq = getattr(hf_config, "moe_layer_freq", 1)
        if n_experts > 0 and num_layers > 0:
            n_moe_layers = sum(
                1 for i in range(num_layers)
                if i >= first_dense and (i - first_dense) % moe_freq == 0
            )
            total_experts = n_experts * n_moe_layers
            components[WEIGHTS_ROUTED] = ComponentInfo(
                exists=True, count=total_experts,
                current_dtype=cur_dtype, current_bits=cur_bits,
                detail=f"{n_experts} experts x {n_moe_layers} layers",
            )
        else:
            components[WEIGHTS_ROUTED] = ComponentInfo(exists=False)

        # --- Shared Experts ---
        n_shared = (getattr(hf_config, "n_shared_experts", 0)
                    or getattr(hf_config, "moe_num_shared_experts", 0))
        if n_shared > 0 and num_layers > 0:
            n_shared_layers = sum(
                1 for i in range(num_layers)
                if i >= first_dense and (i - first_dense) % moe_freq == 0
            )
            s_dtype = "bf16" if has_bf16_shared else cur_dtype
            s_bits = 16 if has_bf16_shared else cur_bits
            components[WEIGHTS_SHARED] = ComponentInfo(
                exists=True, count=n_shared_layers,
                current_dtype=s_dtype, current_bits=s_bits,
                detail=f"{n_shared_layers} layers"
                       + (" (1:1)" if s_bits >= 16 else ""),
            )
        else:
            components[WEIGHTS_SHARED] = ComponentInfo(exists=False)

        # --- Attention ---
        if num_layers > 0:
            components[WEIGHTS_ATTN] = ComponentInfo(
                exists=True, count=num_layers,
                current_dtype=cur_dtype, current_bits=cur_bits,
                detail=f"{num_layers} layers",
            )
        else:
            components[WEIGHTS_ATTN] = ComponentInfo(exists=False)

        # --- Dense MLP (pre-MoE layers) ---
        if n_experts > 0 and first_dense > 0:
            d_dtype = "bf16" if has_bf16_dense else cur_dtype
            d_bits = 16 if has_bf16_dense else cur_bits
            layers_str = (f"layer 0" if first_dense == 1
                          else f"layers 0-{first_dense - 1}")
            components[WEIGHTS_DENSE] = ComponentInfo(
                exists=True, count=first_dense,
                current_dtype=d_dtype, current_bits=d_bits,
                detail=layers_str + (" (1:1)" if d_bits >= 16 else ""),
            )
        else:
            components[WEIGHTS_DENSE] = ComponentInfo(exists=False)

        # --- MTP ---
        n_mtp = getattr(hf_config, "num_nextn_predict_layers", 0)
        if n_mtp > 0:
            components[MTP] = ComponentInfo(
                exists=True, count=n_mtp,
                current_dtype="bf16", current_bits=16,
                detail=f"{n_mtp} predict layer{'s' if n_mtp > 1 else ''}",
            )
        else:
            components[MTP] = ComponentInfo(exists=False)

        # --- KV-Cache (always exists if model has layers) ---
        components[K_CACHE] = ComponentInfo(
            exists=num_layers > 0, count=num_layers,
            current_dtype="bf16", current_bits=16, detail="",
        )
        components[V_CACHE] = ComponentInfo(
            exists=num_layers > 0, count=num_layers,
            current_dtype="bf16", current_bits=16, detail="",
        )

        return components

    @classmethod
    def from_cli(
        cls,
        kv_cache_dtype: Optional[str] = None,
        k_dtype: Optional[str] = None,
        v_dtype: Optional[str] = None,
        weight_dtype: Optional[str] = None,
        weight_dtype_shared: Optional[str] = None,
        weight_dtype_routed: Optional[str] = None,
        weight_dtype_attn: Optional[str] = None,
        weight_dtype_mtp: Optional[str] = None,
        weight_dtype_lm_head: Optional[str] = None,
        weight_dtype_dense: Optional[str] = None,
        model_quant_config: Optional[dict] = None,
        hf_config: Optional[object] = None,
    ) -> "MultiQuantPolicyRegistry":
        """Build registry from CLI arguments + model config."""
        reg = cls()

        # Analyze model structure
        if hf_config is not None:
            reg._components = cls.analyze_model(hf_config, model_quant_config)

        # 1. KV-Cache: --kv-cache-dtype sets both K and V
        if kv_cache_dtype is not None:
            reg.set(K_CACHE, kv_cache_dtype, "cli")
            reg.set(V_CACHE, kv_cache_dtype, "cli")

        # 2. --k-dtype / --v-dtype override individually
        if k_dtype is not None:
            reg.set(K_CACHE, k_dtype, "cli")
        if v_dtype is not None:
            reg.set(V_CACHE, v_dtype, "cli")

        # 3. Weights from model config (if available)
        if model_quant_config:
            model_method = model_quant_config.get("quant_method", "")
            model_bits = model_quant_config.get("bits", 16)
            model_gs = model_quant_config.get("group_size", 0)
            extra_config = model_quant_config.get("extra_config", {})

            if model_method in ("gptq", "awq", "auto-round"):
                w_dtype = f"int{model_bits}"
                for w_cls in [WEIGHTS_ROUTED, WEIGHTS_ATTN]:
                    reg.set(w_cls, w_dtype, "model", model_gs)
                has_bf16_shared = any(
                    "shared_expert" in k and v.get("bits", 0) >= 16
                    for k, v in extra_config.items()
                )
                if has_bf16_shared:
                    reg.set(WEIGHTS_SHARED, "bf16", "model")
                else:
                    reg.set(WEIGHTS_SHARED, w_dtype, "model", model_gs)
                # Dense MLP (layer 0)
                has_bf16_dense = any(
                    "layers.0.mlp" in k and "expert" not in k
                    and v.get("bits", 0) >= 16
                    for k, v in extra_config.items()
                )
                if has_bf16_dense:
                    reg.set(WEIGHTS_DENSE, "bf16", "model")
                else:
                    reg.set(WEIGHTS_DENSE, w_dtype, "model", model_gs)
            elif model_method in ("fp8", "compressed-tensors"):
                for w_cls in [WEIGHTS_SHARED, WEIGHTS_ROUTED, WEIGHTS_ATTN,
                              WEIGHTS_DENSE]:
                    reg.set(w_cls, "fp8", "model")

        # 4. --weight-dtype overrides ALL weight classes
        if weight_dtype is not None:
            for w_cls in [WEIGHTS_SHARED, WEIGHTS_ROUTED, WEIGHTS_ATTN,
                          WEIGHTS_DENSE, LM_HEAD, MTP]:
                reg.set(w_cls, weight_dtype, "cli")

        # 5. Per-class weight overrides (take precedence over global)
        if weight_dtype_shared is not None:
            reg.set(WEIGHTS_SHARED, weight_dtype_shared, "cli")
        if weight_dtype_routed is not None:
            reg.set(WEIGHTS_ROUTED, weight_dtype_routed, "cli")
        if weight_dtype_attn is not None:
            reg.set(WEIGHTS_ATTN, weight_dtype_attn, "cli")
        if weight_dtype_mtp is not None:
            reg.set(MTP, weight_dtype_mtp, "cli")
        if weight_dtype_lm_head is not None:
            reg.set(LM_HEAD, weight_dtype_lm_head, "cli")
        if weight_dtype_dense is not None:
            reg.set(WEIGHTS_DENSE, weight_dtype_dense, "cli")

        cls._active = reg
        return reg

    def log_policy(self) -> None:
        """Log the full policy as startup banner with current/target/detail."""
        display = [
            ("K-Cache", K_CACHE),
            ("V-Cache", V_CACHE),
            ("Routed Experts", WEIGHTS_ROUTED),
            ("Shared Experts", WEIGHTS_SHARED),
            ("Attention", WEIGHTS_ATTN),
            ("Dense MLP", WEIGHTS_DENSE),
            ("LM Head", LM_HEAD),
            ("MTP", MTP),
        ]
        has_components = bool(self._components)

        logger.info("MultiQuant Registry:")
        if has_components:
            hdr = ("  %-20s %-10s %-10s %s" %
                   ("Component", "Current", "Target", "Detail"))
            logger.info(hdr)
            logger.info("  " + "-" * 68)

        for label, cls in display:
            p = self._policies[cls]
            c = self._components.get(cls)
            if c is not None and not c.exists:
                continue  # skip components not in model

            target = p.dtype

            if has_components and c is not None:
                current = c.current_dtype
                # Action annotation
                t_bits = _bits_for_dtype(target)
                c_bits = c.current_bits
                if t_bits < c_bits:
                    action = "RTN"
                elif target != current and p.source == "cli":
                    action = "convert"
                else:
                    action = ""
                detail = c.detail
                if action:
                    detail = f"-> {action}" + (f" {detail}" if detail else "")
                if p.source != "default":
                    detail += f" [{p.source}]"
                logger.info("  %-20s %-10s %-10s %s",
                            label, current, target, detail)
            else:
                src = f" [{p.source}]" if p.source != "default" else ""
                gs = f", gs={p.group_size}" if p.group_size > 0 else ""
                logger.info("  %-20s %-10s (%s%s)%s",
                            label, p.dtype, p.description, gs, src)

    def needs_onthefly_quant(self, layer_type: str,
                             model_dtype: str) -> bool:
        """Check if on-the-fly quantization is needed for a layer.

        True when CLI requests a quantized dtype but the model layer
        is in a higher-precision format (BF16/FP16/FP8).
        """
        policy = self.get_weight_policy(layer_type)
        if policy.source != "cli":
            return False  # no CLI override → load 1:1
        if policy.dtype == model_dtype:
            return False  # same format → no conversion
        # CLI requests lower precision than model has
        return _bits_for_dtype(policy.dtype) < _bits_for_dtype(model_dtype)

    # ─── Stats collection (XFP pack quality) ──────────────────────

    def _ensure_stats_storage(self) -> None:
        if not hasattr(self, "_stats"):
            self._stats: list[tuple[str, object]] = []
            self._stats_logged: bool = False

    def record_stats(self, layer_type: str, stats: object) -> None:
        """Accumulate per-layer XFPPackStats for end-of-load summary."""
        self._ensure_stats_storage()
        self._stats.append((layer_type, stats))

    def log_stats_summary(self) -> None:
        """Aggregate accumulated XFP pack stats as a per-class banner.

        Fires at most once per registry instance. Safe to call repeatedly
        (guard on `_stats_logged`).
        """
        self._ensure_stats_storage()
        if self._stats_logged or not self._stats:
            return
        self._stats_logged = True

        # Group by layer_type
        groups: dict[str, list[object]] = {}
        for lt, st in self._stats:
            groups.setdefault(lt, []).append(st)

        logger.info("")
        logger.info("XFP Summary (%d layers, %d classes):",
                    len(self._stats), len(groups))

        total_params = 0
        total_bits_weighted = 0.0
        total_outlier_weighted = 0.0

        # Display order
        display_names = {
            "attn": "Attention",
            "routed_expert": "Routed MoE",
            "shared_expert": "Shared",
            "dense_mlp": "Dense MLP",
            "lm_head": "LM Head",
            "mtp": "MTP",
            "deltanet": "DeltaNet",
            "other": "Other",
        }
        for lt in ["attn", "routed_expert", "shared_expert",
                    "dense_mlp", "lm_head", "mtp", "deltanet", "other"]:
            if lt not in groups:
                continue
            items = groups[lt]
            n = len(items)
            mean_cos = sum(s.cos_sim for s in items) / n
            mean_outlier = 100.0 * sum(
                getattr(s, 'outlier_fraction', 0) for s in items) / n

            # Bit-width distribution
            from collections import Counter
            bit_counts = Counter(s.bits for s in items)
            bit_str = ", ".join(
                f"{cnt}× xfp{b}" for b, cnt in sorted(bit_counts.items()))

            # Param count for effective bits
            for s in items:
                numel = s.shape[0] * s.shape[1]
                total_params += numel
                total_bits_weighted += numel * s.bits
                total_outlier_weighted += numel * getattr(
                    s, 'outlier_fraction', 0)

            name = display_names.get(lt, lt)
            logger.info(
                "  %-14s (%2d layers): %-28s | avg cos=%.3f | outliers=%.2f%%",
                name, n, bit_str, mean_cos, mean_outlier,
            )

        # Note classes that were targeted but have no stats (e.g. LM Head
        # is VocabParallelEmbedding, not LinearBase — not quantized by XFP)
        for lt in ["lm_head", "mtp", "deltanet"]:
            if lt not in groups:
                p = self.get_weight_policy(lt)
                if p.dtype not in ("bf16", "fp16", "fp32"):
                    name = display_names.get(lt, lt)
                    logger.info(
                        "  %-14s            : bf16 (not a Linear layer, "
                        "target %s skipped)", name, p.dtype)

        if total_params > 0:
            eff_bits = total_bits_weighted / total_params
            avg_outlier = 100.0 * total_outlier_weighted / total_params
            logger.info(
                "  Total: ~%.1f eff. bits/param, %.2f%% outliers avg",
                eff_bits, avg_outlier,
            )
        logger.info("")


# ─── Central weight-method dispatcher ──────────────────────────────

def create_weight_method(
    quant_config,
    layer,
    prefix: str,
):
    """Central dispatch: reads the active MultiQuantPolicyRegistry and
    instantiates the correct weight quantization method for this layer.

    Single source of truth for XFP, Archer TQ/RQ, AutoRound RTN, and
    pass-through. Called by the thin `get_quant_method` wrappers of
    `AutoRoundRTNConfig` (and optionally `ArcherConfig`, Phase 3b).

    Returns a `QuantizeMethodBase` instance or `None`.
    """
    from vllm.model_executor.layers.linear import (
        LinearBase, UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    try:
        from vllm.model_executor.layers.fused_moe import FusedMoE
    except ImportError:
        FusedMoE = None  # type: ignore

    reg = MultiQuantPolicyRegistry.get_active()

    # When no registry is active (shouldn't happen in production), fall
    # through to a pass-through for LinearBase so model loads at least.
    if reg is None:
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None

    layer_type = classify_layer(prefix)
    # Layers classified as "other" (e.g. ViT/vision encoders) are passed
    # through unquantized — their shapes aren't covered by our kernels.
    if layer_type == "other":
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None
    policy = reg.get_weight_policy(layer_type)
    dtype = policy.dtype

    # Debug: log FusedMoE dispatch
    if FusedMoE is not None and isinstance(layer, FusedMoE):
        logger.debug(
            "create_weight_method: FusedMoE prefix='%s' -> type='%s' dtype='%s'",
            prefix, layer_type, dtype,
        )

    # FP8 quant-on-load (bf16 → fp8_e4m3fn at load time)
    if dtype == "fp8":
        if isinstance(layer, VocabParallelEmbedding):
            from vllm.multiquant.fp8_embedding import FP8EmbeddingMethod
            return FP8EmbeddingMethod()
        if isinstance(layer, LinearBase):
            from vllm.multiquant.fp8_linear import FP8LinearMethod
            return FP8LinearMethod(quant_config)
        return None

    # Pass-through for unquantized dtypes
    if dtype in ("bf16", "fp16", "fp32"):
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None

    # XFP family (learned codebook)
    if dtype.startswith("xfp"):
        if isinstance(layer, LinearBase):
            from vllm.multiquant.xfp.online_linear import XFPLinearMethod
            return XFPLinearMethod(quant_config, dtype=dtype)
        if FusedMoE is not None and isinstance(layer, FusedMoE):
            from vllm.multiquant.xfp.online_moe import XFPMoEMethod
            return XFPMoEMethod(
                quant_config,
                dtype=dtype,
                moe_config=getattr(layer, "moe_config", None),
            )
        return None

    # Archer TurboQuant / RotorQuant (rotated + QJL)
    if dtype.startswith("tq") or dtype.startswith("rq"):
        if isinstance(layer, LinearBase):
            from vllm.multiquant.weight_quant.online_linear import (
                ArcherOnlineLinearMethod,
            )
            return ArcherOnlineLinearMethod(quant_config)
        return None

    # AutoRound linear RTN (INT2/INT3/INT4)
    if dtype.startswith("int"):
        if isinstance(layer, LinearBase):
            from vllm.multiquant.autoround.online_linear import (
                AutoRoundRTNLinearMethod,
            )
            return AutoRoundRTNLinearMethod(quant_config, dtype=dtype)
        if FusedMoE is not None and isinstance(layer, FusedMoE):
            # MoE RTN accepts dtypes int2/int3; int4 falls through to BF16.
            if dtype in ("int2", "int3"):
                from vllm.multiquant.autoround.online_moe import (
                    AutoRoundRTNMoEMethod,
                )
                bits = DTYPE_BITS.get(dtype, 4)
                return AutoRoundRTNMoEMethod(
                    quant_config,
                    bits=bits,
                    group_size=getattr(quant_config, "group_size", 128),
                    moe_config=getattr(layer, "moe_config", None),
                )
            return None
        return None

    # Unknown dtype — don't silently pass-through, surface as error upstream
    logger.warning(
        "create_weight_method: unknown dtype '%s' for layer %s (%s), "
        "returning None — check DTYPE_BITS and classify_layer",
        dtype, prefix, type(layer).__name__,
    )
    return None
