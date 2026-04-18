# TASK: MultiQuant — Generisches Quantisierungs-Framework

## Ziel

vLLM um ein generisches Quantisierungs-Framework erweitern, das beliebige KV-Cache- und Gewichts-Quantisierer über ein einheitliches Interface unterstützt. Erster Schritt: TurboQuant und RotorQuant unter einem Dach. Perspektive: On-the-fly Gewichtsquantisierung, 1-bit, MTP.

## Vision

```
BF16 Modell (HuggingFace)
  ↓ vllm serve --quantization multiquant --quant-bits 3 --kv-cache-dtype rq3
  ↓
┌─────────────────────────────────────────────────────┐
│ MultiQuant Framework                                │
│                                                     │
│  Gewichte: BF16 → 3-bit (on-the-fly beim Laden)    │
│  KV-Cache: BF16 → RQ3/TQ3 (compressed, 2.3× less)  │
│  Pruning:  RIY (orthogonal, runtime)                │
│  MTP:      Drafter quantisiert (eigene Bits)        │
│                                                     │
│  Intern: FP8/BF16 als Übergabeformat zwischen Ops  │
└─────────────────────────────────────────────────────┘
```

## Status

### Phase 1: Architektur (sofort)
- [ ] M1: `vllm/multiquant/base.py` — ABC KVQuantizerConfig, KVQuantizer
- [ ] M2: `vllm/multiquant/registry.py` — Quantizer-Registry
- [ ] M3: `vllm/turboquant/` → `vllm/multiquant/turboquant/` migrieren
- [ ] M4: `turboquant_attn.py` → `multiquant_attn.py` generalisieren
- [ ] M5: Backend-Registrierung: TURBOQUANT → MULTIQUANT
- [ ] M6: `attention.py`: `_init_turboquant_buffers` → `_init_multiquant_buffers`
- [ ] M7: CacheDType + torch_utils: rq3, rq4 hinzufügen
- [ ] M8: cuda.py: Backend-Priorität für rq-Prefix

### Phase 2: RotorQuant (nach M1-M8)
- [ ] Siehe TASK.roto.md

### Phase 3: Gewichts-Quantisierung (perspektivisch)
- [ ] M9: `MultiQuantOnlineLinearMethod` mit `uses_meta_device=True`
- [ ] M10: `--quantization multiquant --quant-bits N` CLI
- [ ] M11: MoE-Gewichte on-the-fly quantisieren
- [ ] M12: MTP Drafter-Gewichte quantisieren

### Phase 4: Extreme Quantisierung (Zukunft)
- [ ] M13: 1-bit KV-Cache (2 Centroids, starke QJL-Korrektur)
- [ ] M14: 1-bit Gewichte (Binary/Ternary mit Skalierung)
- [ ] M15: Block-basierte Quantisierung (pro Block eigene Centroids)

## Architektur

### Quantizer Interface

```python
# vllm/multiquant/base.py

class KVQuantizerConfig(ABC):
    head_dim: int
    total_bits: int

    @abstractmethod
    def key_packed_size(self) -> int: ...

    @abstractmethod
    def cache_head_size(self) -> int: ...

    @classmethod
    @abstractmethod
    def from_cache_dtype(cls, dtype_str: str, head_dim: int) -> Self: ...


class KVQuantizer(ABC):
    @abstractmethod
    def init_buffers(self, head_dim: int, seed: int) -> dict[str, Tensor]: ...

    @abstractmethod
    def pack(self, key: Tensor, buffers: dict) -> Tensor: ...

    @abstractmethod
    def unpack(self, packed: Tensor, buffers: dict) -> Tensor: ...

    @abstractmethod
    def attention_score(self, query: Tensor, packed_key: Tensor, buffers: dict) -> Tensor: ...
```

### Registry

```python
# vllm/multiquant/registry.py

QUANTIZER_REGISTRY: dict[str, tuple[type[KVQuantizerConfig], type[KVQuantizer]]] = {
    "tq3": (TurboQuantConfig, TurboQuantizer),
    "tq4": (TurboQuantConfig, TurboQuantizer),
    "rq3": (RotorQuantConfig, RotorQuantizer),
    "rq4": (RotorQuantConfig, RotorQuantizer),
}

def get_kv_quantizer(dtype_str: str) -> tuple[KVQuantizerConfig, KVQuantizer]: ...
def is_multiquant_dtype(dtype_str: str) -> bool: ...
```

### Verzeichnisstruktur

```
vllm/multiquant/
├── __init__.py                    # Exports
├── base.py                        # ABC: KVQuantizerConfig, KVQuantizer
├── registry.py                    # QUANTIZER_REGISTRY
├── turboquant/                    # TurboQuant (migriert von vllm/turboquant/)
│   ├── __init__.py
│   ├── config.py                  # TurboQuantConfig(KVQuantizerConfig)
│   ├── quantizer.py               # TurboQuantizer(KVQuantizer)
│   └── centroids.py               # Lloyd-Max Centroids
├── rotorquant/                    # RotorQuant (neu, aus scrya-com/rotorquant)
│   ├── __init__.py
│   ├── config.py                  # RotorQuantConfig(KVQuantizerConfig)
│   ├── quantizer.py               # RotorQuantizer(KVQuantizer)
│   ├── clifford.py                # Cl(3,0) Algebra
│   └── kernels.py                 # Triton/CUDA Kernels
└── weight_quant/                  # Gewichts-Quantisierung (Phase 3)
    ├── __init__.py
    ├── online_linear.py           # MultiQuantOnlineLinearMethod
    └── online_moe.py              # MultiQuantOnlineMoEMethod
```

## Touch Points in vLLM

| Datei | Was | Warum |
|-------|-----|-------|
| `vllm/config/cache.py` | CacheDType Literal | rq3, rq4 hinzufügen |
| `vllm/utils/torch_utils.py` | STR_DTYPE_TO_TORCH_DTYPE | rq3/rq4 → uint8 |
| `vllm/v1/attention/backend.py` | `is_quantized_kv_cache()` | rq-Prefix erkennen |
| `vllm/v1/attention/backends/registry.py` | AttentionBackendEnum | MULTIQUANT statt TURBOQUANT |
| `vllm/v1/attention/backends/multiquant_attn.py` | Backend-Impl | Generalisiert von turboquant_attn.py |
| `vllm/platforms/cuda.py` | `_get_backend_priorities()` | tq/rq → MULTIQUANT |
| `vllm/model_executor/layers/attention/attention.py` | `_init_multiquant_buffers()` | Registry-basiert statt TQ-hardcoded |

## Kompatibilität

- `--kv-cache-dtype tq3` funktioniert weiterhin (Registry-Lookup)
- `--kv-cache-dtype rq3` neu verfügbar
- `--kv-cache-dtype fp8` unverändert (kein MultiQuant)
- Mixed Page-Size (Q5) funktioniert für alle Quantizer
- Hybrid-Modelle (Q6) funktionieren für alle Quantizer
- `TURBOQUANT` Backend-Enum bleibt als Alias für Rückwärtskompatibilität

## Gewichts-Quantisierung: Bestehendes Pattern

vLLM hat bereits Online-Quantisierung via `uses_meta_device`:

```python
# Bestehendes FP8 Online Pattern (vllm/quantization/fp8.py)
class Fp8OnlineLinearMethod(LinearMethodBase):
    uses_meta_device = True

    def create_weights(self, layer, ...):
        weight = ModelWeightParameter(data=torch.empty(..., device="meta"))
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer):
        qweight, scale = ops.scaled_fp8_quant(layer.weight)
        replace_parameter(layer, "weight", qweight)
```

MultiQuant nutzt das gleiche Pattern:
```python
class MultiQuantOnlineLinearMethod(LinearMethodBase):
    uses_meta_device = True

    def process_weights_after_loading(self, layer):
        compressed = self.quantizer.compress_weight(layer.weight)  # BF16 → N-bit
        replace_parameter(layer, "weight", compressed)
```

## Nicht-Ziele (bewusst ausgeklammert)

- Änderungen an BlockPool, KVCacheCoordinator, SingleTypeKVCacheManager
- Änderungen am Scheduler oder Request-Management
- Training/Finetuning-Support
- Gradient-basierte Quantisierung (kein Calibration-Dataset nötig)
