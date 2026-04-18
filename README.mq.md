# MultiQuant — Generisches Quantisierungs-Framework für vLLM

## Was ist MultiQuant?

MultiQuant ist ein Plugin-Framework für beliebige Quantisierungsmethoden in vLLM. Es abstrahiert das Interface zwischen Quantizer und vLLM-Infrastruktur (Attention Backend, Block Manager, Cache Allocation), sodass neue Quantizer mit minimalem Aufwand integriert werden können.

## Motivation

### Problem: Quant-Delivery-Zirkus

Heute muss jedes Modell in jeder Quantisierungsvariante separat erstellt, gespeichert und verteilt werden:

```
Modell (BF16) → GPTQ-INT4 → Upload → Download → Serve
             → AWQ-INT4  → Upload → Download → Serve
             → FP8       → Upload → Download → Serve
             → NVFP4     → Upload → Download → Serve
```

### Lösung: Load BF16, Quantize On-the-Fly

```
Modell (BF16) → Download → vllm serve --quantization multiquant --quant-bits 3
                                       --kv-cache-dtype rq3
                                       --riy-expert-profile profile.json

Ein Download. Quantisierung, Pruning, KV-Cache-Kompression alles zur Laufzeit.
```

## Architektur

```
┌───────────────────────────────────────────────────────┐
│                    vLLM Serve                          │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Attention    │  │  MoE Layer   │  │  MTP Drafter │ │
│  │  Backend      │  │              │  │              │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                  │         │
│  ┌──────▼─────────────────▼──────────────────▼───────┐ │
│  │              MultiQuant Framework                  │ │
│  │                                                    │ │
│  │  ┌────────────┐  ┌─────────────┐  ┌────────────┐  │ │
│  │  │ TurboQuant │  │ RotorQuant  │  │ Future...  │  │ │
│  │  │ (Dense Pi) │  │ (Clifford)  │  │ (1-bit)    │  │ │
│  │  └────────────┘  └─────────────┘  └────────────┘  │ │
│  │                                                    │ │
│  │  Registry: "tq3" → TQ, "rq3" → RQ, ...           │ │
│  └────────────────────────────────────────────────────┘ │
│                                                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Block Manager (Mixed Page-Sizes, Q5)              │ │
│  └────────────────────────────────────────────────────┘ │
│                                                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │  RIY Expert Pruning (orthogonal)                   │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

## Quantizer Interface

Jeder KV-Cache-Quantizer implementiert zwei Klassen:

### KVQuantizerConfig

```python
class KVQuantizerConfig(ABC):
    head_dim: int       # Originale Head-Dimension (128, 256)
    total_bits: int     # Bits pro Koordinate (3, 4)

    def key_packed_size(self) -> int:
        """Bytes pro komprimiertem Key-Vektor."""

    def cache_head_size(self) -> int:
        """Effektive head_size für Cache-Allokation (= key_packed_size)."""

    @classmethod
    def from_cache_dtype(cls, dtype_str: str, head_dim: int) -> Self:
        """Factory: 'tq3' + D=128 → Config."""
```

### KVQuantizer

```python
class KVQuantizer(ABC):
    def init_buffers(self, head_dim: int, seed: int) -> dict[str, Tensor]:
        """Erzeugt Quantizer-Buffers (Pi-Matrix, Rotoren, Centroids, etc.)."""

    def pack(self, key: Tensor, buffers: dict) -> Tensor:
        """Key-Vektor → compressed uint8."""

    def unpack(self, packed: Tensor, buffers: dict) -> Tensor:
        """Compressed uint8 → rekonstruierter Key-Vektor."""

    def attention_score(self, query: Tensor, packed_key: Tensor, buffers: dict) -> Tensor:
        """Berechnet Attention-Score direkt aus compressed Key."""
```

## Registry

```python
from vllm.multiquant import get_kv_quantizer

config, quantizer = get_kv_quantizer("rq3", head_dim=128)
# config: RotorQuantConfig(head_dim=128, total_bits=3)
# quantizer: RotorQuantizer()
```

Registrierte Quantizer:

| Dtype | Quantizer | Bits | Format |
|-------|-----------|------|--------|
| `tq3` | TurboQuant | 3 | MSE(2bit) + QJL(1bit) + Norms |
| `tq4` | TurboQuant | 4 | MSE(3bit) + QJL(1bit) + Norms |
| `rq3` | RotorQuant | 3 | MSE(2bit) + QJL(1bit) + Norms |
| `rq4` | RotorQuant | 4 | MSE(3bit) + QJL(1bit) + Norms |

## Verwendung

```bash
# RotorQuant 3-bit KV-Cache (schnellste Kompression)
vllm serve <model> --kv-cache-dtype rq3

# TurboQuant 4-bit KV-Cache (bessere Qualität)
vllm serve <model> --kv-cache-dtype tq4

# FP8 Baseline (unverändert)
vllm serve <model> --kv-cache-dtype fp8

# Kombination mit RIY Pruning
vllm serve <model> --kv-cache-dtype rq3 --riy-expert-profile profile.json

# Perspektive: On-the-fly Gewichtsquantisierung
vllm serve <model> --quantization multiquant --quant-bits 3 --kv-cache-dtype rq3
```

## Block Manager Integration

MultiQuant nutzt die Mixed Page-Size Infrastruktur (Q5):

```
Attention Layers → Kleine Blocks (rq3: 68 B/Key × num_kv_heads × block_size)
Mamba Layers     → Große Blocks (State-Size abhängig)
                   ↓
BlockPool: Ein globales num_blocks, verschiedene Tensor-Größen
```

Hybrid-Modelle (Qwen3.5, Jamba, etc.) werden automatisch unterstützt.

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| Q1-Q6 | TurboQuant KV-Cache | Done |
| Phase 1 | MultiQuant Architektur + RotorQuant | In Progress |
| Phase 2 | On-the-fly Gewichtsquantisierung | Geplant |
| Phase 3 | 1-bit, Block-Quantisierung | Zukunft |

## Dateien

```
vllm/multiquant/
├── __init__.py           # Exports + get_kv_quantizer()
├── base.py               # ABC: KVQuantizerConfig, KVQuantizer
├── registry.py           # QUANTIZER_REGISTRY
├── turboquant/           # TurboQuant (Dense-Matrix)
│   ├── config.py
│   ├── quantizer.py
│   └── centroids.py
├── rotorquant/           # RotorQuant (Clifford-Rotoren)
│   ├── config.py
│   ├── quantizer.py
│   ├── clifford.py
│   └── kernels.py
└── weight_quant/         # Gewichts-Quantisierung (Phase 2)
    ├── online_linear.py
    └── online_moe.py
```
