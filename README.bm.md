# Mixed Page-Size Block Management (Q5)

## Problem

vLLM's KV-Cache Block-Manager verlangt eine einheitliche `page_size_bytes` für alle Layer. Das funktioniert für reine Transformer-Modelle (alle Layer haben gleiche KV-Heads/Head-Size), scheitert aber bei:

- **Hybrid-Modellen** (z.B. Qwen3.5 mit Mamba SSM + Attention)
- **TQ-komprimiertem KV-Cache** (TQ3: 28 Bytes/Head statt 128 Bytes/Head)

Wenn TQ-Attention und Mamba verschiedene Page-Sizes haben (z.B. 3.5 KB vs 2 MB) und diese nicht ganzzahlig teilbar sind, wirft `unify_kv_cache_spec_page_size()` einen `NotImplementedError`.

## Lösung: Per-Page-Size Tensor-Gruppen

### Kernidee

Der `BlockPool` verwaltet abstrakte Block-Indices (0..num_blocks-1). Jeder `KVCacheTensor` hat eigene physische Größe. Block-Index 5 bedeutet:

```
TQ-Tensor:    Offset 5 × 3584 B  = 17.920 B
Mamba-Tensor: Offset 5 × 2 MB    = 10 MB
```

**Verschiedene Tensors können verschiedene Page-Sizes haben** — solange `num_blocks` global gleich bleibt.

### Berechnung

```
num_blocks = available_memory / (group_size_tq × page_size_tq + group_size_mamba × page_size_mamba)
```

Jeder Tensor bekommt `page_size_i × num_blocks` Bytes. Kein Padding, kein Speicherverlust.

### Constraint

Layers die einen Tensor **teilen** müssen gleiche Page-Size haben. Layers in **verschiedenen** Tensors nicht.

## Architektur

```
                    BlockPool (num_blocks global)
                    ┌─────────────────────────┐
                    │ Block 0, 1, 2, ... N    │
                    └────┬───────────┬────────┘
                         │           │
              ┌──────────┴──┐  ┌─────┴──────────┐
              │ TQ-Tensors  │  │ Mamba-Tensors   │
              │ page=3584 B │  │ page=2 MB       │
              │ size=N×3584 │  │ size=N×2MB      │
              └─────────────┘  └─────────────────┘
```

## Code-Änderungen

### `vllm/v1/core/kv_cache_utils.py`

| Funktion | Änderung |
|----------|----------|
| `get_kv_cache_groups()` | Try-catch um `unify_kv_cache_spec_page_size()`, Fallback auf Mixed-Path |
| `_get_kv_cache_groups_mixed_page_size()` | **NEU** — Gruppierung ohne Page-Size-Unifizierung |
| `get_kv_cache_config_from_groups()` | Mixed-Path: separate Tensors pro Page-Size-Typ |
| `_max_memory_usage_bytes_from_groups()` | Mixed-Path: per-Page-Size Speicherberechnung |

### Unverändert

- `BlockPool`, `KVCacheCoordinator`, `SingleTypeKVCacheManager`
- `unify_kv_cache_spec_page_size()` (nur try-catch drumrum)
- Reine Transformer-Modelle (nehmen den bisherigen Pfad)
- FP8/BF16 auf Hybrid-Modellen (Unifizierung erfolgreich)

## Pfade

| Szenario | Pfad |
|----------|------|
| GLM-4.7 + TQ3 | `is_kv_cache_spec_uniform()` → True → Uniform-Path (kein Mixed) |
| GLM-4.7 + FP8 | `is_kv_cache_spec_uniform()` → True → Uniform-Path |
| Qwen3.5 + FP8 | `unify_kv_cache_spec_page_size()` → Erfolg → Uniform-Page-Size-Path |
| Qwen3.5 + TQ3 | `unify_kv_cache_spec_page_size()` → `NotImplementedError` → **Mixed-Path** |

## Tests

```bash
# Im Container:
pytest /opt/tests/test_kv_cache_utils.py::test_mixed_page_size_groups -xvs
pytest /opt/tests/test_kv_cache_utils.py::test_mixed_page_size_config_from_groups -xvs

# Alle kv_cache_utils Tests (51 pass):
pytest /opt/tests/test_kv_cache_utils.py -x -q
```

## Image

```bash
podman build -f Dockerfile.tq-bm -t vllm-riy-tq-bm .
```

Basis: `localhost/vllm-ng17e-tq` + Q5 kv_cache_utils.py + pytest
