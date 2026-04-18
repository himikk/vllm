# TASK: Asymmetrisches K/V-Cache Layout für TurboQuant

**Branch**: `turboquant` (aufbauend auf Phase 1-3)
**Ziel**: Echte Speicherersparnis durch komprimierte Keys im KV-Cache

---

## Problem

Phase 1 TurboQuant beweist: Mathematik funktioniert (100% Math, cos>0.9), CUDA Kernel
läuft auf SM121. **Aber der KV-Cache spart keinen Speicher** — Keys werden nach dem
TQ Round-Trip als volle BF16 zurückgeschrieben.

```
Phase 1 (aktuell):
  Key(BF16) → TQ(compress→decompress) → Key'(BF16) → Cache(BF16)
  512 Bytes rein, 512 Bytes raus. Gleicher Speicher wie ohne TQ.

Phase 2 (dieses Task):
  Key(BF16) → TQ(compress) → Cache(100 Bytes)  ← 5× kleiner
  Value(BF16) → Cache(512 Bytes)                ← unchanged
  Gesamt: 612 Bytes statt 1024 Bytes → 1.7× Kompression
```

## Warum symmetrisch nicht reicht

vLLM's KV-Cache Layout ist `(2, num_blocks, block_size, num_kv_heads, head_size)`.
Die `2` steht für K und V — **beide teilen dieselbe `head_size` Dimension**.

Bisherige Quantisierungsansätze (FP8, FP8_E4M3) komprimieren K und V gleich stark:
K=1byte, V=1byte → symmetrisch. Das Layout funktioniert.

TurboQuant komprimiert **nur Keys** aggressiv (3-4 Bit), Values bleiben BF16:
- K: 100 Bytes (TQ3, D=256) oder 128 Bytes (TQ4)
- V: 512 Bytes (BF16, D=256)

Mit dem symmetrischen Layout (`head_size=256`, uint8) hat K 256 Bytes pro Slot
aber nutzt nur 100 → 156 Bytes Padding. Kein Vorteil über FP8 hinaus.

## Lösung: Asymmetrisches Block-Layout

Statt K und V als identische Tensoren zu behandeln, wird jeder Cache-Block
als **flaches Byte-Array** mit verschiedenen K/V-Regionen interpretiert:

```
Block Layout (ein Token × ein Head):
┌──────────────────┬──────────────────────────────────┐
│  K compressed     │  V full precision (BF16)          │
│  100 Bytes (TQ3)  │  512 Bytes                        │
│  128 Bytes (TQ4)  │                                   │
└──────────────────┴──────────────────────────────────┘
Total pro Slot: 612 Bytes (TQ3) oder 640 Bytes (TQ4)
vs 1024 Bytes (BF16) oder 512 Bytes (FP8)
```

## Architektur-Eingriffe (nur vLLM, FlashInfer unverändert)

### 1. Page-Size Berechnung

**Datei**: `vllm/v1/kv_cache_interface.py`

Neue `TQAttentionSpec` Subklasse die `real_page_size_bytes` überschreibt:

```python
@dataclass(frozen=True, kw_only=True)
class TQAttentionSpec(AttentionSpec):
    tq_packed_size: int  # Bytes pro komprimiertem Key-Vektor

    @property
    def real_page_size_bytes(self) -> int:
        k_bytes = self.block_size * self.num_kv_heads * self.tq_packed_size
        v_bytes = self.block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)
        return k_bytes + v_bytes
```

Für Qwen3.5-35B (D=256, 2 KV-heads, block_size=1056, TQ3):
- K: 1056 × 2 × 100 = 211 KB
- V: 1056 × 2 × 512 = 1,055 KB
- **Total: 1,266 KB** statt 2,110 KB (BF16) → **1.67× Kompression**
- Bei 10 GB KV-Budget: ~8,100 Blocks statt ~4,850 → **67% mehr Tokens**

### 2. Cache-Allokation + Reshape

**Datei**: `vllm/v1/worker/gpu/attn_utils.py`

Die Allokation bleibt als `torch.zeros(..., dtype=torch.int8)`. Nur die Reshape-Logik
muss für TQ-Layers angepasst werden:

```python
# Statt: raw_tensor.view(dtype).view(num_blocks, 2, block_size, heads, head_size)
# TQ:    raw_tensor bleibt als int8, wird als Struct-of-Arrays interpretiert
#        k_cache = raw_tensor[:k_bytes].view(num_blocks, block_size, heads, packed_size)
#        v_cache = raw_tensor[k_bytes:].view(num_blocks, block_size, heads, head_size).view(dtype)
```

### 3. Compress-on-Store (K-Pfad)

**Datei**: `vllm/model_executor/layers/attention/attention.py` (unified_kv_cache_update Hook)

Statt tq_round_trip (compress→decompress→store BF16):

```python
if _tq_enabled:
    # K: compress + pack direkt in K-Region des Cache
    tq_compress_and_store(key, k_cache, slot_mapping, Pi, S, centroids)
    # V: normal in V-Region schreiben
    reshape_and_cache_flash(dummy, value, v_cache, ...)
```

### 4. Decompress-on-Read (vor Attention)

**Datei**: FlashInfer Forward-Pfad (Hook in attention.py oder flashinfer.py)

```python
if _tq_enabled:
    # Dekomprimiere K-Blocks in temporären BF16-Buffer
    temp_k = decompress_k_blocks(k_cache, block_table, seq_lens, Pi, S, centroids)
    # Baue Standard kv_cache Tensor für FlashInfer
    kv_for_flashinfer = stack_kv(temp_k, v_cache)
    # FlashInfer liest wie gewohnt
```

### 5. Temporärer Dekompressions-Buffer

Pre-allokierter Buffer für dekomprimierte Keys der aktiven Batch-Requests.
Größe: `max_num_seqs × max_seq_len × num_kv_heads × head_size × 2` (BF16).

Für Batch=1, 256K Context: 256K × 2 × 256 × 2 = 256 MB.
Für Batch=1, 32K Context: 32 MB.

Der Buffer wird einmal allokiert und wiederverwendet.

## Neue/Geänderte Dateien

| Datei | Änderung |
|-------|----------|
| `vllm/v1/kv_cache_interface.py` | `TQAttentionSpec` mit asymmetrischer page_size |
| `vllm/v1/worker/gpu/attn_utils.py` | Asymmetrisches Reshape für TQ-Layers |
| `vllm/model_executor/layers/attention/attention.py` | Compress-on-Store Hook |
| `csrc/quantization/turboquant/tq_compress_store.cu` | CUDA: Key→Packed direkt in Cache |
| `csrc/quantization/turboquant/tq_decompress.cu` | CUDA: Packed→BF16 für Attention |
| `vllm/turboquant/quantizer.py` | Python Wrapper für compress/decompress |
| `Dockerfile.tq` | Image-Update |

## Speicher-Vergleich (Qwen3.5-35B, 10 GB KV-Budget)

| Cache-Typ | Bytes/Slot | Blocks | Max Tokens | Kompression |
|-----------|-----------|--------|------------|-------------|
| BF16      | 1024      | ~4,850 | ~5.1M      | 1.0×        |
| FP8       | 512       | ~9,700 | ~10.2M     | 2.0×        |
| **TQ4**   | **640**   | ~7,750 | **~8.2M**  | **1.6×**    |
| **TQ3**   | **612**   | ~8,100 | **~8.6M**  | **1.7×**    |
| TQ3+FP8V  | 356       | ~14,000| ~14.8M     | 2.9×        |

*TQ3+FP8V: Keys TQ3-komprimiert (100B) + Values FP8 (256B) — theoretisch möglich*

## Risiken

1. **vLLM Block-Manager Kompatibilität**: Slot-Mapping und Block-Table erwarten
   symmetrisches Layout. Muss getestet werden ob asymmetrische Byte-Offsets funktionieren.

2. **Temp-Buffer Overhead**: 256 MB bei 256K Context für den Dekompressions-Buffer.
   Kann durch blockweises Dekomprimieren (Tiling) reduziert werden.

3. **Dekompressions-Latenz**: ~33µs × Anzahl Blocks pro Token beim Decode.
   Bei 128K Context: ~4ms — spürbar bei 30 tok/s Baseline.

4. **Prefix Caching**: Komprimierte Blocks haben andere Hashes als BF16-Blocks.
   Prefix Caching muss TQ-aware sein.

## Implementierungs-Reihenfolge

```
1. TQAttentionSpec mit asymmetrischer page_size         ~ einfach
2. Asymmetrisches Reshape in attn_utils.py               ~ mittel
3. Compress-on-Store CUDA Kernel (adapt v2 + pack)        ~ mittel
4. Decompress-on-Read CUDA Kernel (neuer Kernel)          ~ mittel
5. FlashInfer Forward-Hook (temp buffer + decompress)     ~ komplex
6. Integration Testing (Quality + Memory + Throughput)    ~ mittel
7. Dockerfile + Benchmark                                 ~ einfach
```
