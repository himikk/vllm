# Asymmetrisches K/V-Cache Layout

## Motivation

Standard-LLM-Inferenz behandelt Keys und Values im KV-Cache identisch: gleicher
Datentyp, gleiche Größe pro Element. Das ist bei FP16 und FP8 sinnvoll, weil
beide Operanden gleich stark quantisiert werden.

**TurboQuant ändert das fundamental**: Keys werden auf 3-4 Bit komprimiert,
Values bleiben bei voller Präzision. Ein symmetrisches Cache-Layout verschwendet
Speicher, weil die Key-Slots für die volle Breite allokiert aber nur zu 20-40%
genutzt werden.

## Das Problem

```
Symmetrisches Layout (aktuell):
┌───────────────────────────────────────────┐
│  K-Cache: head_size × dtype_size Bytes    │  ← 256 × 2 = 512 Bytes (BF16)
│  V-Cache: head_size × dtype_size Bytes    │  ← 256 × 2 = 512 Bytes (BF16)
└───────────────────────────────────────────┘
  Total: 1024 Bytes pro Token×Head

Mit FP8 (symmetrisch, funktioniert):
┌───────────────────────────────────────────┐
│  K-Cache: head_size × 1 Byte             │  ← 256 Bytes (FP8)
│  V-Cache: head_size × 1 Byte             │  ← 256 Bytes (FP8)
└───────────────────────────────────────────┘
  Total: 512 Bytes — 2× Kompression ✓

Mit TQ3 im symmetrischen Layout (Speicher verschwendet):
┌───────────────────────────────────────────┐
│  K-Cache: 100 Bytes gepackt + 156 Padding │  ← 256 Bytes Slot, 100 genutzt
│  V-Cache: head_size × 2 Bytes            │  ← 512 Bytes (BF16)
└───────────────────────────────────────────┘
  Total: 768 Bytes — K-Padding verschwendet 156 Bytes (20%)
```

## Die Lösung: Asymmetrisches Layout

```
┌───────────────────────────────────────────┐
│  K-Cache: tq_packed_size Bytes            │  ← 100 Bytes (TQ3) / 128 Bytes (TQ4)
│  V-Cache: head_size × dtype_size Bytes    │  ← 512 Bytes (BF16)
└───────────────────────────────────────────┘
  Total TQ3: 612 Bytes — 1.67× Kompression
  Total TQ4: 640 Bytes — 1.60× Kompression
```

## Warum wurde es symmetrisch gebaut?

Weil bisher **alle** KV-Cache-Quantisierungen K und V gleich behandeln:

| Ansatz | K-Bits | V-Bits | Symmetrisch? |
|--------|--------|--------|-------------|
| BF16   | 16     | 16     | ✓           |
| FP8    | 8      | 8      | ✓           |
| KIVI   | 2-4    | 2-4    | ✓           |
| **TQ** | **3-4**| **16** | **✗**       |

TurboQuant ist der erste Ansatz mit theoretischer Begründung warum Keys
stärker komprimiert werden können als Values:

- **Keys** → bestimmen Attention Scores (`Q·K^T`). TQ's QJL-Korrektur
  liefert **unbiased** Inner Products auch bei 3 Bit — mathematisch bewiesen.
- **Values** → werden per Softmax-gewichteter Summe aggregiert. Fehler
  mitteln sich statistisch raus, aber es gibt keinen vergleichbaren
  Bias-Korrektur-Mechanismus für Values.

## Scope des Eingriffs

**Nur vLLM** — FlashInfer/FlashAttention bleiben unverändert.

Die Attention-Kernels (FlashInfer, FlashAttn, PagedAttention) bekommen
weiterhin Standard-Shape KV-Tensoren. Die asymmetrische Kompression ist
**transparent**: Keys werden on-demand dekomprimiert bevor sie an den
Attention-Kernel übergeben werden.

```
Store-Pfad:
  Key(BF16) → CUDA tq_compress_and_store() → K-Region(gepackt, 100B)
  Value(BF16) → reshape_and_cache_flash() → V-Region(BF16, 512B)

Read-Pfad:
  K-Region(100B) → CUDA tq_decompress() → temp_k(BF16, 512B)
  V-Region(512B) → direkt
  FlashInfer(query, temp_k, v_cache) → output
```

## Betroffene vLLM-Dateien

| Datei | Was ändert sich |
|-------|----------------|
| `vllm/v1/kv_cache_interface.py` | `TQAttentionSpec` mit asymmetrischer `page_size_bytes` |
| `vllm/v1/worker/gpu/attn_utils.py` | Cache-Reshape gibt separate K/V Views zurück |
| `vllm/model_executor/layers/attention/attention.py` | Store-Hook: compress K, store V |
| `csrc/quantization/turboquant/` | Compress-Store + Decompress CUDA Kernels |
| `vllm/turboquant/quantizer.py` | Python Dispatch |

## Speicher-Gewinn

Qwen3.5-35B (head_dim=256, 2 KV-Heads, 10 GB KV-Budget):

| Layout | Bytes/Token×Head | Max Tokens (10GB) | vs BF16 |
|--------|-----------------|-------------------|---------|
| BF16 (symmetrisch) | 1024 | ~5.1M | 1.0× |
| FP8 (symmetrisch) | 512 | ~10.2M | 2.0× |
| **TQ4 (asymmetrisch)** | **640** | **~8.2M** | **1.6×** |
| **TQ3 (asymmetrisch)** | **612** | **~8.6M** | **1.7×** |
| TQ3 + FP8 Values | 356 | ~14.8M | 2.9× |

## Vergleich mit existierenden Ansätzen

| Feature | FP8 KV | KIVI | TurboQuant |
|---------|--------|------|-----------|
| K-Bits | 8 | 2-4 | 3-4 |
| V-Bits | 8 | 2-4 | 16 (BF16) |
| Calibration nötig | Nein | Nein | Nein |
| Mathematisch unbiased | Nein | Nein | **Ja (QJL)** |
| Asymmetrisch K≠V | Nein | Nein | **Ja** |
| Speicher-Gewinn | 2× | 4-8× | 1.7× (Phase 1) |
| Quality bei 3-4 Bit | Schlecht | Mittel | **Excellent (cos>0.9)** |
