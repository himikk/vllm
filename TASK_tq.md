# TASK: TurboQuant KV-Cache Quantisierung in vLLM

**Paper**: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
**Autoren**: Zandieh, Daliri, Hadian, Mirrokni (Google Research / NYU / DeepMind), ICLR 2026
**Branch**: `turboquant` (basiert auf `riy-pr-rebase`)

---

## Status: Phase 1 DONE (2025-03-25)

### Implementiert und getestet (CPU):
- CacheDType `tq3`/`tq4` registriert in Config + Torch Utils + Registry
- TurboQuant Core-Modul: Lloyd-Max Centroids, Quantizer, Pack/Unpack
- Attention Backend: `TurboQuantAttentionBackend` mit PyTorch Fallback + Triton Dispatch
- Attention Layer: Pi/S/centroids Buffers pro Layer
- Triton Kernel Entwuerfe: `triton_tq_reshape_and_cache.py` + `triton_tq_attention_score.py`
- **Kritischer Bug gefixed**: `q @ PiT.T` → `q @ PiT` (PiT = Pi^T, also PiT.T = Pi — falsch!)

### Test-Ergebnisse (CPU, PyTorch):
| Test | TQ3 | TQ4 |
|------|-----|-----|
| Score Korrelation | 0.92 | 0.98 |
| Output Cosine | 0.92 | 0.98 |
| IP Bias | < 0.005 | < 0.002 |
| Needle Top-1 | 100% | 100% |
| Pack/Unpack | Lossless | Lossless |
| Key Kompression | 4.9x | 3.8x |

### Noch offen:
- MetadataBuilder (Stub, braucht volle Scheduler-Integration)
- Triton Kernels auf GPU validieren (kein GPU auf diesem Host)
- Value-Kompression (Phase 1 = FP16 Values)
- E2E Serving auf DGX Spark: `bash tests/turboquant/run_on_dgx.sh test`

---

## Zusammenfassung

TurboQuant komprimiert den KV-Cache auf 2.5–3.5 Bit pro Koordinate bei nahezu null Qualitaetsverlust.
Zwei Stufen:

1. **PolarQuant** (b-1 Bit): Zufaellige Rotation → skalare Quantisierung pro Koordinate
2. **QJL** (1 Bit): Residual-Korrektur via sign-bit Projektion → unverzerrte Inner Products

Kernvorteil: Rein matrixbasiert (GEMV/GEMM), kein Training, kein Calibration-Daten, online anwendbar.

---

## Algorithmus

### TurboQuant_prod (Algorithm 2 aus Paper)

**Setup (einmalig pro head_dim d):**
```
Pi = QR(randn(d, d))          # Rotation matrix, orthogonal
S  = randn(d, d)              # QJL projection matrix
centroids = lloyd(f_X, b-1)   # 2^(b-1) optimale Centroids fuer Beta-Verteilung
```

**Quantisierung (pro KV-Vektor x):**
```
1. norm = ||x||_2                           # Norm separat speichern
2. x_hat = x / norm                         # Auf Einheitskugel normieren
3. y = Pi @ x_hat                           # Rotieren
4. idx[j] = argmin_k |y[j] - centroid[k]|  # Skalare Quantisierung, (b-1) Bit/Koord.
5. x_mse = Pi^T @ centroids[idx]            # Rekonstruktion
6. r = x_hat - x_mse                        # Residual
7. gamma = ||r||_2                           # Residual-Norm
8. qjl = sign(S @ r)                        # QJL: 1 Bit/Koord.
```
Gespeichert: `(idx, qjl, norm, gamma)` → b*d + 32 + 16 Bit pro Vektor

**Dequantisierung / Attention Inner Product:**
```
<q, x_tilde> = norm * (<q, Pi^T @ centroids[idx]> + gamma * sqrt(pi/2)/d * <q, S^T @ qjl>)
```

### Centroids (vorberechnet, head_dim d)

| Bits | Centroids (skaliert mit 1/sqrt(d)) |
|------|-------------------------------------|
| 1 | +/- sqrt(2/(pi*d)) |
| 2 | +/- 0.453/sqrt(d), +/- 1.51/sqrt(d) |
| 3 | 8 Centroids via Max-Lloyd auf Beta((d-1)/2, (d-1)/2) |

### Bit-Konfigurationen

| Config | MSE Bits | QJL Bit | Outlier | Effektiv | Kompression vs FP16 |
|--------|----------|---------|---------|----------|---------------------|
| TQ-2.5 | 1 | 1 | 32ch@3bit | 2.5 | 6.4x |
| TQ-3   | 2 | 1 | - | 3.0 | 5.3x |
| TQ-3.5 | 2 | 1 | 32ch@4bit | 3.5 | 4.6x |
| TQ-4   | 3 | 1 | - | 4.0 | 4.0x |

---

## vLLM Architektur-Analyse

### Bestehender FP8 KV-Cache Flow

```
Forward Pass
  │
  ├─ reshape_and_cache_flash()           # Quantize on Store
  │   ├─ CUDA: csrc/cache_kernels.cu:143   (CopyWithScaleOp, fp8::scaled_convert)
  │   └─ Triton: v1/attention/ops/triton_reshape_and_cache_flash.py:11
  │
  └─ Attention Computation
      └─ gather_and_maybe_dequant_cache()  # Dequantize on Load
          └─ CUDA: csrc/cache_kernels.cu:840
```

### Relevante Dateien

| Datei | Rolle |
|-------|-------|
| `vllm/config/cache.py:14` | `CacheDType` Literal — neuen Typ hinzufuegen |
| `vllm/model_executor/layers/quantization/kv_cache.py` | `BaseKVCacheMethod` — Scales/Params |
| `vllm/v1/attention/backend.py:939` | `is_quantized_kv_cache()` |
| `vllm/v1/attention/backends/flash_attn.py:819` | `_update_kv_cache()` Call-Site |
| `csrc/cache_kernels.cu:128-200` | CUDA reshape+quantize Kernel |
| `csrc/cache_kernels.cu:840-914` | CUDA gather+dequantize Kernel |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | Triton Alternative |
| `vllm/model_executor/layers/attention/attention.py:117` | `_init_kv_cache_quant()` |

---

## vLLM Verankerung: Exakte Integrationspunkte

TurboQuant ist fundamental anders als FP8: FP8 ist ein Dtype-Swap (gleiche Tensor-Shape,
anderer Typ). TurboQuant hat ein **komplett anderes Cache-Layout** (gepackte Indices + Signs + Norms)
und braucht **eigene Attention-Kernels** (kein Standard-FlashAttention auf komprimierten Daten).

### Strategie: Eigenes Attention-Backend

Statt den FP8-Pfad zu erweitern (Dtype-Dispatch), implementieren wir ein **eigenes
Attention-Backend** `TurboQuantAttentionBackend`. Das ist der sauberste Weg, weil:

1. Das Cache-Layout (Shape + Dtype) fundamental anders ist
2. Die Attention-Berechnung eigene Kernels braucht (Fused Score aus komprimierten Keys)
3. Prefill und Decode verschiedene Pfade haben (Prefill: FlashAttn, Decode: Fused TQ)

### Hook 1: CacheDType erweitern

**`vllm/config/cache.py:14-23`** — Neuen Typ registrieren:

```python
CacheDType = Literal[
    "auto", "float16", "bfloat16",
    "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla",
    "tq3", "tq4",    # ← NEU: TurboQuant 3-bit, 4-bit
]
```

**`vllm/v1/attention/backend.py:939`** — Quantisierungs-Check erweitern:

```python
def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype.startswith("fp8") or kv_cache_dtype.startswith("tq")
```

### Hook 2: CLI → CacheConfig

**`vllm/engine/arg_utils.py:394`** — Kein Code noetig, `--kv-cache-dtype tq3` fliesst
automatisch durch `kv_cache_dtype: CacheDType` und `resolve_kv_cache_dtype_string()`.

Einzige Aenderung: `resolve_kv_cache_dtype_string()` muss `tq3`/`tq4` akzeptieren
(aktuell nur FP8-Varianten + auto/float16/bfloat16).

### Hook 3: Cache-Allokation (andere Shape!)

**`vllm/v1/worker/gpu/attn_utils.py:93-151`** — Cache-Tensoren werden hier allokiert.

FP8: Gleiche Shape wie FP16, nur anderer Dtype.
TQ3: **Komplett andere Shape** pro Block:

```
FP8:  [num_blocks, 2, block_size, num_kv_heads, head_size]   dtype=fp8
TQ3:  [num_blocks, 2, block_size, num_kv_heads, tq_packed_size]  dtype=uint8
      wobei tq_packed_size = ceil(head_size * mse_bits / 8)   # MSE Indices
                           + ceil(head_size / 8)               # QJL Signs
                           + 4                                 # norm + gamma (2x fp16)
```

Fuer head_size=128, tq3 (2-bit MSE + 1-bit QJL):
- MSE: 128*2/8 = 32 Bytes
- QJL: 128/8 = 16 Bytes
- Norms: 4 Bytes
- **Total: 52 Bytes** vs 256 Bytes (FP16) = **4.9x Kompression**

**Integration**: `get_kv_cache_shape()` im TQ-Backend gibt die gepackte Shape zurueck.
Die Allokation in `_allocate_kv_cache()` nutzt `torch.int8` als Basis → passt.

### Hook 4: Attention Backend registrieren

**Neues File: `vllm/v1/attention/backends/turboquant_attn.py`**

```python
class TurboQuantAttentionBackend(AttentionBackend):
    supported_kv_cache_dtypes = ["tq3", "tq4"]

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads,
                           head_size, cache_dtype_str):
        packed_size = _tq_packed_size(head_size, cache_dtype_str)
        return (num_blocks, 2, block_size, num_kv_heads, packed_size)

    class TurboQuantImpl(AttentionImpl):
        def __init__(self, ...):
            self.tq = TurboQuantizer(head_dim, bits, ...)
            # Pi, S, centroids als Buffers registrieren

        def _update_kv_cache(self, key, value, kv_cache, slot_mapping, layer):
            # STATT reshape_and_cache_flash():
            tq_reshape_and_cache(key, value, kv_cache, slot_mapping,
                                 self.tq.Pi, self.tq.S, self.tq.centroids)

        def forward(self, query, key, value, kv_cache, attn_metadata, ...):
            if is_prefill:
                # Prefill: FlashAttention auf unkomprimierten K/V
                # Danach quantisieren und in Cache schreiben
                self._update_kv_cache(key, value, kv_cache, ...)
                return flash_attn_varlen_func(query, key, value, ...)
            else:
                # Decode: Fused TQ-Score direkt aus komprimiertem Cache
                self._update_kv_cache(key, value, kv_cache, ...)
                scores = tq_fused_attention_score(
                    query, kv_cache, self.tq.Pi, self.tq.S,
                    self.tq.centroids, attn_metadata)
                # Softmax + V-Aggregation
                return self._aggregate_values(scores, kv_cache, ...)
```

**`vllm/v1/attention/selector.py`** — TQ-Backend in die Selektion einbauen:

```python
# In _cached_select_attention_backend():
if kv_cache_dtype.startswith("tq"):
    return TurboQuantAttentionBackend
```

### Hook 5: Attention Layer — TQ-spezifische Buffers

**`vllm/model_executor/layers/attention/attention.py:90-110`**

FP8: Registriert `_k_scale`, `_v_scale` (float32 Skalare).
TQ: Braucht stattdessen Pi, S, centroids pro Layer:

```python
def set_tq_buffers(layer, head_dim, bits, layer_idx):
    seed = layer_idx * 1337  # deterministisch pro Layer
    Pi = generate_rotation_matrix(head_dim, seed=seed)
    S  = generate_qjl_matrix(head_dim, seed=seed+1)
    centroids = precomputed_centroids[head_dim][bits-1]  # Lookup-Table

    layer.register_buffer("_tq_Pi", Pi)
    layer.register_buffer("_tq_S", S)
    layer.register_buffer("_tq_centroids", centroids)
    layer.register_buffer("_tq_norm_scale", torch.tensor(1.0))
```

**Memory-Kosten**: Pi + S = 2 × 128×128×2 = 64 KB pro Layer (FP16).
Fuer 32 Layers: 2 MB total — vernachlaessigbar.

### Hook 6: Cache Store (Quantize-on-Store)

**Ersetzt**: `reshape_and_cache_flash()` (CUDA: `csrc/cache_kernels.cu:203`)

**Neuer Kernel: `csrc/quantization/turboquant/tq_reshape_and_cache.cu`**

```cuda
// Pro Token, pro Head:
__global__ void tq_reshape_and_cache_kernel(
    scalar_t* key,           // [num_tokens, num_heads, head_dim]
    uint8_t*  kv_cache,      // [num_blocks, 2, block_size, num_heads, packed_size]
    int64_t*  slot_mapping,  // [num_tokens]
    float*    Pi,            // [head_dim, head_dim]
    float*    S,             // [head_dim, head_dim]
    float*    centroids,     // [2^mse_bits]
    int mse_bits, int head_dim
) {
    // 1. Norm + Normalize
    float norm = vec_norm(key[token]);
    float x_hat[D] = key[token] / norm;

    // 2. Rotate: y = Pi @ x_hat (GEMV in shared memory)
    float y[D];
    shared_gemv(Pi, x_hat, y, D);

    // 3. Scalar Quant: idx[j] = nearest(y[j], centroids)
    uint8_t idx[D];
    for j: idx[j] = nearest_centroid(y[j], centroids, mse_bits);

    // 4. Reconstruct: y_hat[j] = centroids[idx[j]]
    // 5. Unrotate: x_mse = Pi^T @ y_hat (GEMV)
    float x_mse[D];
    shared_gemv_transpose(Pi, y_hat, x_mse, D);

    // 6. Residual + QJL
    float r[D] = x_hat - x_mse;
    float gamma = vec_norm(r);
    // sign(S @ r) — reuse QJL kernel pattern from ~/QJL/
    uint8_t signs[D/8];
    qjl_project_and_pack(S, r, signs, D);

    // 7. Pack and store in cache slot
    pack_tq_cache(kv_cache, slot, idx, signs, norm, gamma, mse_bits);
}
```

### Hook 7: Cache Read / Attention Score (Fused Decode)

**Ersetzt**: `gather_and_maybe_dequant_cache()` + FlashAttention

**Neuer Kernel: `csrc/quantization/turboquant/tq_attention_score.cu`**

Basiert auf `~/QJL/qjl_kernel/csrc/qjl_score_kernel.cu`, erweitert um Term 1 (MSE):

```cuda
// Pro Query-Token: Berechne Attention Scores gegen alle gecacheten Keys
__global__ void tq_attention_score_kernel(
    scalar_t* query,         // [batch*heads, head_dim]
    uint8_t*  kv_cache,      // gepackter TQ Cache
    float*    Pi,            // [head_dim, head_dim]
    float*    S,             // [head_dim, head_dim]
    float*    centroids,     // [2^mse_bits]
    float*    scores,        // [batch*heads, seq_len]
    ...
) {
    // Einmalig pro Query (nicht pro KV-Token!):
    // q_rotated = query @ Pi^T  → [head_dim]
    // q_projected = query @ S^T → [head_dim]
    shared float q_rot[D], q_proj[D];
    shared_gemv(PiT, query, q_rot, D);
    shared_gemv(ST, query, q_proj, D);

    // Vorberechnung: q_centroid_table[k] fuer jeden Centroid-Wert
    // q_rot[j] * centroid[k] fuer alle k → 2^mse_bits Werte pro Dimension
    shared float q_cent_table[D][N_CENTROIDS];
    for j, k: q_cent_table[j][k] = q_rot[j] * centroids[k];

    // Pro KV-Token (parallel ueber Warps):
    for each cached token t:
        unpack(kv_cache[t], &idx, &signs, &norm, &gamma);

        // Term 1: <q, Pi^T @ c[idx]> = sum_j q_rot[j] * c[idx[j]]
        //       = sum_j q_cent_table[j][idx[j]]  ← GATHER!
        float term1 = 0;
        for j: term1 += q_cent_table[j][idx[j]];

        // Term 2: QJL <S@q, sign(S@r)> — wie qjl_score_kernel
        float term2 = 0;
        for each byte b in signs:
            for shift 0..7:
                bit = (b >> shift) & 1;
                term2 += bit ? q_proj[dim] : -q_proj[dim];

        scores[t] = norm * (term1 + sqrt(pi/2)/D * gamma * term2);
}
```

**Entscheidender Performance-Trick**: `q_rot` und `q_proj` werden EINMAL pro Query
berechnet (2× GEMV mit 128×128). Danach sind alle Token-Scores O(D) per Token
(Gather + Bit-Unpack), kein GEMV mehr pro Token. Das skaliert wie FP8.

### Hook 8: Value-Aggregation

Values koennen nicht fused aus komprimiertem Format aggregiert werden
(Softmax-gewichtete Summe braucht die echten Vektoren). Zwei Optionen:

**A) Values unkomprimiert lassen** (FP16/BF16):
- Einfach, kein Qualitaetsverlust bei Values
- Cache-Shape: Keys = TQ-gepackt, Values = FP16
- Kompression nur 2.5x statt 4.9x (nur Keys komprimiert)

**B) Values MSE-quantisiert** (wie turboquant-pytorch):
- `gather_and_maybe_dequant_cache()` erweitern fuer MSE-Dequant
- Pi^T @ centroids[idx] * norm → Standard GEMV + Lookup
- Danach normales Attention-V Aggregation

**C) Values per KIVI INT4** (wie PolarQuant Mixed):
- Existierender CUDA-Kernel `cuda_bmm_fA_qB_outer` aus KIVI
- Bewährt, guter Trade-off

→ **Phase 1: Option A** (Values FP16), **Phase 2: Option B oder C**.

### Zusammenfassung der Aenderungen

```
Neue Dateien:
  vllm/v1/attention/backends/turboquant_attn.py     # Attention Backend
  vllm/v1/attention/ops/tq_kernels.py               # Python Wrapper
  csrc/quantization/turboquant/tq_reshape_cache.cu   # Quantize-on-Store
  csrc/quantization/turboquant/tq_attention_score.cu # Fused Decode Score
  vllm/turboquant/                                   # Config, Centroids, Utils

Geaenderte Dateien:
  vllm/config/cache.py              +2 Zeilen  (CacheDType: "tq3", "tq4")
  vllm/v1/attention/backend.py      +1 Zeile   (is_quantized check)
  vllm/v1/attention/selector.py     +3 Zeilen  (Backend-Dispatch)
  vllm/model_executor/layers/       +20 Zeilen (TQ Buffer Registration)
    attention/attention.py
  setup.py / CMakeLists.txt         +Kompilierung der CUDA Kernels
```

### Architektur-Diagramm

```
vllm serve model --kv-cache-dtype tq3
        │
        ▼
  arg_utils.py:394 ─── kv_cache_dtype="tq3" ──► CacheConfig
        │
        ▼
  selector.py ─── kv_cache_dtype.startswith("tq") ──► TurboQuantAttentionBackend
        │
        ▼
  attn_utils.py:93 ─── get_kv_cache_shape() ──► (blocks, 2, blk_sz, heads, 52)
        │                                         statt (blocks, 2, blk_sz, heads, 128)
        ▼
  attention.py:90 ─── set_tq_buffers() ──► Pi, S, centroids als Buffers
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │ TurboQuantImpl.forward()                            │
  │                                                     │
  │  PREFILL:                                           │
  │    flash_attn(Q, K, V)          # Standard, FP16    │
  │    tq_reshape_and_cache(K, V)   # Quantize + Store  │
  │                                                     │
  │  DECODE:                                            │
  │    tq_reshape_and_cache(k, v)   # Neuen Token       │
  │    scores = tq_fused_score(     # CUDA Fused Kernel │
  │        query, kv_cache,                             │
  │        Pi, S, centroids)                            │
  │    attn_out = softmax(scores) @ values              │
  └─────────────────────────────────────────────────────┘
```

---

## Referenz-Implementierung: PolarQuant Repo

**Repo**: `~/PolarQuant/` (geclont von `github.com/ericshwu/PolarQuant`)
**Hinweis**: Keine CUDA-Kernels, alles Pure Python + Triton. `modeling_utils_qjl.py` ist leer (QJL-Code nicht veroeffentlicht).

### Verfuegbare Komponenten

| Datei | Inhalt | Nutzbar? |
|-------|--------|----------|
| `models/kernel4group.py` | **Triton Fused-Attention Kernel** fuer PolarQuant decode | **JA — Kernreferenz** |
| `models/modeling_llama_polar.py` | Llama-Attention mit PolarQuant KV-Cache | JA — Architektur-Referenz |
| `models/modeling_llama_qjl.py` | Llama-Attention mit QJL KV-Cache | JA — QJL Integration |
| `models/modeling_utils_qjl.py` | QJLSketch / QJLKeyQuantizer | LEER (nicht veroeffentlicht) |
| `benchmark/benchmark_matmul.py` | Latenz-Vergleich: FP16 vs Polar vs KIVI | JA — Benchmark-Vorlage |
| `models/kivi_quant/` | KIVI Baseline (Triton pack + CUDA matmul) | Referenz fuer Value-Quant |

### Kernerkenntnisse aus dem Code

#### 1. PolarQuant Quantisierung (NICHT Random-Rotation wie TurboQuant!)

PolarQuant verwendet **Polar-Koordinaten**, nicht Random Rotation:
```python
# models/modeling_llama_polar.py:135-157
key_states = key_states.view(B, N, L//G, G, 2, D)  # head_dim in 2D-Paare splitten
phi = atan2(y_component, x_component)                # Winkel (theta)
radii = norm(key_states, dim=-2)                     # Radius (rho)

# Group-wise min-max Quantisierung
tscale = (phi_max - phi_min) / 2^tbits               # Winkel-Scale
rscale = (radii_max - radii_min) / 2^rbits           # Radius-Scale

indices = (rho_quant << tbits) + theta_quant          # Gepackt in uint8
```

**Wichtiger Unterschied zu TurboQuant**:
- PolarQuant: Cartesian → Polar (atan2), Group-wise min-max, 2D-Paare
- TurboQuant: Random Rotation (Pi@x), Centroid-basierte skalare Quant, ganzer Vektor

#### 2. Fused Decode-Attention Kernel (`kernel4group.py`)

Der Triton-Kernel berechnet **Q·K direkt aus komprimiertem Format** ohne Dequant:
```python
# Fuer jedes der 2^tbits moeglichen Winkel: cos/sin vorberechnen
phi = tscale * (arange(2^tbits) + 0.5) + tmn

# Query gegen alle moeglichen Winkel multiplizieren (Lookup-Table)
tp = sum(query * interleave(cos(phi), sin(phi)), axis=-1)

# Gather: Index in Lookup-Table nachschlagen → Attention Weight
attn = gather(tp, indices & (2^tbits - 1))

# Radius-Lookup und multiplizieren
radii = rscale * (arange(2^rbits) + 0.5) + rmn
attn *= gather(radii, indices >> tbits)
```

**Genialer Trick**: Statt N Vektoren zu dequantisieren, werden nur 2^tbits * 2^rbits
moegliche Werte vorberechnet und per `tl.gather` nachgeschlagen. O(2^b) statt O(N).

#### 3. Architektur-Pattern: Residual-Buffer

```
Prefill:  FlashAttention (unveraendert, FP16)
          ↓
          key_states aufteilen in:
            - key_states_quant → quantisieren → (indices, scales, mins)
            - key_states_full  → Residual-Buffer (letzte residual_length=128 Tokens, FP16)

Decode:   attn_quant = fused_kernel(query, indices, scales)    # Quantisierter Teil
          attn_full  = matmul(query, key_states_full)           # Residual-Buffer
          attn = cat(attn_quant, attn_full) / sqrt(d)

          Wenn buffer voll → quantisieren + an indices anhaengen
```

#### 4. Mixed Variant (K: Polar, V: KIVI INT4)

`LlamaPolarMixedGroupAttention`: Keys polar-quantisiert, Values per KIVI (INT4 group-wise).
Values brauchen `cuda_bmm_fA_qB_outer` (CUDA-Kernel aus KIVI fuer Attention * quantized_V).

#### 5. Benchmark-Ergebnisse (A100, aus Kommentaren)

| Methode | Seq=4K | 8K | 16K | 32K | 64K | 128K | Bits |
|---------|--------|-----|------|------|------|-------|------|
| FP16 matmul | 0.05 | 0.08 | 0.12 | 0.22 | 0.42 | 0.80 | 16 |
| Polar 4+4 | 0.11 | 0.12 | 0.15 | 0.23 | 0.40 | 0.75 | 8 |
| Polar 3+3 | 0.08 | 0.09 | 0.12 | 0.18 | 0.30 | 0.54 | 6 |
| KIVI 4-bit | 0.09 | 0.13 | 0.21 | 0.40 | 0.78 | 1.54 | 4 |
| KIVI 2-bit | 0.07 | 0.11 | 0.18 | 0.32 | 0.62 | 1.20 | 2 |

**Polar 3+3 ist schneller als FP16 ab 32K Tokens** (0.18 vs 0.22 ms) — durch Fused-Kernel!
KIVI skaliert schlecht weil kein Fused-Kernel (dequant → matmul → 2x Memory-Bandwidth).

---

## Revidierter Implementierungsplan

### Strategie-Aenderung nach Code-Review

Die PolarQuant-Referenz zeigt, dass der **Fused Decode-Attention Kernel** der entscheidende
Performance-Gewinn ist — nicht die Quantisierung selbst. Das aendert die Prioritaeten:

**Alt**: Phase 1 Gather+Dequant → Phase 2 Triton Quant/Dequant → Phase 3 Fused
**Neu**: Phase 1 Standalone-Validierung → Phase 2 Fused Triton Decode-Attention → Phase 3 vLLM-Integration

### Phase 1: Standalone TurboQuant Module (Pure PyTorch)

Isoliertes Modul, NICHT in vLLM integriert. Validiert Algorithmus-Korrektheit.

```
vllm/turboquant/
  __init__.py
  config.py           # TurboQuantConfig (bit_width, head_dim, seed)
  quantizer.py        # TurboQuantizer (quantize, dequantize, inner_product)
  centroids.py        # Max-Lloyd Centroid-Berechnung + Lookup-Tables
  test_standalone.py  # MSE bounds, unbiased IP, round-trip
```

Kern-API:
```python
tq = TurboQuantizer(head_dim=128, bit_width=3, seed=42)

# Quantize
state = tq.quantize(key_states)  # [B, N, L, D] → TQState

# Dequantize (fuer Validierung)
key_recon = tq.dequantize(state)  # → [B, N, L, D]

# Fused inner product (fuer Attention)
attn_weights = tq.attention_score(query_states, state)  # → [B, N, 1, L]
```

### Phase 2: Fused Triton Decode-Attention Kernel

Basierend auf `kernel4group.py`, aber fuer TurboQuant angepasst:

**PolarQuant Fused-Kernel Trick adaptiert fuer TurboQuant:**

PolarQuant: `gather(Q·cos/sin(phi_table), theta_idx) * gather(rho_table, rho_idx)`
TurboQuant: `Q·(Pi^T @ centroid_table[idx]) + gamma * sqrt(pi/2)/d * Q·(S^T @ qjl)`

Der erste Term laesst sich ebenfalls als Lookup vorberechnen:
```
# Vorberechnung (einmalig pro Query):
q_rotated = Q @ Pi^T                          # [N, D] @ [D, D] → [N, D]
# Fuer jeden moeglichen Centroid-Wert:
q_centroid_table[k] = q_rotated[:, j] * centroid[k]  # [N, 2^(b-1)]

# Pro KV-Vektor (Index-Lookup):
attn_mse[j] = gather(q_centroid_table, idx[j])  # Skalar pro Dimension
attn_mse = sum(attn_mse, dim=D) * norm           # Summe ueber Dimensionen

# QJL-Korrektur:
q_S = Q @ S^T                                    # [N, D] @ [D, D] → [N, D]
attn_qjl = sum(q_S * qjl, dim=D) * gamma * sqrt(pi/2)/d * norm
```

**Herausforderung vs PolarQuant**:
- PolarQuant: 2^tbits Lookup-Werte, jeweils Skalar → passt in Registers
- TurboQuant: 2^(b-1) Centroid-Werte, PLUS Q@Pi^T und Q@S^T Vorberechnung (GEMV)
- TurboQuant braucht 2 GEMV pro Query (Q@Pi^T, Q@S^T), PolarQuant braucht 0

**Loesung**: Q@Pi^T und Q@S^T sind pro Query-Token, nicht pro KV-Token.
Bei Decode (q_len=1) ist das je 1x GEMV(128x128) = vernachlaessigbar.

### Phase 3: vLLM Integration

Wie in originalem Plan Phase 1.2–1.4 + Phase 4:
- CacheDType Registration (`tq3`, `tq4`)
- Cache Layout in vLLM Block-Manager
- Attention Backend mit Fused-Kernel
- Residual-Buffer Pattern (wie PolarQuant: letzte 128 Tokens unkomprimiert)

### Phase 4: Value-Quantisierung

Keys: TurboQuant (Fused Attention)
Values: Optionen:
- a) FP16 (einfach, aber halber Speichergewinn)
- b) KIVI-style INT4 group-wise (wie `LlamaPolarMixedGroupAttention`)
- c) TurboQuant auf Values (Gather+Dequant, kein Fused-Trick moeglich)

---

## Referenz-Implementierung: turboquant-pytorch

**Repo**: `~/turboquant-pytorch/` (geclont von `github.com/tonbistudio/turboquant-pytorch`)
**Status**: Vollstaendige PyTorch-Referenz — KEIN Triton/CUDA, aber algorithmisch korrekt.

### Verfuegbare Komponenten

| Datei | Inhalt | Nutzbar? |
|-------|--------|----------|
| `turboquant.py` | `TurboQuantMSE`, `TurboQuantProd`, `TurboQuantKVCache` | **JA — Hauptreferenz** |
| `compressors.py` | `TurboQuantCompressorV2` (Asymmetric Attention), `TurboQuantCompressorMSE` | **JA — Attention-Integration** |
| `lloyd_max.py` | `LloydMaxCodebook`, `solve_lloyd_max()`, Beta-PDF + Gaussian-Approx | **JA — Centroid-Berechnung** |
| `test_turboquant.py` | Tests: MSE bounds, IP unbiasedness, needle-in-haystack, GPU bench | **JA — Validierung** |
| `validate.py` | E2E Validierung mit Qwen2.5-3B, echte Attention-Score Vergleiche | **JA — Qualitaetsmessung** |

### Algorithmus-Details aus dem Code

#### Lloyd-Max Codebook (`lloyd_max.py`)
```python
# Koordinaten-PDF nach Random Rotation (exakt):
f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)
# Gaussian-Approx (d>=64): N(0, 1/d)

# Lloyd-Max: 200 Iterationen, Initialisierung uniform in [-3.5*sigma, 3.5*sigma]
# Boundaries = Midpoints, Centroids = E[X | X in partition_i] via scipy.integrate.quad
```

#### TurboQuantProd Inner Product (`turboquant.py:165-192`)
```python
# Term 1: <y, x_mse> (MSE-Rekonstruktion)
x_mse = Pi^T @ centroids[indices]    # Dequant
term1 = sum(y * x_mse)

# Term 2: QJL-Korrektur (unverzerrter Schaetzer)
y_projected = y @ S^T                # Query projizieren
qjl_ip = sum(y_projected * qjl_signs)
term2 = residual_norm * sqrt(pi/2) / m * qjl_ip
```

#### Asymmetric Attention (`compressors.py:123-158`)
Batch-faehige Version fuer (B, H, S_q, D) × (B, H, S_k, D):
```python
# Term 1: Q @ K_mse^T — Standard GEMM
term1 = matmul(queries, k_mse.T)      # (B,H,S_q,S_k)

# Term 2: QJL Korrektur — auch GEMM
q_projected = matmul(queries, S.T)     # (B,H,S_q,D)
qjl_ip = matmul(q_projected, signs.T)  # (B,H,S_q,S_k)
term2 = sqrt(pi/2)/m * qjl_ip * r_norm
```

**Wichtig**: Beide Terme sind GEMMs, nicht GEMVs! Das skaliert besser als der
GEMV-Ansatz aus dem Paper wenn S_q > 1 (Prefill).

#### Norm-Handling
```python
# Compress: Vektor-Norm separat speichern, auf Einheitskugel normieren
vec_norms = norm(x)
x_hat = x / vec_norms
# Dann quantisieren...
# Bei Dequant: x_recon *= vec_norms
```

### Was NICHT im Code ist (muss ergaenzt werden)

1. **Bit-Packing** — Indices als uint8 gespeichert, nicht gepackt (verschwendet Speicher bei b<8)
2. **Outlier-Channel Handling** — Kein split in outlier/regular channels fuer 2.5/3.5 Bit
3. **Triton/CUDA Kernels** — Alles Pure PyTorch, keine Fused Ops
4. **Residual-Buffer** — KV-Cache quantisiert alles, kein unkomprimierter Buffer fuer letzte Tokens
5. **vLLM Integration** — Standalone, kein Paged-Attention Support

---

## Offene Fragen

1. ~~QJL-Code fehlt~~ → **Geloest**: `turboquant-pytorch` hat vollstaendige QJL-Implementierung

2. **`tl.gather` auf SM121** — Der PolarQuant-Kernel nutzt `tl.gather` extensiv.
   Triton-Support fuer SM121 ist experimentell. Testen ob `tl.gather` funktioniert.

3. **Residual-Buffer vs vLLM Paged-Attention** — PolarQuant haelt die letzten 128 Tokens
   unkomprimiert. vLLM's Paged-Attention hat feste Block-Groessen. Wie integrieren?
   → Moeglicherweise eigener Block-Typ oder immer ganze Blocks quantisieren.

4. **Prefill-Quantisierung** — PolarQuant quantisiert Keys erst nach Prefill (FlashAttn
   auf FP16). Das passt zu vLLM wo Prefill und Decode getrennt sind.

5. **MSE vs Asymmetric Attention Trade-off** — `turboquant-pytorch` speichert k_mse als
   FP16 (schnell, aber 16 Bit pro Koord. fuer Rekonstruktion). Fuer echte Kompression
   muessen wir aus Indices rekonstruieren → braucht Pi^T GEMV on-the-fly.

---

## Referenz-Implementierung: QJL (CUDA Kernels!)

**Repo**: `~/QJL/` (geclont von `github.com/amirzandieh/QJL`)
**Autoren**: Zandieh et al. — **selber Erstautor wie TurboQuant!**
**Status**: Vollstaendige CUDA-Implementierung der QJL-Komponente (Stage 2 von TurboQuant).

### CUDA Kernels (4 Stueck)

| Kernel | Datei | Funktion |
|--------|-------|----------|
| **QJL Quantize** | `qjl_kernel/csrc/qjl_quant_kernel.cu` | Key-Vektoren → sign(S@k) gepackt als uint8 |
| **QJL Score** | `qjl_kernel/csrc/qjl_score_kernel.cu` | Fused Attention-Score aus komprimierten Keys |
| **QJL GQA Score** | `qjl_kernel/csrc/qjl_gqa_score_kernel.cu` | Wie Score, aber mit GQA (multiple Q-Heads pro KV-Head) |
| **Value Quant** | `qjl_kernel/csrc/quantization.cu` | INT-Quantisierung fuer Values (KIVI-style) |

### Kernel-Architektur-Details

#### QJL Quant Kernel (`qjl_quant_kernel.cu`)

**Input**: `key_states [B, H, N_groups, group_size, emb_dim]`, `rand_prj [sketch_dim, emb_dim]`
**Output**: `key_quant [B, H, N, group_size, sketch_dim/8]` (uint8, bitgepackt)

**Algorithmus**:
1. Keys in Shared Memory laden: `shared_keys[EMB_DIM][WARP_SIZE]` (128×32 float)
2. Pro Projektionsrichtung: `sketched = sum(rand_prj[p,:] * key[:])` (Dot-Product)
3. Outlier-Channels separat behandeln (eigene Sketch + eigene Norm)
4. Sign-Bits in uint8 packen: `hashed_key = sum(bit << shift for each sign)`
5. **L2 Persistent Cache Hint** fuer rand_prj Matrix (feste Daten, haeufig gelesen)

**Performance-Tricks**:
- 32 Warps pro Block (WARPS_PER_BLOCK=32), jeder Warp verarbeitet eine Projektionsrichtung
- EMB_DIM=128 fest verdrahtet (passt perfekt fuer LLM head_dims)
- Outlier/Inlier Split: Outlier-Channels werden separat quantisiert mit eigener Sketch-Dim

#### QJL Score Kernel (`qjl_score_kernel.cu`)

**Input**: `query_sketch [BH, sketch_dim]`, `key_quant [BH, N, group_size, hash_dim]`
**Output**: `scores [BH, N*group_size, 1]`

**Algorithmus**:
1. Query-Sketch in Shared Memory: `shared_q_sketch[WARP_SIZE][8]`
2. Query-Outlier-Sketch berechnen: `q_outlier_sketch = sum(query[outlier_idx] * rand_prj[outlier_idx, :])`
3. **Fused Inner Product**: Fuer jedes Byte der gepackten Key-Quant:
   ```
   for shift in 0..7:
     bit = (key_byte >> shift) & 1
     ip += bit ? +q_sketch_val : -q_sketch_val   # sign(S@k) * (S@q)
   ```
4. Warp-Reduce sum ueber sketch_dim Chunks
5. Skalierung: `score = sqrt(pi/2)/sketch_dim * norm_k * ip + sqrt(pi/2)/outlier_sketch_dim * norm_outlier * outlier_ip`

**Entscheidend**: Der Score-Kernel berechnet `<S@q, sign(S@k)>` DIREKT aus den Bit-Packed
Signs ohne Dequantisierung. Das ist die QJL-Formel aus dem Paper.

#### GQA Score Kernel (`qjl_gqa_score_kernel.cu`)

Wie Score-Kernel, aber verarbeitet `GQA_GROUP_SIZE=4` Query-Heads pro KV-Head gleichzeitig.
Shared Memory: `shared_q_sketch[GQA_GROUP_SIZE][WARP_SIZE][8]` — 4× Queries parallel.

### Python-Wrapper (`llama3_utils_qjl.py`)

```python
class QJLSketch:
    # Random Projection Matrix: randn(sketch_dim, emb_dim)
    # Optional: QR-orthogonalisiert in Chunks fuer bessere Qualitaet
    proj_dir_quant  # Fuer Quantisierung: S^T [emb_dim, sketch_dim]
    proj_dir_score  # Fuer Score: S [sketch_dim, emb_dim] (optional rotiert)

class QJLKeyQuantizer:
    # Verwaltet den KV-Cache mit Residual-Buffer
    # build_sketch(): Prefill → quantize + outlier detection
    # update_sketch(): Decode → append to buffer, quantize when full
    # attention_score(): query_sketch + CUDA score kernel + residual matmul
```

**Outlier-Handling**:
```python
# Top-k Outlier Channels per Group (nicht global!)
norms = key_states.norm(dim=-2)          # Norm ueber Tokens pro Channel
_, outlier_indices = norms.topk(k)       # k=8 oder 16 Outlier-Channels
# Outliers separat quantisiert mit eigener (kleinerer) Sketch-Dimension
```

### Direkte Nutzbarkeit fuer TurboQuant

| QJL-Komponente | In TurboQuant | Aenderung noetig? |
|----------------|---------------|-------------------|
| `qjl_quant_kernel.cu` | Stage 2: sign(S@r) | JA — Input ist Residual r, nicht Key k |
| `qjl_score_kernel.cu` | Term 2 von TQ_prod | JA — muss mit Term 1 (MSE) kombiniert werden |
| `qjl_gqa_score_kernel.cu` | GQA Support | JA — wie oben |
| Outlier-Detection | Outlier-Channel Split | Direkt nutzbar |
| Bit-Packing | uint8 Sign-Packing | Direkt nutzbar |
| L2 Cache Hints | Performance | Direkt nutzbar |

**Was fehlt fuer TurboQuant**:
- Stage 1 (MSE): Random Rotation (Pi@x) + Centroid-Quantisierung + Pi^T@c[idx]
- Combined Score: Term1 (<q, Pi^T@c[idx]>) + Term2 (QJL auf Residual)
- Residual-Berechnung: r = x - Pi^T@c[idx] vor QJL-Quantisierung

---

## Finaler Implementierungsplan (revidiert nach 3 Repos)

### Verfuegbare Bausteine

```
turboquant-pytorch/     → Algorithmus-Referenz (PyTorch, korrekt aber langsam)
  turboquant.py           TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
  compressors.py          Asymmetric Attention (batched GEMM-basiert)
  lloyd_max.py            Centroid-Berechnung (scipy)

PolarQuant/             → Triton Fused-Decode-Attention (Polar-Koordinaten)
  models/kernel4group.py  tl.gather-basierter Fused-Kernel (PolarQuant, nicht TQ)

QJL/                    → CUDA Kernels fuer QJL (Stage 2 von TurboQuant)
  qjl_kernel/csrc/        4 CUDA Kernels (quant, score, gqa_score, value_quant)
  models/                 QJLSketch, QJLKeyQuantizer (Python Wrapper)
```

### Phase 1: Standalone Validierung (1-2 Tage)

**Ziel**: turboquant-pytorch Tests auf unserer Hardware laufen lassen.

1. `cd ~/turboquant-pytorch && python test_turboquant.py` — MSE bounds, IP unbiasedness
2. `python validate.py` auf DGX mit GLM-4.7-Flash — echte Attention-Score Korrelation
3. Baseline: FP8 KV-Cache Qualitaet messen (gleiche Prompts)

### Phase 2: CUDA-Kernel Adaption (3-4 Tage)

**Ziel**: QJL CUDA-Kernels um Stage 1 (MSE) erweitern → vollstaendiger TurboQuant CUDA-Kernel.

1. QJL-Kernels auf SM121 kompilieren (setup.py + arch flags)
2. **Neuer Kernel `tq_quant_kernel.cu`**:
   - Input: key_states, Pi, centroids, S
   - Stage 1: `y = Pi @ x_hat` → `idx = nearest(y, centroids)` → `x_mse = Pi^T @ c[idx]`
   - Stage 2: `r = x - x_mse` → sign(S @ r) (QJL, existierender Code)
   - Output: indices (gepackt) + qjl_signs (gepackt) + norms
3. **Neuer Kernel `tq_score_kernel.cu`**:
   - Term 1: `<q, Pi^T @ c[idx]>` — Centroid-Lookup + GEMV, analog zu PolarQuant gather-trick
   - Term 2: QJL Score (existierender `qjl_score_kernel.cu` Code)
   - Combined: `norm * (term1 + correction_scale * gamma * term2)`

### Phase 3: vLLM Integration (3-4 Tage)

1. CacheDType: `"tq3"`, `"tq4"` in `vllm/config/cache.py`
2. Cache-Allokation: Gepacktes Format (indices + signs + norms) in Block-Manager
3. `reshape_and_cache_tq()`: Ruft `tq_quant_kernel` auf
4. Attention Backend: Ruft `tq_score_kernel` auf (decode) oder gather+dequant (prefill)
5. Residual-Buffer: Letzte N Tokens unkomprimiert (wie QJL/PolarQuant)

### Phase 4: Benchmarks & Tuning (2 Tage)

1. Qualitaet: Needle-in-Haystack, LongBench
2. Throughput: tok/s bei verschiedenen Seq-Lengths
3. Memory: Max Batch-Size / Seq-Length Vergleich vs FP8/FP16
4. SM121-spezifisch: Shared Memory Limits, Warp-Konfiguration

---

## Offene Fragen (aktualisiert)

1. ~~QJL-Code fehlt~~ → **Geloest**: QJL Repo hat vollstaendige CUDA-Kernels!
2. **SM121 Kompilation**: QJL-Kernels nutzen `__shfl_down_sync`, `atomicAdd` — sollte auf SM121 funktionieren. `EMB_DIM=128` hardcoded passt. Testen!
3. **L2 Persistent Cache auf SM121**: QJL nutzt `cudaAccessPropertyPersisting` — SM121 Support pruefen.
4. **Residual-Buffer vs Paged-Attention**: QJL hat selbes Pattern (buffer_size=128). Integration in vLLM Paged-Attention Design klaren.
5. **Combined Kernel Complexity**: TQ braucht Pi@x GEMV + Centroid-Lookup + Pi^T@c GEMV + Residual + S@r GEMV = 3 GEMVs + Lookups pro KV-Vektor. QJL allein braucht nur 1 GEMV (S@k). Trade-off: Qualitaet (TQ >> QJL) vs Latenz.

---

## Referenzen

- [TurboQuant Paper](https://arxiv.org/abs/2504.19874) — Algorithmen 1+2, Theoreme 1-3
- [PolarQuant](https://arxiv.org/abs/2502.02617) — Verwandter Ansatz (Polar-Koordinaten)
- [PolarQuant Code](https://github.com/ericshwu/PolarQuant) — Triton Fused-Kernel Referenz
- [KIVI](https://arxiv.org/abs/2402.02750) — Bestehende KV-Cache Quantisierung (Baseline)
- [QJL Code](https://github.com/amirzandieh/QJL) — CUDA Kernels fuer Sign-Bit Quantisierung + Fused Score
- [turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) — PyTorch Referenz-Implementierung
- vLLM FP8 KV-Cache: `csrc/cache_kernels.cu`, `vllm/config/cache.py`
