# TASK.tq_ext.md — TurboQuant v2: WHT Block-Compression

## Ausgangslage

### Was funktioniert
- Algorithmus mathematisch korrekt (Unit Test cos 0.87-0.95)
- 47-Layer Akkumulation cos=0.997 mit random std=0.2
- CUDA Graph Replay cos=1.000
- CUDA Kernel kompiliert und läuft
- Slot-Mapping Fix (`forward_includes_kv_cache_update=True`)
- FP8 Baseline auf gleichem Image: korrekt

### Was nicht funktioniert
- **Live-Serve: Müll-Output** trotz korrekter Integration (Slots, Cache-Pointer, Block-Table)
- Root Cause: **Quantisierungsfehler × scharfe Softmax** bei realen Aktivierungen
  - K/V-Normen >50 + Attention-Entropy <0.5 → cos bricht auf 0.69 ein
  - FP16-Normen sind NICHT schuld (0.02% Fehler)
  - Problem: D×D Orthogonale Rotation erhält die Input-Verteilung → Centroids nicht optimal

### Referenz: github.com/animehacker/llama-turboquant
Funktionierender llama.cpp Port des TurboQuant Papers. Fundamentaler Unterschied:

| | Unsere Impl (kaputt) | Referenz (funktioniert) |
|---|---|---|
| Transform | D×D Random Orthogonal (per Layer) | **WHT auf 32-Element-Blöcke** (deterministisch) |
| Normierung | L2-Norm des ganzen Vektors | **amax pro 32er-Block** |
| Centroids | Lloyd-Max für N(0,1/D) | Lloyd-Max für **N(0,1)**: ±2.16, ±1.33, ±0.74, ±0.24 |
| Packed (D=256, 3bit) | 132 bytes | **112 bytes** (kleiner UND besser) |
| State pro Layer | Pi[D,D] + S[D,D] | **Nichts** (WHT deterministisch) |
| V-Dekompression | D×D GEMV (D² FLOPs) | **WHT inverse (D·5 FLOPs)** |
| Perplexity | Müll im Live-Serve | **~5% Degradation** (Paper) |

**Warum WHT funktioniert**: Walsh-Hadamard auf 32 Elementen → Central Limit Theorem → near-Gaussian → Lloyd-Max Centroids optimal. amax-Normierung → alle Werte im Centroid-Bereich.

## Implementierungsplan

### Phase 1: WHT Core + Centroids

**Neue Datei: `vllm/multiquant/shared/wht.py`**
- `wht32_forward(x)`: Sign-Flips + 5 Butterfly-Stages + 1/√32
- `wht32_inverse(x)`: = `wht32_forward` (selbst-invers mit Normierung)
- `TQ3_SIGNS[32]`: festes Sign-Pattern aus Referenz
- Pure PyTorch, CUDA-unabhängig

**Modify: `vllm/multiquant/shared/centroids.py`**
- `WHT_CENTROIDS_3BIT = [-2.1573, -1.3336, -0.7434, -0.2428, +0.2428, +0.7434, +1.3336, +2.1573]`
- `WHT_THRESHOLDS_3BIT = [-1.7455, -1.0385, -0.4906, 0.0, 0.4906, 1.0385, 1.7455]`
- `get_wht_centroids(bits)` Funktion

**Test**: `wht_inverse(wht_forward(x)) ≈ x`, KS-Test auf Gaussianität

### Phase 2: Config + Registry

**Modify: `vllm/multiquant/turboquant/config.py`**
- `TurboQuantWHTConfig(block_size=32)` — konfigurierbare Blockgröße
- `packed_size = (D // block_size) * bytes_per_block`
  - 3-bit: `bytes_per_block = 14` (8 qs + 4 qr + 2 gamma)
  - D=256: `packed_size = 8 * 14 = 112`

**Modify: `vllm/multiquant/registry.py`**
- `tq3w` / `tq4w` als neue dtype-Strings registrieren
- Mapping zu `TurboQuantWHTConfig`

### Phase 3: Pack/Unpack

**Modify: `vllm/multiquant/shared/bitpack.py`**

Neues Block-Format (14 Bytes pro 32 Werte):
```
[qs: 8 bytes (lower 2 bits, 4/byte)]
[qr: 4 bytes (upper 1 bit, 8/byte)]  
[gamma: 2 bytes (fp16 scale)]
```

Pack-Pipeline (vektorisiert):
1. `blocks = x.reshape(N, D//32, 32)`
2. `rotated = wht32_forward(blocks)`
3. `amax = rotated.abs().amax(dim=-1)`
4. `gamma = amax / 2.1573`
5. `normalized = rotated / gamma.unsqueeze(-1)`
6. `idx = threshold_quantize(normalized)` (keine argmin, sondern Schwellwert-Vergleich)
7. Bitpack → `qs`, `qr`, `gamma.to(fp16)`

**Entscheidender Test: Cos-vs-Std Sweep**
- Muss `cos > 0.90` für std=0.2 bis 50.0 zeigen (aktuell 0.69 bei std=5.0)

### Phase 4: CUDA Decode Kernel

**Neue Datei: `kernels/turboquant/tq_wht_decode.cu`**

Grid: `(num_q * num_q_heads)`, Block: `(HEAD_DIM)` Threads

Pro cached Token:
1. **K-Score**: Warp (32 Threads) = 1 WHT-Block
   - Lade 14 packed Bytes, unpack idx + gamma
   - `score += q_wht[t] * gamma * centroid[idx[t]]` (im WHT-Raum)
   - Warp-Reduce-Sum → Block-Score

2. **Online Softmax**: wie bisher

3. **V-Reconstruct**: Warp = 1 V-Block
   - `v_wht[t] = gamma * centroid[idx[t]]`
   - WHT-Inverse via `__shfl_xor_sync` (5 Stages, KEIN Shared Memory!)
   - `v_acc[t] += weight * v_recon[t]`

Performance: 8 Warps × 5 Shuffles = 40 Ops vs D²=65536 FMA im alten Kernel.

### Phase 5: Backend-Integration

**Modify: `vllm/v1/attention/backends/multiquant_attn.py`**

`__init__`:
- WHT-Mode: keine Pi/S Matrizen, nur 8 Centroids
- Spart 2×D²×4 Bytes × 47 Layer ≈ 25 MB (D=256)

`_pack_batch` / `_torch_pack`:
- WHT-Pfad: `wht32_forward` → `amax` → quantize → block-pack
- Alter Pfad bleibt für `tq3`/`tq4`

`_forward_decode`:
- WHT-Pfad: `q_wht = wht32_forward(q)`, WHT-CUDA-Kernel
- Kein Post-GEMV nötig (WHT-Inverse ist im Kernel)

**Modify: `vllm/model_executor/layers/attention/attention.py`**
- WHT-Mode: kein `_tq_Pi`/`_tq_S` registrieren
- Nur `_tq_centroids` (8 Werte) + `_tq_use_wht=True` Flag

### Phase 6: Tests

| # | Test | Kriterium |
|---|------|-----------|
| 1 | WHT Round-Trip | `wht(wht(x)) ≈ x` (< 1e-5 Fehler) |
| 2 | WHT Gaussianisierung | KS-Test p > 0.05 für verschiedene Input-Verteilungen |
| 3 | Pack/Unpack Round-Trip | `unpack(pack(data)) == data` (bit-exakt) |
| 4 | **Cos vs Std Sweep** | **cos > 0.90 für std 0.2-50.0** (aktuell 0.69@std=5) |
| 5 | 47-Layer Akkumulation | cos > 0.99 mit realistischen std |
| 6 | CUDA vs Python | cos=1.0 zwischen CUDA und Python Decode |
| 7 | Live-Serve | `3+4=7`, Mozart, 664+124=788 |

## Reihenfolge

```
Phase 1+2 (WHT + Config)     → Test 1,2
Phase 3   (Pack)              → Test 3,4 (DER ENTSCHEIDENDE TEST)
Phase 4   (CUDA Kernel)       → Test 6
Phase 5   (Integration)       → Test 5,7
```

## Risiken

- **D % 32 ≠ 0**: GLM-4.7 hat D=256 (ok). Für andere: Zero-Padding des letzten Blocks.
- **Performance**: WHT-Kernel SCHNELLER als D×D GEMV. Triton-Fallback als Safety Net.
- **Cache-Inkompatibilität**: `tq3w` ≠ `tq3`. Separate dtype-Strings verhindern Verwechslung.
- **Backward-Compat**: `tq3`/`tq4` bleiben für alte Rotation-basierte Variante.

## Referenzen

- Paper: TurboQuant (Google, 2024)
- Referenz-Impl: https://github.com/animehacker/llama-turboquant
- Debug-Log: `DEBUGLOG.md` (Datenfluss-Analyse, Slot-Mapping Fix, Std-Sweep Ergebnisse)
