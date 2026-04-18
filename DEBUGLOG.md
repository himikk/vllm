# MultiQuant Live-Serve Debug Log

## Stand: 2026-04-03

### Algorithmus (PASS)

| Test | Ergebnis | Status |
|------|----------|--------|
| Unit Test 1 Layer eager | cos 0.87 (TQ3) / 0.95 (TQ4) | PASS |
| Unit Test Prefill | cos=1.000 | PASS |
| Unit Test 47 Layer Akkumulation | cos 0.997 | PASS |
| CUDA Graph Replay isoliert | cos=1.000 vs eager | PASS |
| CUDA Kernel Argumente | Pi/S eingefügt, korrekte Signatur | PASS |
| FP8 Baseline gleiches Image | 3+4=7 | PASS |

### Live-Serve Datenfluss (GEPRÜFT)

- **Attention-Output-Normen**: gesund (0.2-10.8 über 47 Layer, kein NaN)
- **KV-Cache pro Layer**: separater Tensor (verschiedene data_ptr bestätigt)
- **Slot-Mapping**: jetzt korrekt — unique Slots pro Token (16,17,18,19,20,21)
- **Block-Table im Decode**: korrekt (`bt=[[1,0,0,0]]` passt zu Slot 16+)
- **do_kv_cache_update**: wird aufgerufen mit richtigen Shapes `[6, 20, 256]`
- **Pi/S/Centroids**: pro Layer aus `_get_matrices()` gecacht, float32+contiguous

### Root Causes gefunden & gefixt

1. **`unified_kv_cache_update` liefert Null-Slots beim Prefill**
   - `compute_slot_mapping` wird erst beim Decode aufgerufen, nicht beim Prefill
   - Fix: `forward_includes_kv_cache_update=True`, KV-Write in `forward()` mit `attn_metadata.slot_mapping`
   
2. **CUDA Kernel Call hatte falsche Arg-Reihenfolge**
   - Pi und S fehlten → Kernel scheiterte immer → Triton Fallback
   - Fix: korrekte 17-Argument-Signatur (9 Tensors + 8 Scalars)

3. **Mount-Pfad falsch**
   - Image hat vLLM unter `/usr/local/lib/python3.12/dist-packages/vllm/`
   - Alter Mount ging nach `/opt/vllm/vllm/` → gemountete Dateien wurden ignoriert

4. **`block_table.int()` / `seq_lens.int()`**
   - vLLM sendet int64 (Long), CUDA Kernel erwartet int32
   - Fix: `.int()` im Kernel-Call (pre-allocated Buffers für Graph-Safety)

### ROOT CAUSE GEFUNDEN — Softmax-Schärfe × Quantisierungsfehler

**Das Problem ist NICHT die Norm-Speicherung, sondern die Interaktion von 
Attention-Schärfe mit Quantisierungsfehler.**

Echte Modell-Aktivierungen (GLM-4.7, D=256) haben:
- K/V-Normen: 30-100 pro Vektor (std ~2-5)
- Attention-Entropy: 0.1-0.5 (nahezu one-hot)

Bei scharfer Attention (entropy<0.5) trifft der TQ-Quantisierungsfehler (~5% bei TQ4)
direkt auf den einzelnen am stärksten gewichteten V-Vektor → cos bricht zusammen.

| K/V std | Entropy | TQ4 cos | Status |
|---------|---------|---------|--------|
| 0.2 | 2.08 | 0.974 | ✓ |
| 1.0 | 1.75 | 0.955 | ✓ |
| 5.0 | 0.22 | 0.688 | ✗ |
| 10.0 | 0.03 | 0.610 | ✗ |

Isoliert getestet:
- Nur Q skalieren (Softmax schärfer): cos sinkt (z.B. temp=0.01 → cos=0.86)
- Nur K/V skalieren (Normen größer): cos sinkt ebenfalls (KV=50 → cos=0.69)
- Beides zusammen: stärkster Effekt (BOTH=5 → cos=0.69)

### Erledigte Punkte

1. **Round-Trip Pack→Decode mit Offset-Slots**: PASS (cos=0.97, vorheriger cos=0.0
   war Testfehler mit falschem block_table-Objekt)
2. **Pi/S pro Layer**: Jeder Layer hat eigenen `kv_cache` (verschiedene `data_ptr`).
   Pi/S werden pro Layer aus `_get_matrices()` gecacht. Korrekt.
3. **head_size**: GLM-4.7 hat `qk_head_dim=256`, `v_head_dim=256`, `head_dim=64` (RoPE).
   Mit `VLLM_MLA_DISABLE=1` ist der effektive head_size=256. Stimmt mit D=256 überein.

4. **head_size Mapping**: vLLM gibt `head_size=packed_size` (132 für TQ4/D=256).
   `_recover_head_dim()` mapped 132 → 256. Wenn das fehlschlägt, wird D=132 benutzt
   → falsche Reshape-Dimensionen, Müll.

5. **Num-Heads Zuweisung**: GLM-4.7 hat 20 Q-Heads und 20 KV-Heads (MHA, nicht GQA).
   Unser Code setzt `num_kv_groups = num_heads // num_kv_heads`. Wenn die Zuordnung
   falsch ist (z.B. 32 Q-Heads mit 8 KV-Heads), wird die Attention falsch berechnet.

### Hypothese

Der wahrscheinlichste verbleibende Bug ist ein **Block-Table-Offset-Problem** im CUDA Kernel.
Der Kernel nutzt `block_table[q_token * max_blocks_per_seq + bi]` wobei `q_token` der
Batch-Index (0 bei Single-Request) und `bi = pos / block_size` der logische Block ist.
Wenn `max_blocks_per_seq` nicht mit `block_table.shape[1]` übereinstimmt, liest der Kernel
falsche Blöcke.

### Testdateien

- `tests/multiquant/test_vllm_integration.py` — Stufe A-F Tests
- `/tmp/test_multilayer_accumulation.py` — 47 Layer Akkumulation
- `/tmp/test_pack_roundtrip_live.py` — Pack→Cache→Decode Roundtrip
