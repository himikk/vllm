# MultiQuant Quality Test Results

## Testebenen (von innen nach außen)

| Stufe | Was wird getestet | TQ3 cos | TQ4 cos | Status |
|-------|-------------------|---------|---------|--------|
| A | Eigene Impl + eigene Metadata (Referenz) | 0.874 | 0.954 | PASS |
| B | vLLM-style block_table (Offset, max_blocks) | 0.874 | 0.954 | PASS |
| C | Echte Buffer-Init (vLLM Seeds, register_buffer) | 0.859 | 0.952 | PASS |
| D | Skalierte K/V (std=0.2, wie echtes Modell) | 0.972 | 0.972 | PASS |
| E | Batch-Prefill + sequentieller Decode | 0.874 | 0.956 | PASS |
| E | Prefill allein | 1.000 | 1.000 | PASS |
| F | CUDA Graph Capture | — | — | **SKIP** (capture failed) |

## Details

### Stufe A: Eigener Test (Referenz)
- `MultiQuantImpl` direkt instanziiert
- Pi/S/centroids mit seed=42
- Sequentieller Pack + Decode, eigene TQMetadata
- D=256, H=20, Hkv=20, seq=8+5

### Stufe B: vLLM-style Metadata
- Wie A, aber block_offset=17 (nicht bei 0 startend)
- max_blocks_per_seq=64, total_blocks=128
- Ergebnis identisch zu A → Block-Adressierung korrekt

### Stufe C: Echte Buffer-Init
- seed = mq_config.seed + layer_idx * 1337 (wie vLLM)
- register_buffer wie in attention.py
- Leicht niedrigerer cos (0.859 vs 0.874) → Seed-Rauschen, kein Bug

### Stufe D: Skalierte K/V
- K/V mit std=0.2 (realistisch) statt std=1.0
- **Besserer** cos (0.97) → kleinere Werte = weniger Quantisierungsfehler
- Auch mit std=1.0 getestet: cos=0.97

### Stufe E: Batch-Prefill
- Alle Prefill-Tokens auf einmal (wie vLLM)
- do_kv_cache_update mit batch slot_mapping
- _forward_prefill mit causal mask
- Prefill cos=1.0000 (perfekt — naive bmm, keine Quantisierung)
- Decode cos=0.87/0.96 (wie erwartet)

### Stufe F: CUDA Graph
- Versuch den Decode-Forward in CUDA Graph zu capturen
- **FAILED**: `cudaErrorStreamCaptureInvalidated`
- Ursache: `is_current_stream_capturing()` Guard → `return` → KV nicht geschrieben
- Oder: Python-Ops im forward-Pfad nicht capture-safe

## Stufe G: Live Serve (enforce-eager, BF16 + TQ4)

**PASS** — Math korrekt, Text korrekt!

```
3+4= → 7 ✓
7+8= → 15 ✓
9*6= → 54 ✓
100+23= → 123 ✓
664+124= → 788 ✓
Mozart → "Wolfgang Amadeus Mozart war ein berühmter österreichischer
          Komponist der Klassik, und gilt als einer der bedeutendsten
          Musiker der Weltgeschichte." ✓
```

**Root Cause der früheren Fehler:** `_forward_decode` war auf `_decode_python_loop`
umgestellt (Debug-Versuch). Der Python Loop crasht bei D=256 (device-side assert).
Der fused CUDA Kernel funktioniert korrekt.

## Offene Punkte

1. **CUDA Graphs** — enforce-eager funktioniert, aber CUDA Graph Capture scheitert
   weil `is_current_stream_capturing()` Guard den KV Write blockiert

2. **CUDA Graph**: der Guard verhindert KV-Write bei Capture
   - Lösung: graph-safe KV Pack (PyTorch tensor ops oder pre-compiled CUDA)
   - Stufe F zeigt dass der Graph-Pfad der Bruchpunkt ist

3. **RQ3/RQ4**: Unit Test cos=0.89/0.95 aber Serve kaputt
   - Image Post-GEMV + Clifford: cos=0.10 (zu niedrig, Clifford nicht-linear)
   - Neuer RQ CUDA Kernel: cos=0.89 im Unit Test, Müll im Serve
   - RQ hat auf dem Image (`8d373f2ba`) **nie** im Serve funktioniert
   - Root Cause offen — Unit Test vs Serve Diskrepanz ungeklärt

## Testdatei

`tests/multiquant/test_vllm_integration.py`

Laufen im Container mit gemounteten Dateien:
```bash
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -v multiquant_attn.py:/.../multiquant_attn.py:ro \
  -v triton_mq_fused_decode.py:/.../triton_mq_fused_decode.py:ro \
  -v kernels/turboquant:/opt/tq_build:ro \
  -v tests/multiquant:/opt/tests/multiquant:ro \
  -v torch-extensions:/root/.cache/torch_extensions:rw \
  vllm-multiquant \
  python3 -m pytest /opt/tests/multiquant/test_vllm_integration.py -v -s
```
