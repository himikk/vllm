# TASK: Archer — MultiQuant Online Weight Quantization

## Ziel

BF16/FP8 Modelle laden und beim Laden auf 2-4 Bit quantisieren (RotorQuant/TurboQuant). Bei Inferenz dekomprimieren (als BF16 oder FP8, umschaltbar wie Marlin W4A8/W4A16). Kein Pre-Quantisiertes Modell nötig.

**Archer** = Fused Decompress+GEMM Kernel (Schützenfisch, Toxotes).

## Vision

```
HuggingFace BF16/FP8 Modell
  ↓ vllm serve --quantization multiquant --quant-bits 3
  ↓ process_weights_after_loading()
  ↓ Pro Gewichtszeile: Rotation + Lloyd-Max → 3-bit packed uint8
  ↓
┌──────────────────────────────────────┐
│  VRAM: 3-bit Gewichte (3.6× kleiner)│
│  + Rotoren/Pi + Centroids + Scales   │
│                                      │
│  Forward: Archer Kernel              │
│    decompress(W_packed) → BF16/FP8   │
│    → GEMM(x, W_decompressed)         │
│    (fused in Phase 2)                │
└──────────────────────────────────────┘
```

## Status

### Phase 1: Python-Prototyp
- [ ] A1: `MultiQuantWeightConfig(QuantizationConfig)` — CLI integration
- [ ] A2: `MultiQuantOnlineLinearMethod(LinearMethodBase)` — `uses_meta_device=True`
- [ ] A3: `process_weights_after_loading()` — BF16 → MQ packed
- [ ] A4: `apply()` — decompress + torch.nn.functional.linear (Python, langsam)
- [ ] A5: MoE-Variante: `MultiQuantOnlineMoEMethod`
- [ ] A6: Unit Tests: compress → decompress → MSE
- [ ] A7: Integration Test: `--quantization multiquant --quant-bits 3` startet

### Phase 2: Archer Kernel (CUDA)
- [ ] A8: `archer_gemm.cu` — Fused decompress+GEMM (W_packed × x → y)
- [ ] A9: `archer_moe.cu` — MoE Expert-GEMM Variante
- [ ] A10: Python bindings + Dispatch
- [ ] A11: Benchmark: Archer vs Marlin vs FP8

### Phase 3: Optimierung
- [ ] A12: FP8 Activation-Modus (W3A8, wie Marlin W4A8)
- [ ] A13: MTP Drafter-Gewichte quantisieren
- [ ] A14: Streaming-Quantisierung (layer-by-layer, Peak RAM < 2× Layer)

## Architektur

### Input-Formate

| Modell-Format | Pfad |
|---------------|------|
| BF16 | Direkt quantisieren |
| FP8 | Dequant zu BF16/FP32 → quantisieren |
| INT4 (GPTQ/AWQ) | Dequant → quantisieren (falls Archer besser) |

### Activation-Precision (umschaltbar)

| Modus | Gewichte | Activation | Compute | Analogie |
|-------|----------|------------|---------|----------|
| W3A16 | 3-bit MQ | BF16 | BF16 GEMM | Marlin W4A16 |
| W3A8  | 3-bit MQ | FP8  | FP8 GEMM  | Marlin W4A8 |
| W2A16 | 2-bit MQ | BF16 | BF16 GEMM | (neu, sehr aggressiv) |
| W4A16 | 4-bit MQ | BF16 | BF16 GEMM | Marlin W4A16 |

### Bestehendes Pattern: FP8 Online

```python
# vllm/model_executor/layers/quantization/fp8.py
class Fp8OnlineLinearMethod(LinearMethodBase):
    uses_meta_device = True

    def create_weights(self, layer, ...):
        weight = ModelWeightParameter(data=torch.empty(..., device="meta"))

    def process_weights_after_loading(self, layer):
        qweight, scale = ops.scaled_fp8_quant(layer.weight)
        replace_parameter(layer, "weight", qweight)

    def apply(self, layer, x, bias):
        return ops.fp8_gemm(x, layer.weight, layer.weight_scale)
```

### Neues Pattern: MultiQuant Online

```python
# vllm/multiquant/weight_quant/online_linear.py
class MultiQuantOnlineLinearMethod(LinearMethodBase):
    uses_meta_device = True

    def create_weights(self, layer, input_size, output_size, ...):
        weight = ModelWeightParameter(
            data=torch.empty(output_size, input_size, device="meta"))
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer):
        W = layer.weight.data.float()  # (out, in)

        # Pro Zeile: Rotation (Pi oder Rotor) + Lloyd-Max → packed uint8
        packed_W, row_norms, rotation_buf = self._compress(W)

        replace_parameter(layer, "weight", packed_W)       # (out, packed_in) uint8
        layer.register_buffer("weight_scales", row_norms)  # (out,) float16
        layer.register_buffer("_rotation", rotation_buf)   # Pi (in,in) oder Rotors (in/3, 8)
        layer.register_buffer("_centroids", centroids)     # (n_levels,)

    def apply(self, layer, x, bias=None):
        # Phase 1: Python decompress + standard GEMM
        W_decompressed = self._decompress(layer)  # (out, in) BF16
        return F.linear(x, W_decompressed, bias)

        # Phase 2: Archer fused kernel
        # return ops.archer_gemm(x, layer.weight, layer.weight_scales,
        #                        layer._rotation, layer._centroids, self.bits)
```

### Archer Kernel Design

```
Input:  x (M, K) BF16/FP8
        W_packed (N, packed_K) uint8
        scales (N,) float16
        rotation: Pi (K, K) oder Rotors (K/3, 8)
        centroids (n_levels,) float32

Output: y (M, N) BF16

Algorithm (per output tile):
  1. Load block of W_packed for rows [n..n+TN]
  2. For each packed byte:
     - Extract MSE indices (2-3 bits each)
     - Lookup centroids → quantized values
  3. Inverse rotation (Pi^T @ values oder R̃ sandwich)
  4. Scale by row_norms
  5. GEMM: accumulate x @ W_decompressed^T
```

## CLI

```bash
# BF16 Modell → 3-bit RotorQuant Gewichte, BF16 Activation
vllm serve <bf16-model> \
    --quantization multiquant \
    --quant-bits 3 \
    --quant-method rq

# FP8 Modell → 3-bit, FP8 Activation (W3A8)
vllm serve <fp8-model> \
    --quantization multiquant \
    --quant-bits 3 \
    --quant-act-dtype fp8

# Kombination: Gewichte + KV-Cache + RIY Pruning
vllm serve <model> \
    --quantization multiquant \
    --quant-bits 3 \
    --kv-cache-dtype rq3 \
    --riy-expert-profile profile.json
```

## Dateien

### Phase 1

| Datei | Aktion |
|-------|--------|
| `vllm/multiquant/weight_quant/__init__.py` | NEU |
| `vllm/multiquant/weight_quant/config.py` | NEU: MultiQuantWeightConfig |
| `vllm/multiquant/weight_quant/online_linear.py` | NEU: MultiQuantOnlineLinearMethod |
| `vllm/multiquant/weight_quant/online_moe.py` | NEU: MultiQuantOnlineMoEMethod |
| `vllm/model_executor/layers/quantization/__init__.py` | EDIT: Register "multiquant" |
| `tests/multiquant/test_weight_quant.py` | NEU: Unit Tests |

### Phase 2

| Datei | Aktion |
|-------|--------|
| `csrc/quantization/archer/archer_gemm.cu` | NEU: Fused decompress+GEMM |
| `csrc/quantization/archer/archer_moe.cu` | NEU: MoE Expert-GEMM |
| `vllm/multiquant/weight_quant/archer_ops.py` | NEU: Python bindings |

## Speicher-Rechnung (GLM-4.7-Flash, 31B MoE)

| Format | Gewichte | KV-Cache (10G) | Total |
|--------|----------|----------------|-------|
| BF16   | ~62 GB   | 10 GB          | 72 GB |
| FP8    | ~31 GB   | 10 GB          | 41 GB |
| INT4   | ~16 GB   | 10 GB          | 26 GB |
| W3A16  | ~12 GB   | 10 GB (fp8)    | 22 GB |
| W3A16+RQ3 | ~12 GB | 4 GB (rq3)   | 16 GB |
| W2A16+RQ2 | ~8 GB  | 3 GB (rq2)   | 11 GB |

W2+RQ2: **11 GB** für ein 31B MoE Modell — passt in jede GPU mit 12 GB!

## Nomenklatur

**Archer** (Schützenfisch, Toxotes jaculator):
- Trifft Ziele mit einem gezielten Wasserstrahl aus der Distanz
- Analog: trifft Gewichte mit minimalen Bits aus komprimiertem Storage
- Fischnamen-Konvention (wie Marlin = Schwertfisch)
