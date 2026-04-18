# Archer — MultiQuant Online Weight Quantization

## Was ist Archer?

Archer quantisiert Modellgewichte **beim Laden** von BF16/FP8 auf 2-4 Bits mittels RotorQuant oder TurboQuant. Die komprimierten Gewichte werden bei der Inferenz on-the-fly dekomprimiert. Kein pre-quantisiertes Modell nötig.

**Name**: Schützenfisch (Toxotes) — trifft Ziele mit minimalen Bits, Fischnamen-Konvention wie Marlin.

## Verwendung

```bash
# BF16 Modell → 3-bit RotorQuant Gewichte
vllm serve <model> --quantization multiquant

# Explizit: Bits und Methode
vllm serve <model> --quantization multiquant --quant-bits 3 --quant-method rq

# Kombination mit KV-Cache Kompression + RIY Pruning
vllm serve <model> \
    --quantization multiquant \
    --kv-cache-dtype rq3 \
    --riy-expert-profile profile.json
```

## Architektur

```
BF16 Weight (out × in)
  ↓ process_weights_after_loading()
  ↓
  Pro Zeile:
    1. Normalisieren: w_hat = w / ||w||
    2. Rotation: R × w_hat × R̃ (Rotor) oder Π @ w_hat (Dense)
    3. Lloyd-Max: 2-3 Bits/Koordinate → Indices
    4. QJL Residual: sign(S @ residual) → 1 Bit Korrektur
    5. Rekonstruktion: w_recon = ||w|| × (w_mse + w_qjl)
  ↓
  BF16 rekonstruiert (Phase 1) oder uint8 gepackt (Phase 2)
  ↓
  F.linear(x, W_recon) bei Inferenz
```

## Phasen

### Phase 1: Python-Prototyp (aktuell)

- `process_weights_after_loading()` quantisiert und rekonstruiert sofort zu BF16
- `apply()` nutzt Standard `F.linear` — kein Custom Kernel
- **Korrekte Ergebnisse**, aber noch keine VRAM-Einsparung (BF16 bleibt im RAM)
- Nützlich für Qualitäts-Validierung

### Phase 2: Archer Kernel (geplant)

- Gewichte bleiben als packed uint8 im VRAM (echte Kompression)
- `archer_gemm.cu`: Fused Decompress+GEMM Kernel
- Analog zu Marlin: packe N-bit Gewichte, decompress on-the-fly im Kernel
- **Echte VRAM-Einsparung**: W3 = 3× kleiner als BF16

### Phase 3: Optimierung (Zukunft)

- W3A8 Modus: FP8 Activation + 3-bit Gewichte (wie Marlin W4A8)
- MoE Expert-GEMM Variante
- MTP Drafter-Gewichte quantisieren

## Speicher-Rechnung

| Modell | BF16 | FP8 | INT4 | W3 (Archer) | W2 (Archer) |
|--------|------|-----|------|-------------|-------------|
| GLM-4.7 (31B) | 62 GB | 31 GB | 16 GB | 12 GB | 8 GB |
| Qwen3.5 (35B) | 70 GB | 35 GB | 18 GB | 13 GB | 9 GB |

Mit KV-Cache-Kompression (RQ3):

| Setup | Gewichte | KV-Cache | Total |
|-------|----------|----------|-------|
| FP8 + FP8 KV | 31 GB | 10 GB | 41 GB |
| W3 + RQ3 KV | 12 GB | 4 GB | 16 GB |
| W2 + RQ2 KV | 8 GB | 3 GB | 11 GB |

**W2+RQ2: 11 GB für ein 31B MoE Modell** — passt in jede 12 GB GPU!

## Activation-Precision (umschaltbar)

| Modus | Gewichte | Activation | Analogie |
|-------|----------|------------|----------|
| W3A16 | 3-bit MQ | BF16 | Marlin W4A16 |
| W3A8  | 3-bit MQ | FP8  | Marlin W4A8 |
| W2A16 | 2-bit MQ | BF16 | (neu) |
| W4A16 | 4-bit MQ | BF16 | Marlin W4A16 |

## Dateien

```
vllm/multiquant/weight_quant/
├── __init__.py           # Package
├── config.py             # ArcherConfig(QuantizationConfig)
├── online_linear.py      # ArcherOnlineLinearMethod (Phase 1)
└── (Phase 2)
    ├── archer_ops.py     # Python bindings
    └── csrc/archer_gemm.cu  # Fused kernel
```
