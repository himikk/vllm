# RotorQuant — Clifford-Algebra KV-Cache Compression

## Übersicht

RotorQuant ist eine Reimplementierung des TurboQuant-Algorithmus (ICLR 2026) unter Verwendung von **Clifford-Algebra-Rotoren** statt dichter Rotationsmatrizen. Es komprimiert den KV-Cache auf 3-4 Bits pro Koordinate bei exakt gleicher Qualität wie TurboQuant, aber mit **10-19× schnelleren Kernels** und **44× weniger Parametern**.

## Algorithmus

### Stage 1: MSE-Quantisierung (PolarQuant)

```
Key-Vektor k ∈ R^d
  ↓ Embed in Cl(3,0): Gruppen von 3 → Grade-1 Elemente
  ↓ Rotor Sandwich: R × v × R̃ (dekorreliert Koordinaten)
  ↓ Lloyd-Max Quantisierung: 2-3 Bits/Koordinate
  ↓ Inverse Sandwich: R̃ × q × R
  ↓ Extract: Grade-1 → R^3
  = k_mse (rekonstruierter Key)
```

### Stage 2: QJL-Korrektur (1-bit Residual)

```
Residual r = k - k_mse
  ↓ Random Projection: S @ r (Gaussian JL-Matrix)
  ↓ Sign Quantisierung: sign(S @ r) → 1 Bit/Dimension
  = Unbiased Inner-Product Korrektur
```

### Attention Score Berechnung

```
<q, k̂> ≈ <q, k_mse> + ||r|| × √(π/2)/m × <S@q, sign(S@r)>
            ↑ Term 1        ↑ Term 2 (QJL Korrektur)
```

## Clifford-Algebra Cl(3,0)

Basis: `{1, e1, e2, e3, e12, e13, e23, e123}` (8 Dimensionen)

- **Grade 0** (Skalar): 1
- **Grade 1** (Vektor): e1, e2, e3 — hier werden die Daten eingebettet
- **Grade 2** (Bivektor): e12, e13, e23 — Rotationsebenen
- **Grade 3** (Pseudoskalar): e123

**Rotor**: `R = s + b12*e12 + b13*e13 + b23*e23` (4 Freiheitsgrade, normiert)

**Sandwich-Produkt**: `R × v × R̃` rotiert v orthogonal in 3D. Nur Grades 1 und 3 überleben → grade-aware Quantisierung möglich.

## Packed Cache Format

Identisch zu TurboQuant:

```
┌─────────────────┬──────────────┬───────────┐
│ MSE Indices     │ QJL Signs    │ Norms     │
│ d × bits / 8 B  │ d / 8 B      │ 4 B       │
└─────────────────┴──────────────┴───────────┘

TQ3/RQ3 (D=128): 48 + 16 + 4 = 68 Bytes/Key  (vs 128 B FP8, 256 B BF16)
TQ4/RQ4 (D=128): 64 + 16 + 4 = 84 Bytes/Key
```

## Performance

### Kernel-Benchmarks (vs PyTorch)

| Kernel | Speedup |
|--------|---------|
| Rotor Sandwich (Triton) | 80-166× |
| Full Fused Pipeline (Triton) | 128-652× |
| Rotor Sandwich (CUDA) | ~100× |
| Fused Attention Score | 1.1-1.5× |

### Inference (Qwen2.5-3B, RTX 5090)

| Kontext | FP16 | RotorQuant 3-bit | Overhead |
|---------|------|-----------------|----------|
| 2K Tokens | 8.0 tok/s | 6.9 tok/s | -14% |
| 8K Tokens | 289 MB Cache | 60 MB Cache | **4.8× Kompression** |

## Verwendung (geplant)

```bash
# RotorQuant 3-bit KV-Cache
vllm serve <model> --kv-cache-dtype rq3

# RotorQuant 4-bit KV-Cache
vllm serve <model> --kv-cache-dtype rq4

# TurboQuant bleibt verfügbar
vllm serve <model> --kv-cache-dtype tq3
```

## Quelle

- Repository: https://github.com/scrya-com/rotorquant
- Paper: TurboQuant (ICLR 2026, Zandieh, Daliri, Brand)
- Clifford-Algebra Grundlagen: Geometric Algebra for Computer Science (Dorst, Fontijne, Mann)
