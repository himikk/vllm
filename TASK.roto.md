# TASK: RotorQuant Integration

## Ziel

RotorQuant (scrya-com/rotorquant) als zweiten KV-Cache-Quantizer neben TurboQuant integrieren. Gleicher Algorithmus (MSE + QJL), aber mit Clifford-Rotoren statt Dense-Matrizen — 44× weniger Parameter, 10-19× schnellere Kernels.

## Quelle

- GitHub: https://github.com/scrya-com/rotorquant
- Paper: TurboQuant (ICLR 2026, Zandieh et al.)
- RotorQuant = Clifford-Algebra-Reimplementierung des TurboQuant-Algorithmus

## Status

- [ ] R1: RotorQuant Python-Module portieren
- [ ] R2: Clifford-Algebra (Cl(3,0)) integrieren
- [ ] R3: Triton Fused Kernels portieren
- [ ] R4: CUDA Kernels portieren (rotor_fused_kernel.cu)
- [ ] R5: QJL Kernels (shared mit TQ oder eigene)
- [ ] R6: Unit Tests (Pack/Unpack, Compressed Score, Forward)
- [ ] R7: Integration in MultiQuant Registry
- [ ] R8: Serving-Test (GLM-4.7 + Qwen3.5)
- [ ] R9: Benchmark (RQ3 vs TQ3 vs FP8)

## RotorQuant vs TurboQuant

| | TurboQuant | RotorQuant |
|---|---|---|
| Rotation | Dense d×d Orthogonalmatrix | Clifford Rotor (8-D Multivektor) |
| Parameter | 16384 (für D=128) | ~380 (44× weniger) |
| Kernel | tiled GEMV, 33µs | Sparse GeomProd, 1.7-3.3µs |
| Speedup | Baseline | **10-19×** schneller |
| QJL (Stage 2) | identisch | identisch |
| Packed Format | MSE indices + QJL signs + norms | **identisch** |
| Qualität | Referenz | **exakt gleich** |
| KV-Cache Savings | 2.3× vs FP8 | 2.3× vs FP8 (gleich) |

## Kernkonzept: Clifford-Rotoren

Statt einer dichten d×d Rotationsmatrix (Pi) nutzt RotorQuant Cl(3,0)-Rotoren:
- Vektoren werden in Gruppen von 3 geteilt: `n_groups = d // 3`
- Jede Gruppe wird als Grade-1 Element in Cl(3,0) eingebettet: `v = v1*e1 + v2*e2 + v3*e3`
- Rotor `R = cos(θ/2) + sin(θ/2)*B̂` (8-D Multivektor)
- Sandwich-Produkt `R × v × R̃` = orthogonale Transformation
- Nur Grades 1 und 3 überleben → grade-aware Bit-Allokation

## Zu portierende Dateien

| Quelle (scrya-com/rotorquant) | Ziel (vllm/multiquant/rotorquant/) |
|---|---|
| `turboquant/rotorquant.py` | `quantizer.py` — RotorQuantMSE, RotorQuantProd |
| `turboquant/clifford.py` | `clifford.py` — Cl(3,0) Algebra |
| `turboquant/triton_kernels.py` | `kernels.py` — Triton Fused Kernels |
| `turboquant/cuda_backend.py` | `cuda_backend.py` — CUDA Wrapper |
| `turboquant/lloyd_max.py` | (shared) — bereits in multiquant/turboquant/centroids.py |
| `turboquant/csrc/rotor_fused_kernel.cu` | `csrc/rotor_fused_kernel.cu` |
| `turboquant/csrc/qjl_*.cu` | (shared mit TQ oder eigene Kopie) |

## Triton Kernels (Priorität)

| Kernel | Funktion | Speedup vs PyTorch |
|--------|----------|-------------------|
| `triton_rotor_sandwich` | Forward Rotor Sandwich | 80-166× |
| `triton_rotor_full_fused` | Full Pipeline (embed→sandwich→quant→inverse→extract) | 128-652× |
| `triton_rotor_inverse_sandwich` | Inverse für Dequantisierung | ~80× |
| `triton_fused_attention` | Q@K^T auf compressed Keys | 1.1-1.5× |

## Abhängigkeiten

- scipy (Lloyd-Max, bereits für TQ installiert)
- triton (für Triton Kernels, bereits in vLLM-Image)
- Keine neuen externen Dependencies
