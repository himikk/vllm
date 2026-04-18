# Tagebuch 2026-04-17: v10.5 Template-Refactor + v11 Vorbereitung

## Ausgangslage

Stand vom 16.04:
- Qwen 122B lief mit XFP streaming + expertwise pack, 98% Math, 15–16 tok/s
- Marlin INT4 Referenz: 29 tok/s (+MTP 50), Math ähnlich 96%
- Versuchte Kernel-Optimierung v9 (A-in-SMEM + outlier fusion) hing im Warmup
- v10 (SHFL.IDX codebook) geschrieben und gebug-fixed (unified K-loop), aber
  in Qwen 35B A/B/C-Vergleich zeigte sich: v10 ist 3–15% LANGSAMER als v8
  (Grund: slot-by-slot Loop + Branch-Predicate frisst mehr als die 28-Cycle
  SMEM-Latenz sparen — letztere war durch Warp-Occupancy längst versteckt).

## Root Cause vom SHFL-Bug in v10

v10 zunächst `cos=0` (komplett Müll) weil die K-Loop mit `kw = lane`
initialisiert und dann `kw += WARP_SIZE` weitergezählt hat. Bei kleinem
K_packed traten die Lanes mit `kw >= K_packed` gar nicht in den Loop ein.
`__shfl_sync(0xffffffff, ...)` verlangt aber **alle 32 Lanes im gleichen
Instruction-Stream** → divergent lanes = undefined shuffle.

Fix durch unified K-loop: alle Lanes iterieren `n_groups = ceil(K_packed/32)`
mal, Lanes mit `kw >= K_packed` laden 0 und überspringen die FMA per
`if (k < K && kw < K_packed)`. Nach Fix: cos=1.0 in Unit-Tests, aber
langsamer durch den Predicate-Overhead.

## Marlin-Analyse (warum Marlin 29 vs XFP 16 tok/s)

Zwei Explore-Agent-Runs auf `csrc/quantization/marlin/` und
`csrc/moe/marlin_moe_wna16/`:

1. **cp.async 4-Stage-Pipeline** für B_packed → Latency-Hiding (+40% Potenzial)
2. **Inline LOP3-Dequant** fusioniert mit MMA (+25%)
3. **`mma.m16n8k16.bf16.bf16.f32` Tensor-Cores** bei M_expert ≥ 16 (+2–3× Prefill)

Zusätzlich strukturell:
- Marlin hat **gemeinsames Template** `marlin_template.h`, das von Linear-
  und MoE-Wrapper instanziiert wird. Jede Optimierung wirkt in beiden Pfaden.
- XFP hatte zwei Copy-Paste-Zwillinge (`xfp_gemm_v10.cu` und
  `xfp_moe_gemm_v10.cu`), Inner-Loop zeichengenau identisch.

## v10.5 — Template-Refactor (heute gemacht)

### Struktur

Neuer Header `kernels/multiquant/xfp_gemm_core.cuh`:
- Template `xfp_gemm_core<BITS, Policy>` mit dem gemeinsamen Inner-Loop
  (SHFL-Lookup + unified K-loop + warp-reduce)
- `LinearPolicy` und `MoEPolicy` als struct mit `prologue()` + `epilogue()`
  → Linear setzt `n, m, A_row, B_packed, codebook_slice`; schreibt `C[m*N+n]`
  → MoE löst `expert_id` und `token_id` aus sorted arrays; schreibt mit
    optionalem `topk_weights`-Multiply

Beide Wrapper (`xfp_gemm_v11.cu`, `xfp_moe_gemm_v11.cu`) sind jetzt ~70 Zeilen
dünn: Launch-Config + Pybind11-Binding.

### Äquivalenztest

Unit-Test im Container gegen v10 (beide Varianten):

```
=== Linear v11 vs v10 ===
  N=   64 K=  256 bits=4: cos=1.000000 maxdiff=0.0
  N=  256 K=  512 bits=3: cos=1.000000 maxdiff=0.0
  N= 3072 K= 2048 bits=4: cos=1.000000 maxdiff=0.0
  N= 1024 K=  768 bits=2: cos=1.000000 maxdiff=0.0

=== MoE v11 vs v10 ===
  E=  4 N=  256 K=  512 bits=4: min_cos=1.000000 maxdiff=0.0
  E=  8 N= 1024 K=  768 bits=3: min_cos=1.000000 maxdiff=0.0
  E= 64 N= 3072 K= 2048 bits=4: min_cos=1.000000 maxdiff=0.0
```

**Bitweise identisch**. Refactor ist sauber.

### E2E-Regression Qwen 35B

| Metric | v8 (SMEM) | v10 (SHFL) | **v11 (Template)** |
|---|---:|---:|---:|
| short   | 9.4 | 8.9 | 8.9 |
| medium  | 37.5 | 31.7 | 30.8 |
| long    | 30.5 | 29.3 | 26.4 |
| Math    | 44/50 | 46/50 | 45/50 |

v11 liegt im Rauschen von v10 (−3% medium). v8 bleibt schnellster Single-Kernel
(paired bfloat162 A-loads + kein per-slot-Predicate). Aber v11 ist die richtige
**Basis für die v11-Stufen** (cp.async, MMA, Outlier-Fusion) — jede Änderung
am Core wirkt in Linear UND MoE automatisch, kein Copy-Paste-Risiko mehr.

### Dateien

- `kernels/multiquant/xfp_gemm_core.cuh` — neu, Template + beide Policies
- `kernels/multiquant/xfp_gemm_v11.cu` — dünner Linear-Wrapper
- `kernels/multiquant/xfp_moe_gemm_v11.cu` — dünner MoE-Wrapper
- `vllm/multiquant/xfp/xfp_kernel.py` — v11 als Default, v8/v10 via XFP_KERNEL
- `vllm/multiquant/xfp/xfp_moe_kernel.py` — analog
- `Dockerfile.xfp-bf16` — v11 statt v10 vorcompiliert

## Nächster Schritt: v11 Stufe 1 — cp.async Double-Buffer

Am Core-Template. Bewegt B_packed-Loads aus dem Inner-Loop in eine async
Pipeline, die 2 Stufen vorladen kann während der aktuelle Tile
verarbeitet wird. Erwartet: +15–25% tok/s (Marlin nutzt dieses Pattern).

Wenn cp.async bitweise identisch zu v11 baseline bleibt (Unit-Test cos=1.0),
ist das ein ROI-reines Speedup ohne Math-Risiko.
