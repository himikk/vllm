# Tagebuch 2026-04-13: Fused XFP MoE Kernel

## Fortführung von gestern

### Memory-Fix: BF16 sofort freigeben
- `_batched_pack_and_repack`: `del W_stack`, `del W_flat`, etc. nach Benutzung
- `layer.w13_weight.data = torch.empty(0)` nach Pack
- Ergebnis: 78 GB frei während Packing (vorher System-Hang bei Layer 44)

### Lloyd-Iterationen: 20 → 5 für MoE
- MoE-Experts homogen → 5 Iterationen reichen
- Parametrisierbar via `XFP_MOE_LLOYD_ITERS` Env-Var
- Packing-Zeit: 15 Min → 5 Min (4× schneller)

### Fused MoE Kernel Tests
- Single Expert: cos=1.0 PASS
- Multi-Expert (4 Experts, 2 topk): cos=1.0 PASS  
- Full MoE Pfad (gate_up → SiLU → down → reduce): cos=1.0 PASS
- Bug gefunden und gefixt: Down-GEMM muss top_k=1 + identity_ids nutzen
  (activated ist pro sorted-Entry, nicht pro Original-Token)

### E2E Problem
- Server startet, aber Modell gibt Müll aus (0% Math)
- Kernel isoliert korrekt — Problem ist in der vLLM-Integration
- Debug-Logging eingebaut, nächster Server-Start läuft

## E2E Ergebnisse

### Bug gefunden: Kernel schreibt C[token_id] in Original-topk-Order
Der Kernel schreibt `C[token_id]` wobei token_id aus sorted_token_ids kommt.
Das platziert Ergebnisse an der ORIGINALEN topk-Position, nicht der sortierten.
Python-Seite las es als sorted order → falsche Expert-Zuordnung.

Fix: gate_up/down als [B*topk, N] allokieren, Down-GEMM mit topk_ids direkt
als expert_ids, Scatter-Reduce über [0..BT) mit topk_weights.

### Fused MoE XFP E2E Benchmark (enforce-eager)

| Config | tok/s (long) | Math |
|--------|-------------|------|
| XFP4 attn+shared only, BF16 MoE (eager) | 24.6 | 66% |
| XFP4 attn+shared only, BF16 MoE (CUDA Graphs) | 32.5 | 66% |
| **XFP4 ALL mit fused MoE Kernel (eager)** | **29.5** | **56%** |
| XFP4 ALL mit fused MoE Kernel (CUDA Graphs) | CRASH | - |

### CUDA Graphs Crash
`cudaErrorStreamCaptureUnsupported` — moe_align_block_size oder Python-seitige
Tensor-Allokationen im apply() nicht Graph-kompatibel. Braucht custom op Wrapper
wie bei xfp_apply/xfp_outlier_scatter.

### CUDA Graphs Fix: torch.argsort statt moe_align_block_size
`moe_align_block_size` (C++ op) nicht Graph-capture-kompatibel.
Ersetzt durch `topk_ids.reshape(-1).argsort(stable=True)` — pure torch, Graph-safe.
Plus custom op Wrapper `xfp_moe_forward` für torch.compile Boundary.

### Ergebnis mit CUDA Graphs: 49.6 tok/s!

| Config | tok/s (long) | Math |
|--------|-------------|------|
| XFP4 attn+shared, BF16 MoE, graphs | 32.5 | 66% |
| XFP4 ALL, fused MoE, eager | 29.5 | 56% |
| **XFP4 ALL, fused MoE, CUDA Graphs** | **49.6** | **56%** |
| Marlin INT4 Referenz | 55.6 | ~78% |

Tag: `xfp_fast`

### Profiling bei 49.6 tok/s

Per-Token Budget (20.2 ms):
- Fused MoE gate_up: 3.05 ms (46 Layers)
- Fused MoE down: 1.89 ms
- Attn+shared XFP: 6.53 ms (einzelne Kernel-Calls)
- **XFP Kernel total: 11.47 ms (57%)**
- Rest (attn compute, norm, routing): 8.73 ms

Fused MoE Speedup: 130-440× vs Python-Loop!
Gap zu Marlin: nur noch 2.2 ms (11%)

### XFP auto alle Klassen

`--weight-dtype xfp` setzt jetzt ALLE Klassen (inkl LM Head, MTP, Dense).
Auto-Select wählt pro Layer die niedrigste Bitbreite die cos > 0.98 schafft.

GLM-4.7-Flash Ergebnis:
- 327 Linear-Layer: 325× xfp3, 2× xfp4 (attn_qb)
- 46 MoE-Layer: alle xfp3
- Outliers: 0.02–0.94% pro Layer

| Config | tok/s | Math |
|--------|-------|------|
| XFP4 all | 49.6 | 56% |
| XFP auto (routed+attn+shared) | 51.1 | 54% |
| XFP auto (ALL classes) | **52.6** | 46% |
| Marlin INT4 | 55.6 | ~78% |

52.6 tok/s = 95% von Marlin! Math fällt bei xfp3 auf Dense+LM Head.
→ LM Head und Dense MLP sollten xfp4 erzwingen oder höheren cos-Schwellwert.

Optimierungspotential:
1. Attention-Kernel batchen (7 Calls/Layer → 1-2 fused)
2. Outlier scatter eliminieren (1.6 ms)
3. Kernel SMEM-Prefetch / Tile-Tuning
4. LM Head cos-Schwellwert erhöhen (0.995 statt 0.98)
5. Embedding/LM Head FP8: `--weight-dtype-lm-head fp8` → weight.to(float8_e4m3fn),
   halber Speicher, FP8 GEMM im Sampler. Kein XFP, nur dtype-Cast in MultiQuant.
   ParallelLMHead.forward() wird nie aufgerufen — Weights direkt im Sampler als GEMM.
   Eigener kleiner Punkt, nicht XFP.

## FP8 LM Head implementiert

`--weight-dtype-lm-head fp8` → FP8EmbeddingMethod für VocabParallelEmbedding.
- process_weights: bf16 → float8_e4m3fn + scale (154880×2048, spart 317 MB)
- apply: torch._scaled_mm für FP8×FP8 → BF16 GEMM (kein Dequant nötig)
- embedding: F.embedding in fp8, dequant output
- Naive dequant war 33.7 tok/s (Regression!), cached bf16 = 52.6 tok/s (neutral)
- _scaled_mm Variante wird getestet

Bonusfrage: XFP auf LM Head möglich? Ja — ist ein GEMM, xfp_gemm Kernel funktioniert.
Aber VocabParallelEmbedding ≠ LinearBase → braucht eigenen Pfad in create_weight_method.
xfp3 auf LM Head = 100 MB statt 600 MB. Auto-select müsste Qualität prüfen.

## Qwen3.5-122B-A10B Tests

### INT4 AutoRound Marlin: 25.8 tok/s, 96% Math
Exzellente Math-Qualität. Nur 2 Fehler (große Multiplikation, negative Subtraktion).

### XFP quant-on-load: OOM!
250 GB BF16 → 120 GB Unified Memory → System-Hang.
Grund: vLLM lädt alle Safetensors (BF16) bevor process_weights_after_loading läuft.
Auch FP8LinearMethod/FP8MoEMethod helfen nicht — die BF16-Tensoren müssen erst
komplett im RAM sein.

### TODO: Streaming Quant-on-Load
Pro Layer laden → quantisieren → BF16 freigeben → nächster Layer.
Betrifft DefaultModelLoader + AutoWeightsLoader — multiquant-übergreifend,
nicht XFP-spezifisch. Alle quant-on-load Methoden profitieren.

## Streaming Quant-on-Load Implementierung

### Versuch 1: numel-tracking + process_weights trigger
- weight_loader wrapper trackt geladene Elemente pro Layer
- Bei Completion → process_weights_after_loading
- Problem: Parameter schon bei create_weights in BF16 allokiert → Peak = volles Modell

### Versuch 2: Meta device + lazy materialization  
- Parameter auf meta device (0 RAM) nach create_weights
- Materialisierung beim ersten weight_loader call
- GLM: Memory stabil bei 70-80 GB frei während Laden (funktioniert!)
- Crash nach Laden: FusedMoE weight_loader hat spezielle Signatur
  (param, loaded_weight, weight_name, shard_id, expert_id)
  die der generische Wrapper nicht versteht
- MoE-Layer bleiben auf meta device → crash bei XFP-Packing

### Nächster Schritt
FusedMoE weight_loader Signatur im Streaming-Wrapper handhaben.
Oder: separate Streaming-Logik für Linear (einfach) vs FusedMoE (komplex).

## FP8 LM Head _scaled_mm: 57.9 tok/s!

Cached transposed FP8 weight + torch._scaled_mm = FP8 Tensor Core GEMM.
Schneller als BF16 F.linear! **57.9 tok/s = 104% von Marlin INT4.**
Tag: `xfp_faster_than_marlin`

## Math-Test Analyse

Format-Vergleich (FP8 baseline, 10 Aufgaben):
- Completions `{a} {op} {b} = `: **90%** ← bestes Format
- Chat (alle Varianten): **0%** — GLM startet mit "Analyze the Request..."
- Completions ohne Spaces: 70%

Fehler-Muster im 50er Test (FP8 baseline = 60%):
- 6/50 negative Subtraktionen → Modell versagt bei allen (a-b mit a<b)
- 4-5 große Multiplikationsfehler (off-by-one, 3-stellig × 3-stellig)
- Completions-Artefakte (Modell erklärt statt rechnet)

→ 40% der Fehler sind Modell-inherent, nicht Quantisierung.
→ Realer XFP-Impact: 4-12% Math-Differenz zu FP8.
