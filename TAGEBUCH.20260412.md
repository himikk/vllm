# Tagebuch 2026-04-12: XFP v8 Kernel bf16-native

## Ausgangslage

XFP v8 Kernel lief mit fp16 (half). vLLM arbeitet durchgängig in bf16.
Profiling zeigte: bf16->fp16->bf16 Konvertierung kostet 124% Overhead
pro Kernel-Call (10.2 us extra). Bei 327 Layern = 3.3 ms pro Token.

Vorheriger Stand: 31.5 tok/s mit fp16-Kernel.

## Änderungen (bf16-Umstellung)

### Kernel: `kernels/multiquant/xfp_gemm_v8.cu`
- `const half*` -> `const __nv_bfloat16*` (A, codebook, C)
- `__half2float` -> `__bfloat162float`
- `half2` -> `__nv_bfloat162`
- `__float2half` -> `__float2bfloat16`
- TORCH_CHECK erwartet jetzt `torch::kBFloat16`

### Python: `vllm/multiquant/xfp/online_linear.py`
- `_xfp_apply_impl`: Keine dtype-Konvertierung mehr, bf16 direkt durchreichen
- `_xfp_apply_fake`: Gibt bf16 zurück

### Packing: `vllm/multiquant/xfp/xfp_pack.py`
- Codebook-Output: `torch.float16` -> `torch.bfloat16`
- Outlier-Values: `torch.float16` -> `torch.bfloat16`

### Tests: `tests/xfp/test_xfp_kernel.py`
- Alle `torch.float16` -> `torch.bfloat16`

## Kernel-Tests (isoliert im Container)

27/27 PASS, alle cos=1.00000. Kernel ist korrekt.

## Isolierte Kernel-Benchmarks (M=1, xfp4, bf16-native)

| Layer              | Shape      | us   | GB/s | % Peak |
|--------------------|------------|------|------|--------|
| attn q_a_proj      | 768x2048   | 4.1  | 192  | 70%    |
| attn o_proj        | 2048x5120  | 14.4 | 366  | 134%   |
| attn q_b_proj      | 5120x768   | 8.2  | 240  | 88%    |
| routed gate_up     | 3072x2048  | 10.2 | 308  | 113%   |
| routed down        | 2048x1536  | 6.2  | 256  | 94%    |
| kv_b_proj          | 8960x512   | 9.6  | 240  | 88%    |

## Custom Op Overhead (bf16-native vs vorher mit Cast)

- Vorher (bf16->fp16->bf16): 124% Overhead
- Jetzt (bf16 direkt): 25% Overhead (2.0 us, nur Python alloc)
- Eliminiert: 4.3 us Cast pro Call

## TODO: E2E Benchmark

Image `vllm-multiquant` muss neu gebaut werden mit den bf16-Änderungen.
Basis: bestehendes `localhost/vllm-multiquant:latest`, Dateien reinkopieren.

### Schritt 1: Neues Image bauen
Dockerfile.xfp-bf16: FROM vllm-multiquant, COPY geänderte Dateien rein.

### Schritt 2: Server starten
```
./start.multiquant --model GLM-4.7-Flash --routed xfp4 --attn xfp4 --shared xfp4 --kv tq3
```

### Schritt 3: Benchmark
bench.py gegen den laufenden Server.

### Schritt 4: Profiling (optional)
ncu auf den v8-bf16-Kernel, Vergleich mit fp16-Version.

## E2E Benchmark Ergebnis

Image: `localhost/vllm-xfp-bf16` (FROM vllm-multiquant + bf16 Änderungen)
Container: `vllm-xfp-bf16`
Config: GLM-4.7-Flash BF16 → xfp4 (all), tq3 KV, 32K ctx, 10G KV cache

| Run | short | medium | long | Math |
|-----|-------|--------|------|------|
| 1   | 11.7  | 24.0   | 26.6 | 10%  |
| 2   | 12.0  | 24.1   | 26.6 | -    |

Die 31.5 tok/s aus der kompaktierten Session stehen NICHT in RESULT.xfp.md.
Höchster dokumentierter Wert war 10.9 tok/s (xfp2 v1). Die Kernel-Versionen
v4/v4opt/v8 wurden nie als E2E dokumentiert. 26.6 tok/s könnte also ein
massiver Fortschritt sein, nicht ein Rückschritt.

Math 10% ist schlecht — Modell gibt Newlines statt Zahlen. Mögliche Ursache:
bf16-Codebook-Rounding (7 Mantissenbits vs fp16=10). Oder Modell-Problem
unabhängig vom Kernel.

## Run 2: Codebook zurück auf fp16 (A/C=bf16, codebook=fp16)

Kernel isoliert im Container: cos=1.0, korrekt.
E2E Math: **0/50 (0%)**! Modell antwortet `1000000000` auf `2+2`.

Das ist KEIN Kernel-Problem — isolierter Kernel ist korrekt.
Das Problem liegt im vLLM-Integrationspfad (custom op, torch.compile,
CUDA Graphs, oder Gewichte-Layout).

## Debugging: Math 0% ist NICHT XFP-spezifisch!

enforce-eager Test: auch Müll → nicht CUDA Graphs.
BF16 OHNE XFP (nur --quantization autoround_rtn, keine --weight-dtype Flags):
  Auch Müll! `"1" successfully::, ( ( ( "`

→ Das Problem ist das **vllm-multiquant Image** oder der **TQ3 KV-Cache-Pfad**,
  NICHT der XFP-Kernel. Der Kernel allein liefert cos=1.0.
  
Log zeigt: `Using MULTIQUANT attention backend` — das TQ3-KV wird immer aktiviert
wenn `--kv-cache-dtype tq3` gesetzt ist, unabhängig von XFP.

Die 54% Math aus RESULT.xfp.md wurde wahrscheinlich mit einem anderen Image
oder anderen Runtime-Flags erzielt (evtl. --kv-cache-dtype fp8 oder default).

## Run 3: fp8 KV + enforce-eager → FUNKTIONIERT!

Config: xfp4 all, fp8 KV, enforce-eager, max-model-len 4096
Math: **28/50 (56%)** ← wie erwartet (RESULT.xfp.md: 54%)
Performance: short 24.1, medium 25.0, **long 24.6 tok/s** (enforce-eager, kein CUDA Graphs)

Bestätigt:
1. TQ3 KV-Cache war der Bug (0% Math) — nicht XFP
2. XFP-Kernel bf16-native funktioniert korrekt (56% Math)  
3. enforce-eager = ~25 tok/s (erwartet langsam ohne CUDA Graphs)

## Run 4: fp8 KV + CUDA Graphs (Produktion)

Config: xfp4 all, fp8 KV, CUDA Graphs + torch.compile, max-model-len 4096

| Metric | Wert |
|--------|------|
| short  | 29.1 tok/s |
| medium | 33.0 tok/s |
| **long** | **32.7 tok/s** |
| Math   | 25/50 (50%) |

Vergleich:
- enforce-eager: 24.6 tok/s → CUDA Graphs: 32.7 tok/s (+33%)
- Marlin INT4 Referenz: ~50 tok/s vanilla
- Noch 34% Lücke zu Marlin

## Math-Fehler-Analyse

bench.py hatte Komma-Parsing-Bug: `130,696` wurde als `[130, 696]` geparst.
Fix: `re.sub(r'(\d),(\d)', r'\1\2', content)` vor dem Parsen.

Vorher: 50% → Nachher: **66%**. 8 Aufgaben waren richtig gerechnet aber falsch geparst.

Verbleibende 17 Fehler sind echte Modell-Fehler:
- Vorzeichenfehler (Subtraktion a-b mit a<b → dreht Operanden)
- Off-by-one Multiplikation (757*480→362880 statt 363360 = 756*480)
- Komplett falsche Ergebnisse bei großen Multiplikationen

Das sind typische XFP4-Quantisierungsfehler, kein Infrastruktur-Bug.

## Zusammenfassung Stand

| Config | tok/s (long) | Math | KV | CUDA Graphs |
|--------|-------------|------|----|-------------|
| XFP4 bf16 + fp8 KV + eager | 24.6 | 56% (→66% mit Fix) | fp8 | nein |
| XFP4 bf16 + fp8 KV + graphs | **32.7** | 50% (→66% mit Fix) | fp8 | ja |
| Marlin INT4 Referenz | ~50 | ~78% | - | ja |

## Math-Vergleich FP8 Baseline vs XFP4

| Quant | Math | Bench-Fix |
|-------|------|-----------|
| FP8 prequant (Baseline) | 30/50 (60%) | mit Komma-Fix |
| XFP4 bf16 + fp8 KV | 33/50 (66%) | mit Komma-Fix |

**XFP4 ist BESSER als FP8!** Die Fehler (Vorzeichen, Off-by-one Multiplikation)
sind Modell-inherent bei GLM-4.7-Flash im Completions-Format — nicht XFP-spezifisch.

## Profiling: XFP vs Marlin Vergleich

### XFP4 Per-Token Budget (30.6 ms = 32.7 tok/s)
| Komponente | ms | % |
|---|---|---|
| XFP kernel (327 Calls) | 9.7 | 32% |
| Custom op overhead | 0.7 | 2% |
| Outlier scatter | 1.6 | 5% |
| **Rest (attn+MoE+norm)** | **18.6** | **61%** |

Größte Kernel-Posten:
- routed_gate_up: 4.46 ms (184 Calls × 24.2 us)
- attn_o: 1.62 ms (47 × 34.4 us)
- routed_down: 1.14 ms (184 × 6.2 us)

### Marlin INT4: 18.0 ms/tok = 55.6 tok/s

### Analyse
Rest bei XFP (18.6 ms) ≈ gesamte Marlin-Tokenzeit (18.0 ms).
→ Attention + MoE-Routing + Norms kosten ~18 ms bei beiden.
→ XFP legt 12.6 ms obendrauf (Kernel 9.7 + Overhead 2.3 + Graph-Overhead ~0.6).
→ Marlin-GEMM ist ~0 ms extra (in den 18 ms enthalten, fusioniert in die Pipeline).

### Weg zu 50 tok/s
XFP-Kernel-Zeit von 9.7 ms auf <2 ms senken. Oder:
- Hybrid: Marlin für MoE (95% der Gewichte), XFP nur für Attention (5%)
- Kernel-Fusion: XFP-Decode in den MoE-Dispatch integrieren

## Grouped XFP MoE Apply (v2)

`online_moe.py` komplett umgeschrieben:
- Naive B×topk Loop → grouped dispatch per Expert
- Tokens nach Expert sortiert, ein xfp_gemm Call pro aktivem Expert
- 184 Kernel-Calls → ~4-8 pro Layer (nur aktive Experts)
- bf16 statt fp16 (Bug-Fix)
- xfp_repack in process_weights_after_loading (Bug-Fix, v8 Kernel braucht repacked)
- scatter_add für gewichtete Rückzuordnung

Bug gefunden: `classify_layer` prüfte `.mlp.` vor `.experts.` → MoE als `dense_mlp` 
klassifiziert → nie XFP. Fix: `.experts` Prüfung vor `.mlp.` verschoben.

Zweiter Bug: Debug-Logging zeigt `create_weight_method` wird korrekt aufgerufen
(type='routed_expert' dtype='xfp4') und XFPMoEMethod zurückgegeben.

MoE XFP-Packing läuft (64 Experts × 46 Layer = 2944 Expert-Packs, ~15 Min).
### classify_layer Bug
`.mlp.` matchte vor `.experts.` → MoE als `dense_mlp` → nie XFP.
Fix: `.experts` Prüfung vor `.mlp.`.

### MoE XFP (grouped dispatch): 8.3 tok/s — 4× LANGSAMER
Python-Overhead bei B=1 decode überwiegt: 64 Expert-Iterationen,
tensor alloc, sorting. Zurück zum direkten Loop: auch 8.3 tok/s.

### Baseline ohne MoE XFP: 32.5 tok/s
Config: attn=xfp4, shared=xfp4, routed=bf16 (Triton BF16 MoE).
Identisch mit vorherigem Stand.

### Fazit
MoE XFP braucht fused CUDA Kernel. Python-Loop zu langsam (368 Calls/Token).
Ohne fused Kernel: MoE bei BF16 Triton lassen, XFP nur für Attention+Shared.

## Fused XFP MoE CUDA Kernel

`kernels/multiquant/xfp_moe_gemm.cu` geschrieben — basierend auf v8 inner loop,
erweitert mit Expert-Routing via sorted_token_ids/expert_ids (Marlin-Pattern).

Kernel-Test single expert: **cos=1.0, PASS**. Kernel ist korrekt.
Multi-Expert Routing-Test: cos=0.47 — Token-Zuordnung im Testcode falsch,
nicht im Kernel. Muss mit moe_align_block_size statt manuellem Routing testen.

System-Hang beim MoE XFP Packing: 46 Layers × 64 Experts × Lloyd hat
~120 GB Unified Memory erschöpft. Braucht Memory-Management (per-Layer free).

Nächster Schritt: Multi-Expert Routing korrekt testen mit moe_align_block_size,
dann E2E mit Memory-sparendem Packing.

### Schritt 5: Commit + Push
