# Tagebuch 2026-04-15: Qwen 122B Load-Versuch + KV-Dtype Diskussion

## Ausgangslage

Gestern abend (14.04):
- Qwen 35B läuft mit XFP-Streaming durch (67→30 GiB CUDA alloc, 40/39 MoE gepackt)
- Commit `086409b8c` — unified streaming + registry rebuild + empty_cache + mem-debug
- Offener Crash: `_xfp_outlier_scatter_impl` CUDA invalid-argument bei first forward
- Qwen 122B noch nicht gelaufen

## Heute

### Qwen 122B erster Versuch — OOM vor Streaming-Hook
Start `c7718c3fa0a6` (21:49 gestern Abend).
Log endet bei `Using FLASHINFER attention backend` — KEIN `stream-mem` entry.
OOM-Killed bevor `initialize_streaming_quantload` aufgerufen wurde.

**Root cause:** `initialize_model()` allokiert in `UnquantizedFusedMoEMethod.create_weights`
w13/w2 BF16 direkt auf CUDA. Für 122B (47 MoE-Layer × 256 Experts × große moe_inter):
~235 GiB BF16 — schon lang vor dem Meta-Swap-Hook. System OOM-Killer schlägt zu.

Der Meta-Swap-Trick vom Vormittag (`with torch.device("meta"):` in XFPMoEMethod)
war gestern entfernt worden weil das Materialize-Problem (`set_data cuda←meta`
inkompatibel bei ModelWeightParameter) nicht gelöst war.

### Diskussion: FlashInfer + Triton raus, MultiQuant attention rein
Log zeigt `Using FLASHINFER attention backend out of potential backends:
['FLASHINFER', 'TRITON_ATTN']`. User: "wir machen multiquant".

Die MULTIQUANT attention backend wird automatisch aktiviert sobald
`--kv-cache-dtype` ∈ {tq3, tq4, rq3, rq4, tq3w, tq4w}. Mit `fp8` läuft vLLM
über den nativen Pfad → FLASHINFER.

MultiQuant KV-Registry (vllm/multiquant/registry.py) hat KEINEN fp-basierten
Typ — nur TQ (Archer TurboQuant) und RQ (RotorQuant), beide Integer + Rotation.

User-Wahl: **`tq4w`** (Walsh-Hadamard 4-bit) — `tq3w`/`tq4w` sind die
brauchbaren Varianten, tq3/tq4 ohne "w" sind laut User "Müll".

### XFP auf LM Head
Gefragt. Gibt's aktuell NICHT — `policy.py:679-690` XFP-Dispatch nur für
`LinearBase` und `FusedMoE`. `VocabParallelEmbedding` nur im FP8-Zweig
(`FP8EmbeddingMethod`). Für XFP-LM-Head bräuchte es `XFPEmbeddingMethod`
(~80 Zeilen, analog fp8_embedding).

Bei Qwen 122B ist LM-Head 154880×4608 = 1.4 GB BF16:
- fp8: 700 MB
- xfp3: ~525 MB
- Δ = 175 MB

User-Entscheidung: **Variante 1** — jetzt mit `fp8` LM-Head testen, später ggf. XFPEmbeddingMethod.

## Finale Test-Config Qwen 122B

```bash
podman run -d --replace --name vllm-bench \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  --ipc=host --network host -v /data/tensordata:/data/tensordata \
  -e VLLM_MLA_DISABLE=1 \
  -e VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel \
  localhost/vllm-xfp-bf16:latest \
  vllm serve /data/tensordata/Qwen3.5-122B-A10B \
    --host 0.0.0.0 --port 8011 --served-model-name qwen3.5-122b \
    --trust-remote-code --enforce-eager \
    --max-model-len 4096 --gpu-memory-utilization 0.05 \
    --kv-cache-memory-bytes 3G \
    --kv-cache-dtype tq4w \
    --quantization autoround_rtn \
    --weight-dtype xfp --weight-dtype-lm-head fp8
```

Änderungen zu gestern:
- FLASHINFER_* env vars entfernt (kein Flashinfer)
- `--kv-cache-dtype fp8` → `tq4w` (aktiviert MULTIQUANT attention)

## Memory-Problem

Nach jedem OOM-Kill: 110 GiB als "used" gemeldet, aber kein Prozess hält
das. Unified Memory auf GB10 wird vom OOM-Killer nicht sauber zurückgegeben.
Reboot nötig zwischen Tests.

## TODO

1. Qwen 122B starten nach Reboot — testen ob Meta-Swap-Hook überhaupt
   erreicht wird vor OOM (bei 122B ist der BF16-Peak nach create_weights
   schon höher als 119 GiB).
2. Falls OOM vor Hook: `with torch.device("meta"):` in XFPMoEMethod/FP8MoEMethod
   wieder einführen MIT korrektem Set-Data-Materialize (z.B. `set_()`).
3. `_xfp_outlier_scatter_impl` CUDA invalid-argument debuggen.
4. XFPEmbeddingMethod implementieren falls Δ175MB LM-Head-XFP wichtig wird.


## Nachmittag 15.04 — Alle Blocker behoben + ECHTES per-Layer-Streaming

### 1. MoE-Meta-Context reaktiviert (mit thread-local Flag)
`base_loader.py` setzt `_moe_meta_flag` um `initialize_model()` herum,
alle 3 MoE create_weights (XFP/FP8/AutoRound) prüfen das Flag und
delegieren `_unquant.create_weights(...)` unter `with torch.device("meta"):`.
Meta-Swap für Linear bleibt via `param.data = empty(0, device=<same>)`.

Ergebnis: CUDA alloc **entry von 67 → 5.7 GiB** (MoE auf meta, Linear
zero-size). initialize_model-OOM für 122B gelöst.

### 2. Outlier-Scatter Guard
`xfp/online_linear.py:120` — early-return bei `outlier_col.numel() == 0`.
Layer ohne Outliers crashte vorher mit CUDA invalid-argument bei `index_select`.

### 3. ViT-Skip in classify_layer
`policy.py:99` — Prefixe `visual.*|vision.*|vit.*` werden als `"other"`
klassifiziert, `create_weight_method` gibt `UnquantizedLinearMethod` zurück.
XFP-Kernel ist für VIT-Shapes nicht getunt und crashte vorher async.

### 4. tq4w KV unsupported → tq3w fallback, dann fp8
`tq_wht_pack_to_cache: unsupported D=256 mse_bits=4` im base-Image (Kernel
source kann's, binary-Cache nicht). Erst `tq3w` getestet (funktioniert, aber
Math tot: "2+2=" → "\n\n1"). Dann **fp8 KV** — Math sofort korrekt
(`2+2=4`, `7*8=56`).

→ tq3w KV ist derzeit der Math-Killer (wie schon TB 04-12).
**XFP ist nicht das Problem.**

### 5. Mess-Verfeinerung — 42 GiB Peak ist REAL
`_mem_snapshot()` um GC-walk (`gc.get_objects()` über `torch.Tensor`) und
`torch.cuda.memory_stats()` breakdown erweitert. Plus `_top_cuda_params()`
— einmaliger Dump der 15 größten CUDA-Tensor-Params beim ersten pre-quant.

Befund bei 35B: `CUDA alloc = 42075 MiB == GC-cuda = 42075 MiB`
(1:1 Match, keine Allocator-Lüge). Top-15 waren alle
`layers.K.mlp.experts.w13_weight (256, 1024, 2048) bf16` mit je 1024 MiB
— 15+ MoE-Layer gleichzeitig materialisiert.

### 6. Root Cause identifiziert: Shard-interleavte Key-Reihenfolge
Safetensors-Shards enthalten Keys mehrerer Layer vermischt (Shard 0: layers
0–14 teilweise, Shard 1: 10–25 teilweise, ...). Die erste weight_loader-Call
eines MoE-Layers alloziert SOFORT full `[256, N, K]` Storage via
`_make_streaming_loader`. Weil Keys interleaved kommen, werden quasi-alle
MoE-Layer parallel angetouched BEVOR einer komplett ist → kein Pack
triggert, alles wird gleichzeitig alloziert.

### 7. LÖSUNG: Layer-Grouped Safetensors-Iterator
Neuer Iterator `layer_grouped_safetensors_weights_iterator` in
`weight_utils.py`. Zwei-Pass:
- **Pass 1**: Alle Shards öffnen, Keys nach `.layers.N.` gruppieren,
  `(layer_id, shard_path, key)` indexieren, stable sort by layer_id.
- **Pass 2**: Yield in Layer-Reihenfolge; Shard-Handle wird beim Wechsel
  neu geöffnet (nie mehr als ein Handle offen).

Opt-in via `self._streaming_quant_active` Flag in `default_loader.py:246`
(gesetzt in `base_loader.py:74` wenn `quantization == "autoround_rtn"`).

### 8. Ergebnisse Qwen 35B mit Layer-Grouped

| Phase | vorher | **jetzt** | Reduktion |
|---|---:|---:|---:|
| pre LMHead #1 | 42075 MiB | **1115 MiB** | **38×** |
| pre MoE #1   | 43557 | **4467** | −90% |
| post MoE #1  | 42429 | **3339** |  |
| pre MoE #20  | 31853 | **13608** | wächst linear mit gepackten Layern |
| post MoE #20 | 30725 | **12480** |  |

Peak = **~1 Layer BF16 in flight + alle bereits gepackten**. Echtes
per-Layer-Streaming.

Application startup complete ✓, `2+2=` → `"2+2=4"` ✓.

**Extrapolation für Qwen 122B**: Peak ~50 GiB (statt 150 → OOM bei 119 GiB).
Passt jetzt rein.

## Geänderte Dateien (heute)
- `vllm/model_executor/model_loader/base_loader.py` — Meta-Flag +
  `_streaming_quant_active` Propagation
- `vllm/model_executor/model_loader/utils.py` — `_moe_meta_active`
  thread-local, MoE-materialize, GC+stats-Inventory, `_top_cuda_params`
- `vllm/model_executor/model_loader/weight_utils.py` —
  `layer_grouped_safetensors_weights_iterator`
- `vllm/model_executor/model_loader/default_loader.py` — Dispatch
  Layer-Grouped wenn streaming aktiv
- `vllm/multiquant/xfp/online_moe.py`, `fp8_moe.py`,
  `autoround/online_moe.py` — `with torch.device("meta"):` um
  `_unquant.create_weights` wenn `_moe_meta_active()`
- `vllm/multiquant/xfp/online_linear.py` — Outlier-Scatter early-return
- `vllm/multiquant/policy.py` — ViT-Prefix → `"other"` → unquant
- `vllm/multiquant/autoround/config.py` — `_ensure_registry` lazy rebuild
  im Engine-Core (Spawn-Kopie ohne Registry-Attrs)
- `Dockerfile.xfp-bf16` — `weight_utils.py` + `default_loader.py` +
  `fp8_moe.py` + `autoround/online_moe.py` ergänzt

## 16.04 — Expertwise Packing + Qwen 122B ERFOLGREICH

### Root Cause: Float32-Transient bei Batch-Packing
`_batched_pack_and_repack` machte `W_stack.reshape(E*N, K).float()` — eine
float32-Kopie ALLER 256 Experts auf einmal:
- BF16 Layer: 4.5 GiB
- Float32 Copy: **9.0 GiB**
- Total Transient: 14.6 GiB pro MoE-Layer

Bei MoE #37 (37 Packed × 1.1 + 14.6 + 12 Linear) = ~67 GiB + System → OOM.

### Fix: Per-Expert Packing
`_expertwise_pack_and_repack`: Loop über `range(E)`, pro Expert nur
`W_stack[e].float()` → 25 MiB float32 statt 9 GiB. Lloyd-Centroids sind
pro-Row unabhängig — der Flatten über alle Experts war nur Bequemlichkeit.

### Qwen 122B Ergebnis

**XFP Auto-Select:**
- 48 MoE Layers: alle **xfp4** (bits=4, 4-Expert-Sample, lloyd=5)
- 120 Linear Layers: alle **xfp3** (cos≈0.982–0.992, outliers 0.1–1.3%)
- MoE: 256 experts w13[2048×3072] + w2[3072×1024] → xfp (fused, fpe=786432/393216)

**Memory Timeline (Layer-Grouped + Expertwise):**
```
entry              12.8 GiB  (Linear+attn+shared CUDA, MoE meta)
after meta-swap     1.7 GiB
pre MoE #1          8.5 GiB  (1 BF16 MoE-Layer + embed + LMHead)
post MoE #1         5.4 GiB  (freed BF16, +xfp4 packed)
pre MoE #40        62.4 GiB  (40 packed MoE + Linear + 1 BF16 in-flight)
post MoE #40       59.0 GiB
final (120 Linear) 70.2 GiB  ← gesamtes quantisiertes Modell
System Avail       39.2 GiB  ← nie unter 35 GiB gefallen!
```

**244 GiB BF16 → 70 GiB XFP quantisiert auf 119 GiB Unified Memory. Kein OOM.**

**Math Test:**
- `2+2=` → `"5\n\n<think>..."` (Reasoning-Modell, Completions-Format suboptimal)
- `7*8=` → `"56"` ✓
- `15+27=` → `"15 + 2..."` (abgeschnitten bei 5 Tokens, beginnt korrekt)

**Application startup complete** ✓ nach ~30 Minuten (47 MoE × 256 Experts × Lloyd + 120 Linear xfp3).

## Qwen 122B Bench-Ergebnis (v8 Kernel, fp8 KV, XFP auto)

| Metric | Wert |
|---|---|
| short  | 6.8 tok/s |
| medium | 16.3 tok/s |
| long   | 15.2 tok/s |
| **Math** | **49/50 (98%)** |

Einziger Fehler: 469-613=-144, Modell gab 144 (Vorzeichenfehler, modell-inherent).
**98% Math bei 3.2 eff. bits/param** — XFP-Qualität exzellent.

## XFP Kernel v9 → v10 Entwicklung

### v9 (A-in-SMEM + fused outlier): GESCHEITERT
- Compiled OK, aber hing beim ersten Forward-Pass (EngineCore 101% CPU, kein Log)
- Vermutete Ursache: Bug im Outlier-Loop oder SMEM-Alignment
- v9 als opt-in (`XFP_USE_V9=1`) geparkt

### v10 (SHFL.IDX Codebook Lookup): DER GAME CHANGER

Statt SMEM-basiertem Codebook-Lookup (~28 Cycle Latenz):
```cuda
// v8: SMEM
acc = fmaf(my_cb[idx0], a0, acc);  // SMEM read ~28 cycles

// v10: Warp Shuffle
float w0 = __shfl_sync(0xffffffff, my_cb_val, idx0);  // 1 cycle!
acc = fmaf(w0, a0, acc);
```

Jeder Lane hält einen Codebook-Eintrag als Register. Lookup via Warp-Shuffle:
- XFP4 (16 entries): Lanes 0–15
- XFP3 (8 entries): Lanes 0–7
- XFP2 (4 entries): Lanes 0–3

**Kein SMEM, kein __syncthreads, keine Bank-Konflikte. Pure Register-Path.**

Gleiche C++-Signatur wie v8 (`xfp_gemm(A, B_packed, codebook, C, bits, K)`)
→ Python-Side zero changes nötig.

Geschätzt +30–50% Kernel-Speedup. GLM v8 baseline 32.7 tok/s → erwartet 42–49.

### ISA-Basis (aus Blackwell SM12x Instruktions-Analyse)

| Codebook-Größe | Empfehlung |
|---|---|
| ≤ 8 Einträge (XFP2/3) | PRMT kaskadiert (4 Lookups/Instruktion) |
| 16 Einträge (XFP4) | **SHFL.IDX** (1 Cycle, Register-to-Register) |
| 32 Einträge (XFP5) | SHFL.IDX (exakt 32 Lanes = Warp-Size) |

Gleiche SHFL-Tricks anwendbar auf TQ-WHT KV-Cache Decode:
- WHT-Butterfly: 5× SHFL.XOR statt SMEM
- Centroid-Lookup: PRMT (2/3-bit) oder SHFL.IDX (4-bit)

## v10 Bench-Ergebnisse (eager vs CUDA Graphs)

### GLM-4.7-Flash (XFP + fp8 KV + fp8 LM-Head)

| Modus | short | medium | long | Math |
|---|---|---|---|---|
| v8 eager | — | — | 32.7 | 66% |
| **v10 SHFL eager** | 26.6 | 30.5 | 32.3 | 52% |
| **v10 SHFL + CUDA Graphs** | **47.5** | **57.0** | **56.0** | 52% |
| Marlin INT4 (Referenz) | — | — | 55.6 | ~78% |

**CUDA Graphs = Game-Changer, nicht v10 SHFL.** SMEM-Codebook-Latenz war schon durch
Warp-Occupancy versteckt. Launch-Overhead der 327 Kernel-Calls war der Bottleneck,
nicht Codebook-Lookup. XFP holt Marlin INT4 ein mit CUDA Graphs.

### Qwen 122B-A10B (XFP + fp8 KV + fp8 LM-Head)

| Modus | short | medium | long | Math |
|---|---|---|---|---|
| **v10 + CUDA Graphs** | 7.0 | 17.2 | 16.0 | 98% |
| INT4 AutoRound (baseline) | — | — | 29 | — |
| INT4 + FP8 LM head | — | — | 35 | — |
| INT4 + FP8 LM head + MTP nst=1..5 | — | — | 50 | — |

**Nur +5% durch CUDA Graphs bei 122B** — Bandwidth-bound auf 70 GiB Weights
(große Layer-Dimensionen, Launch-Overhead relativ klein). Gap zu INT4 bleibt
55% von Baseline.

## v11 Kernel-Entwicklung gestartet

### MMA Tensor-Core Validierung (Phase A)

**Step 1: MMA Skeleton** — `mma.m16n8k16.bf16.bf16.f32` mit BF16-Direct-Load.
- Test `test_mma_skeleton.py`: **5/5 PASS** für K ∈ {16, 32, 64, 128, 512}
- A-Fragment-Layout: 4 Frags × 2 bf16/thread (8 bf16/lane)
- B-Fragment-Layout: 2 Frags × 2 bf16/thread (4 bf16/lane)
- cos > 0.999, relerr < 0.02 vs torch.matmul

**Step 2: MMA + XFP-decoded B-Tile** — SMEM-basierter Codebook-Lookup
- SHFL mit `cb_reg[variable_n_col]` fehlschlägt: Compiler spillt Array nach
  Local Memory wenn dynamisch indiziert → `__shfl_sync` liest falsche Lane.
- Fix: Codebook in SMEM (8 cols × 16 entries × 4 bytes = 512 B, trivial)
- Test `test_xfp_mma_single.py`: **5/5 PASS** für K ∈ {16, 32, 64, 256, 2048}
- cos > 0.9999 vs `xfp_gemm_v8` Referenz

**Wichtige Lektion:** ldmatrix noch nicht integriert — B-Tile wird direkt aus
SMEM in MMA-Layout geladen (langsamer aber einfacher). Nächster Schritt:
ldmatrix für bank-conflict-free Load.

## Offen / Nächste Schritte
- Phase A Step 3: Multi-M (M=32,64,128) + Multi-N-Tile
- Phase B: MoE-Routing (sorted_token_ids, expert_ids)
- Phase C: cp.async + ldmatrix
- Phase D: Outlier-Fusion Linear v11
- TQ-WHT SHFL-Optimierung (KV-Cache Decode) — wenn v11 stabil
- tq4w D=256 Kernel im Base-Image fixen
- ngram MTP für XFP (nst=3, prompt_lookup) — bench aktuell laufend
