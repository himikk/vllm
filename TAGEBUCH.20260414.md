# Tagebuch 2026-04-14: Streaming Quant-on-Load für MoE + Registry-Fix

## Ausgangslage

- Gestern (04-13) abend: Streaming-Pfad für Linear funktionierte, für FusedMoE
  nicht. Qwen 122B lud zwar nicht, 35B Test stand aus.
- Image `vllm-xfp-bf16:latest` vom Morgen 08:22 war VOR Fix-Commit gebaut —
  enthielt den MoE-Streaming-Pfad `_make_moe_streaming_loader` noch gar nicht.

## Was heute gelaufen ist

### 1. MoE-Streaming-Loader-Ansatz verworfen
`Qwen3.5MoeForConditionalGeneration.load_weights` ruft `param.weight_loader(...)`
direkt — umgeht jeden Wrapper auf `module.load_weights`. Also:
- MoE-Sonderzweig (`_make_moe_streaming_loader`) wieder entfernt.
- Unified Path: alle Params (Linear + FusedMoE 3D) gehen durch Param-Ebene-Wrapping.

### 2. Param-Subklassen bleiben erhalten via in-place Mutation
Mehrere Iterationen:
- `torch.nn.Parameter(...)` neu erstellen → verliert `load_merged_column_weight`
- `param.data = meta_tensor` → `set_data` fail: meta ≠ CUDA nicht kompatibel
  (vLLM `ModelWeightParameter.__torch_function__` prüft TensorImpl-Kompatibilität)
- **Lösung:** `param.data = torch.empty(0, dtype, device=<same>)` — 0-size auf
  demselben Device, Storage wird befreit, Subklasse+Methoden bleiben intakt.

### 3. XFP/FP8 MoE-Dispatch schlug fehl (UnquantizedFusedMoEMethod statt XFPMoE)
Root cause: `AutoRoundRTNConfig.from_config` rebuild Registry nur im **API-Server**.
Engine Core (spawned Prozess) bekommt gepicklte Config ohne Registry-Info →
`MultiQuantPolicyRegistry._active is None` → `create_weight_method` gibt
`UnquantizedLinearMethod` / `None` zurück.

**Fix:** `mq_dtype` / `mq_gs` als Instanz-Attribute speichern (überleben Pickle).
`_ensure_registry()` lazy beim ersten `get_quant_method` im Engine Core.

### 4. Meta-Device-Context in XFPMoEMethod.create_weights entfernt
`with torch.device("meta"):` um `_unquant.create_weights` war ein Versuch, 60 GB
BF16-MoE nie zu allokieren. Folge: Streaming-Loader musste meta → cuda set_data,
was an Subklassen-typecheck scheiterte. Entfernt — MoE-Create-Weights allokiert
kurzzeitig BF16 auf CUDA, dann Streaming-Loader swappt zu 0-size.

## Memory-Instrumentierung eingebaut

`_mem_snapshot()` in `utils.py`:
- `VmRSS`, `VmHWM`, `VmData`, `PssAnon` aus `/proc/self/{status,smaps_rollup}`
- `MemTotal/Avail/Free/Cached` aus `/proc/meminfo`
- `torch.cuda.memory_allocated() / memory_reserved()`
- Logged bei streaming-entry, nach meta-swap, und pre/post-quant per Layer
  (alle 20 Linear gekoppelt, jedes MoE, gc.collect dazwischen).

## torch.cuda.empty_cache() nach jedem Layer-Quant

Befund aus erstem mem-trace: bei MoE#1 sprang `CUDA resv` von 67 → 85 GiB
(Lloyd-Iteration reservierte 17 GiB Pool-Blocks, die nicht zurückgingen).
`empty_cache()` löst das — `resv ≈ alloc` danach, System-MemUsed −20 GiB.

## Ergebnisse Qwen 35B Streaming Load

Mit allen Fixes, 35B-A3B (40 Layer, 256 Experts, moe_inter=512):

| Phase | RSS | CUDA alloc | CUDA resv | MemUsed | Avail |
|---|---:|---:|---:|---:|---:|
| entry | 1.5 | 67.1 | 67.1 | 94 | 28 |
| after meta-swap | 1.5 | **0.15** | 67.1 | 93 | 28 |
| post-Load (vor MoE) | 3.5 | 43.6 | 67.1 | 78 | 44 |
| post-quant MoE#1 | 3.8 | 42.4 | **42.5** | 77 | 45 |
| post-quant MoE#3 | 4.8 | 41.2 | 41.2 | 76 | 46 |
| pre-quant MoE#20 | 3.3 | **31.9** | 31.9 | 67 | 55 |

Alles in GiB. CUDA alloc sinkt stetig: jede MoE-Layer gibt ~600 MiB netto frei
(BF16 raus → xfp4 rein, Faktor 4:1 bei den Weights).

**XFP Auto-Select (Qwen 35B):**
- **40 MoE-Layer:** alle xfp4 (auto-select bei 4-Expert-Sample)
- **Linear-Layer:** aus ~192 Samples: 166× xfp3 (cos≈0.982, outliers 0.15–0.5%),
  26× xfp4 (cos≈0.994, meist `[64×2048]` gates oder `[1152×1152]` attn qkv)

## Offener Crash nach Packing (nicht Streaming-bezogen)

`vllm/multiquant/xfp/online_linear.py:122`:
```python
x_cols = x_cast.index_select(1, outlier_col)
# torch.AcceleratorError: CUDA error: invalid argument
```

Passiert beim ersten Forward/Warmup-Pass nach erfolgreichem Packing aller
Gewichte. Vermutlich `outlier_col` int64 indices out-of-bounds oder
dtype-Mismatch. Separater Bug vom Streaming — morgen.

## Dateien geändert

- `vllm/model_executor/model_loader/utils.py` — Streaming-Loader unified,
  `_mem_snapshot()` mit allen Kategorien, `empty_cache()` per Layer
- `vllm/multiquant/autoround/config.py` — `mq_dtype`/`mq_gs` als Instanz-Attr,
  `_ensure_registry()` für Engine-Core-Rebuild
- `vllm/multiquant/xfp/online_moe.py` — `with meta:` wieder entfernt
- `vllm/multiquant/fp8_moe.py` — dito
- `vllm/multiquant/policy.py` — MQ-DISPATCH log auf debug
- `start.multiquant` — `--weight-dtype` + `--weight-dtype-lm-head` Flags

## TODO morgen

1. `_xfp_outlier_scatter_impl` CUDA invalid-argument debuggen (invalid indices?)
2. Qwen 122B test — sollte jetzt passen (erwarteter Peak alloc ~90 GiB)
3. FP8 vs XFP Vergleich sobald smoke test läuft
