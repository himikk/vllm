# TASK: Quant-on-Load INT2/INT3/INT4 — Unified Storage

## Status (2026-04-09)

- [x] Phase 1: ComponentInfo + analyze_model() + enhanced log_policy()  (c984df175)
- [x] Phase 3a: rtn_pack_gptq() — INT2/INT3/INT4  (df2c9b427)
- [x] Phase 2: Per-class RTN routing  (96f8485a7)
- [x] Phase 3b/c: Linear + MoE unified storage  (96f8485a7)
- [x] Phase 4: CLI + start.multiquant flags  (19f7c9ee0)
- [ ] Phase 5: E2E Test im Container auf DGX/RTX
- [ ] INT4-best AutoRound laeuft auf Spiegel 2 (seit 09:09 UTC)

Verifiziert (CPU):
- rtn_pack_gptq Roundtrip: INT2 cos=0.74, INT3 cos=0.96, INT4 cos=0.99
- Per-class Routing: routed→int3, shared→bf16, attn→int4 korrekt
- Registry Banner zeigt Current/Target/Detail

Noch offen:
- E2E im vLLM Container (GPU, echte Inference)
- Marlin-Integration fuer INT4 RTN (aktuell: CPU decompress fallback)
- Performance-Vergleich RTN vs prequant

## Ziel

BF16-Modelle beim Laden per RTN (Round-to-Nearest) in INT2/INT3/INT4 quantisieren,
wobei dasselbe Speicherformat und derselbe Inference-Pfad genutzt wird wie bei
pre-quantisierten AutoRound/GPTQ-Modellen.

Nutzer waehlen per CLI pro Komponentenklasse die Zielquantisierung:

```bash
./start.multiquant --model GLM-4.7-Flash \
  --routed int3 --shared bf16 --attn int4 --kv tq3
```

## Motivation

- RTN ist minimal schlechter als kalibrierte Quantisierung, aber taugt fuer on-the-fly
- Primaer INT3 und INT4, INT2 nur fuer experimentierfreudige Nutzer bei groesseren Modellen
- Kein separater Decompress-Pfad: RTN packt in GPTQ-Format, Inference nutzt dieselben Kernels
- Nutzer sollen visuell sehen: welche Klassen erkannt, IST-Quantisierung, SOLL-Quantisierung

## Architektur

### Zwei Pfade, ein Ziel-Format

```
Pre-quantized (AutoRound/GPTQ safetensor):
  safetensor qweight/scales/qzeros
  → INCConfig/GPTQConfig loads
  → MQSub4LinearMethod.apply()  → mq_gemm_int2/int3 Kernel
  → MQSub4MoEMethod.apply()     → per-expert mq_gemm_int2/int3
  → (INT4: Marlin Kernel)

RTN on-the-fly (BF16 Modell):
  safetensor BF16 weight
  → Normal load
  → process_weights_after_loading(): rtn_pack_gptq() → GPTQ-Format
  → DERSELBE MQSub4LinearMethod.apply()  → mq_gemm_int2/int3
  → DERSELBE MQSub4MoEMethod.apply()     → per-expert mq_gemm_int2/int3
  → (INT4: Marlin)
```

### Komponentenklassen

| Klasse | Beispiel GLM-4.7-Flash | Typisches Ziel |
|--------|------------------------|----------------|
| `routed_expert` | 64 Experts x 46 MoE-Layer | int3, int4 |
| `shared_expert` | 46 Shared-Expert-Layer | bf16 (1:1) |
| `attn` | 47 Attention-Layer | int4, bf16 |
| `dense_mlp` | Layer 0 (vor MoE) | bf16 (1:1) |
| `mtp` | MTP predict layers | bf16 |
| `k_cache` / `v_cache` | KV-Cache | tq3, tq4, fp8 |

### Registry-Ausgabe (Startup-Banner)

```
MultiQuant Registry:
  ┌────────────────────┬──────────┬──────────┬────────────────────────────┐
  │ Component          │ Current  │ Target   │ Detail                     │
  ├────────────────────┼──────────┼──────────┼────────────────────────────┤
  │ K-Cache            │ bf16     │ tq3w     │ → compress [cli]           │
  │ V-Cache            │ bf16     │ tq3w     │ → compress [cli]           │
  │ Routed Experts     │ bf16     │ int3     │ 64×46 = 2944 → RTN [cli]  │
  │ Shared Experts     │ bf16     │ bf16     │ 46 layers (1:1)            │
  │ Attention          │ bf16     │ int4     │ 47 layers → RTN [cli]      │
  │ Dense MLP          │ bf16     │ bf16     │ Layer 0 (1:1)              │
  │ MTP                │ bf16     │ bf16     │ 1 layer (1:1)              │
  └────────────────────┴──────────┴──────────┴────────────────────────────┘
```

Bei pre-quantisierten Modellen:

```
  │ Routed Experts     │ int4     │ int4     │ 64×46 GPTQ gs=128 [model] │
  │ Shared Experts     │ bf16     │ bf16     │ 46 layers (1:1) [model]    │
```

## Implementierung

### Phase 1: Model-Introspection + Enhanced Log

**Dateien**: `vllm/multiquant/policy.py`

1. `ComponentInfo` Dataclass — beschreibt was im Modell gefunden wurde
2. `analyze_model(hf_config, quant_config)` — introspektiert HF-Config:
   - `n_routed_experts`, `n_shared_experts`, `num_nextn_predict_layers`
   - `first_k_dense_replace`, `moe_layer_freq`
   - `quantization_config` fuer aktuellen Quant-Zustand
3. Enhanced `log_policy()` — zeigt Current/Target/Detail Tabelle
4. `classify_layer(prefix: str)` Helper — ordnet Parameternamen den Klassen zu

**Erkennung aus hf_config Attributen:**

| Attribut | Modell | Klasse |
|----------|--------|--------|
| `n_routed_experts` / `num_local_experts` | GLM, DeepSeek, Qwen | routed experts count |
| `n_shared_experts` / `moe_num_shared_experts` | GLM, Ernie | shared experts count |
| `num_nextn_predict_layers` | GLM, DeepSeek | MTP layers |
| `first_k_dense_replace` | GLM, DeepSeek | dense MLP layers (0..k-1) |
| `num_hidden_layers` | alle | total decoder layers |

### Phase 2: Per-Class RTN Routing

**Dateien**: `vllm/multiquant/autoround/config.py`, `vllm/multiquant/policy.py`

1. `AutoRoundRTNConfig.get_quant_method(layer, prefix)` nutzt `classify_layer(prefix)` 
   um Layer-Typ zu bestimmen und Policy abzufragen
2. Wenn `policy.is_quantized` und Layer BF16 → RTN-Methode zurueckgeben
3. Wenn Policy "1:1" (bf16) → `None` zurueckgeben (kein Quant)
4. Policy-Registry wird via `VllmConfig` an Config durchgereicht

### Phase 3: Unified RTN Packing

**Dateien**: 
- `vllm/multiquant/autoround/rtn_pack.py` (NEU)
- `vllm/multiquant/autoround/online_linear.py` (umbauen)
- `vllm/multiquant/autoround/online_moe.py` (NEU)

#### `rtn_pack_gptq(W, bits, group_size)` — Shared Packing

```python
def rtn_pack_gptq(W: Tensor, bits: int, group_size: int) 
    -> tuple[Tensor, Tensor, Tensor]:
    """BF16 → GPTQ-kompatibles Packed Format.
    
    Input:  W [N, K] float (PyTorch Linear: [out_features, in_features])
    Output: qweight [K_packed, N] int32  (GPTQ Konvention)
            scales  [n_groups, N] float16
            qzeros  [n_groups, N_zp] int32
    
    INT2: 16 values/int32, symmetric, zp = 0x55555555
    INT3: bitstream, 32 bits/int32, symmetric
    INT4: 8 values/int32, GPTQ-Standard, Marlin-kompatibel
    """
```

#### Linear RTN (online_linear.py umbauen)

```python
class AutoRoundRTNLinearMethod:
    def process_weights_after_loading(self, layer):
        W = layer.weight.data  # [N, K] BF16
        qweight, scales, qzeros = rtn_pack_gptq(W.float(), bits, group_size)
        # Ersetze: BF16 weight → GPTQ Tensoren
        layer.qweight = Parameter(qweight)
        layer.scales = Parameter(scales)
        layer.qzeros = Parameter(qzeros)
        layer.g_idx = Parameter(torch.empty(0, dtype=torch.int32))
        del layer.weight  # BF16 RAM freigeben
        layer._rtn_bits = bits
        layer._rtn_group_size = group_size
    
    def apply(self, layer, x, bias=None):
        # Delegiert an MQSub4LinearMethod (INT2/3) oder Marlin (INT4)
        return self._sub4_method.apply(layer, x, bias)
```

#### MoE RTN (online_moe.py, NEU)

```python
class AutoRoundRTNMoEMethod:
    def create_weights(self, layer, ...):
        # Alloziere BF16 wie Standard FusedMoE
        # ODER: alloziere direkt GPTQ-Tensoren wenn Ziel bekannt
    
    def process_weights_after_loading(self, layer):
        # Per Expert: rtn_pack_gptq() auf gate/up/down
        # Speichere als w13_qweight, w2_qweight etc. (MoE-Format)
        # Dann: MQSub4MoEMethod.process_weights_after_loading()
        #   → transformiert von MoE-Layout zu Kernel-Format
    
    def apply(self, layer, x, topk_weights, topk_ids, shared_input):
        # Delegiert an MQSub4MoEMethod.apply()
```

### Phase 4: CLI + start.multiquant

**Dateien**: `vllm/engine/arg_utils.py`, `start.multiquant`

1. `--weight-dtype-mtp` CLI Arg (neben bestehenden)
2. start.multiquant erweitern:

```bash
# Per-Class Flags
--routed TYPE    # → --weight-dtype-routed TYPE
--shared TYPE    # → --weight-dtype-shared TYPE  
--attn TYPE      # → --weight-dtype-attn TYPE
--mtp TYPE       # → --weight-dtype-mtp TYPE

# Beispiele:
./start.multiquant --model GLM-4.7-Flash --routed int3 --kv tq3
./start.multiquant --model GLM-4.7-Flash --routed int4 --attn int4 --shared bf16
./start.multiquant --model GLM-4.7-Flash-int4-AutoRound  # alles vom Modell
```

### Phase 5: Verifikation

| Test | Kommando | Erwartung |
|------|----------|-----------|
| Registry-Ausgabe BF16 | `--model GLM-4.7-Flash` | Alle Klassen bf16, erkannte Struktur |
| Registry-Ausgabe prequant | `--model GLM-4.7-Flash-int4-AutoRound` | Routed=int4, Shared=bf16 |
| RTN INT4 Linear | `--model GLM-4.7-Flash --attn int4` | Attn RTN→INT4, cos>0.94 |
| RTN INT3 MoE | `--model GLM-4.7-Flash --routed int3` | Routed RTN→INT3, cos>0.80 |
| RTN INT4 + prequant | `--model GLM-4.7-Flash-int4-AutoRound --attn int4` | 1:1 laden |
| Selective | `--model GLM-4.7-Flash --routed int3 --shared bf16` | Nur Routed quantisiert |
| bench.py | Vergleich tok/s: RTN INT4 vs prequant INT4 | Gleich (selber Kernel) |
| bench.py | Vergleich cos: RTN INT3 vs AutoRound-best INT3 | RTN etwas schlechter |

## Bestehender Code (wiederverwenden)

| Was | Wo | Wiederverwenden fuer |
|-----|-----|---------------------|
| MQSub4LinearMethod | `weight_quant/mq_sub4_linear.py` | Inference nach RTN-Pack (INT2/3) |
| MQSub4MoEMethod | `weight_quant/mq_sub4_moe.py` | MoE-Inference nach RTN-Pack |
| mq_gemm_int2/int3 | `kernels/multiquant/` | Fused dequant+GEMM Kernels |
| MoeWNA16Method | `quantization/moe_wna16.py` | MoE Weight-Allocation Format |
| rtn_pack_gptq Logik | `autoround/online_linear.py:58-111` | Basis fuer neue rtn_pack.py |
| classify_layer patterns | `quantization/inc.py` | Layer-Name → Klassen-Mapping |
| HF config parsing | `config/model.py`, `config/speculative.py` | MoE/MTP Erkennung |

## Reihenfolge

1. **Phase 1** — Registry + Model-Analyse + Log (sichtbar, low risk)
2. **Phase 3a** — `rtn_pack_gptq()` extrahieren + testen (Fundament)
3. **Phase 2** — Per-Class Routing
4. **Phase 3b/c** — Linear + MoE RTN umbauen auf Unified Storage
5. **Phase 4** — CLI + start.multiquant
6. **Phase 5** — E2E Verifikation
