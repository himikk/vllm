# TASK: INT4 RTN mit echtem Marlin-Kernel — kein Fallback

## Problem

Der aktuelle INT4 RTN-Pfad nutzt einen dequant-cache Fallback:
- Speichert die volle FP16-Matrix im RAM → kein Speichervorteil
- Langsamer als BF16 (15 tok/s vs 28 tok/s)
- Kein Mensch braucht das
- Der Marlin-Kernel funktioniert isoliert (cos=0.994) aber nicht im vLLM-Serving

## Root Cause

`GPTQMarlinLinearMethod` erwartet spezifische Parameter-Typen:
- `qweight`: `PackedvLLMParameter(input_dim=0, output_dim=1, packed_dim=0, packed_factor=8)`
- `scales`: `GroupQuantScaleParameter(output_dim=1, input_dim=0)`
- `qzeros`: `PackedvLLMParameter(input_dim=0, output_dim=1, packed_dim=1, packed_factor=8)`
- `g_idx`: `RowvLLMParameter(input_dim=0)`

Unser RTN erstellt `nn.Parameter(torch.empty(...))` — ohne die Metadata
(`packed_dim`, `packed_factor`, `input_dim`, `output_dim`). Der
`MarlinMPLinearKernel.process_weights_after_loading` nutzt diese Metadata
fuer `permute_param_layout_` und `gptq_marlin_repack`.

## Loesung

Zwei Ansaetze:

### A: create_weights mit PackedvLLMParameter (bevorzugt)

```python
class AutoRoundRTNLinearMethod:
    def create_weights(self, layer, ...):
        # Fuer INT4: alloziere GPTQ-Format direkt
        if self.bits == 4:
            # Gleiche Parameter wie GPTQMarlinLinearMethod.create_weights
            qweight = PackedvLLMParameter(
                data=torch.empty(K // 8, N, dtype=torch.int32),
                input_dim=0, output_dim=1,
                packed_dim=0, packed_factor=8,
                weight_loader=weight_loader,  # <-- hier muss ein custom loader
            )
            scales = GroupQuantScaleParameter(
                data=torch.empty(n_groups, N, dtype=params_dtype),
                output_dim=1, input_dim=0,
                weight_loader=weight_loader,
            )
            ...
        else:
            # INT2/INT3: BF16 weight laden, in process_weights_after_loading packen
            weight = ModelWeightParameter(...)
```

Problem: der `weight_loader` aus vLLM laedt safetensor-Keys die zum
Parameter-Namen matchen. Wenn wir `qweight` registrieren, sucht vLLM
nach `...qweight` im BF16-safetensor — das existiert nicht!

### B: BF16 laden, dann Parameter ERSETZEN (pragmatisch)

```python
class AutoRoundRTNLinearMethod:
    def create_weights(self, layer, ...):
        # BF16 laden wie bisher
        weight = ModelWeightParameter(...)
        layer.register_parameter("weight", weight)
        # Merke die Dimensionen
        layer._rtn_K = input_size_per_partition
        layer._rtn_N = sum(output_partition_sizes)
    
    def process_weights_after_loading(self, layer):
        W = layer.weight.data  # [N, K] BF16, korrekt geladen
        qw, sc, qz = rtn_pack_gptq(W.float(), 4, 128)
        
        # ERSETZE Parameter mit korrekten Typen
        layer.register_parameter("qweight", PackedvLLMParameter(
            data=qw, input_dim=0, output_dim=1,
            packed_dim=0, packed_factor=8))
        layer.register_parameter("scales", GroupQuantScaleParameter(
            data=sc, output_dim=1, input_dim=0))
        layer.register_parameter("qzeros", PackedvLLMParameter(
            data=qz, input_dim=0, output_dim=1,
            packed_dim=1, packed_factor=8))
        layer.register_parameter("g_idx", RowvLLMParameter(
            data=torch.empty(0, dtype=torch.int32), input_dim=0))
        
        # Erstelle denselben Marlin-Kernel wie GPTQMarlinLinearMethod
        kernel = MarlinMPLinearKernel(config, ...)
        kernel.process_weights_after_loading(layer)  # Repack!
        layer._marlin_kernel = kernel
        
        del layer.weight  # BF16 freigeben
    
    def apply(self, layer, x, bias):
        return layer._marlin_kernel.apply_weights(layer, x, bias)
```

## Dateien

| Datei | Aenderung |
|-------|-----------|
| `online_linear.py` | Ansatz B implementieren: PackedvLLMParameter in process_weights |
| `online_linear.py` | _apply_int4_dequant und _apply_marlin ENTFERNEN |
| `online_linear.py` | apply() fuer INT4 → kernel.apply_weights() |

## Verifizierung

1. `--weight-dtype-attn int4` → Marlin Kernel, ~49 tok/s, 50% Math
2. `--weight-dtype-attn int4 --weight-dtype-routed int4` → Attention Marlin, MoE BF16
3. Kein dequant-cache, kein Fallback
4. GPU Memory soll ~25% weniger sein als BF16 fuer Attention-Gewichte

## Referenz

Prequant INT4 Marlin (funktioniert, 49 tok/s):
- `GPTQMarlinLinearMethod` in `gptq_marlin.py:329`
- `MarlinMPLinearKernel` in `kernels/linear/mixed_precision/marlin.py`
- `PackedvLLMParameter` in `parameter.py:353`
- `GroupQuantScaleParameter` in `parameter.py` (suchen)
