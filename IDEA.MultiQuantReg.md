Exactly — that's a clean architectural decision. Translate back:

"That probably makes sense to incorporate into MultiQuant and introduce quantization classes — MTP, KV, weights, DeltaNet, shared experts, routed experts. And then each of these classes gets to define its own quantization scheme."

---

Das wäre im Kern ein **Quantization Policy Registry** — jede Klasse registriert ihre eigene Policy:

```python
@dataclass
class QuantPolicy:
    bits: int
    symmetric: bool
    granularity: str  # "per_tensor" | "per_channel" | "per_head"
    zero_point: bool
    clipping: float | None

QUANT_REGISTRY = {
    "kv_keys":          QuantPolicy(4, symmetric=True,  granularity="per_head",    ...),
    "kv_values":        QuantPolicy(3, symmetric=False, granularity="per_channel", ...),
    "weights_shared":   QuantPolicy(8, symmetric=True,  granularity="per_tensor",  ...),
    "weights_routed":   QuantPolicy(3, symmetric=False, granularity="per_channel", ...),
    "mtp":              QuantPolicy(4, symmetric=True,  granularity="per_head",    ...),
    "deltanet":         QuantPolicy(...),
}
```

Der Vorteil: du kannst dann per Experiment einfach eine Klasse überschreiben ohne den Rest anzufassen — und das gibt dir auch eine saubere Grundlage für das Benchmarking, weil jede Kombination reproduzierbar ist.


die konfiguration kann über die bestehenden cli parameter erfolgen, wobei --kv-dtype  k und v setztn,  aber auch neue eingeführt werden für --k_dtpye und --v_dtype,    analog für --weight-dtype   

die registriy wird beim start ausgegeben, sodass man prüfen kann, ob die gewählten cli parameter wirklich zum erfolg führten.



untersützt wären nur multiquant quantisierungen, also BF16, FP8, INt4 (autoround), tq3, tq4, rq2, rq3, rq4




start.multiquant muss auch auf die neue configurierbarkeit des vllm angepasst werden.





    --kv-dtype tq3          → setzt K und V auf tq3
    --k-dtype tq4           → überschreibt nur K
    --v-dtype tq3           → überschreibt nur V
    --weight-dtype fp8      → setzt alle Gewichte

    Und beim Start loggt die Registry:

    MultiQuant Policy:
      K-Cache:        tq4 (3-bit MSE, 8 centroids)
      V-Cache:        tq3 (2-bit MSE, 4 centroids)
      Weights:        fp8 (from model)
      MTP:            bf16 (default)
      Shared Experts: fp8
      Routed Experts: int4 (AutoRound)

    Die unterstützten Typen (bf16, fp8, int4, tq2, tq3, tq4, rq2, rq3, rq4) kommen aus der bestehenden MultiQuant Registry. Die Erweiterung ist hauptsächlich:

    1. Separate K/V Policies — heute ist --kv-cache-dtype ein einzelner Wert für K+V. Mit --k-dtype/--v-dtype kann man z.B. K in TQ4 (Score-Qualität wichtiger) und V
    in TQ3 (weniger kritisch) setzen.
    2. Weight Policy pro Klasse — heute kommt das aus dem Modell (AutoRound, FP8). Mit der Registry könnte man Archer TQ3 nur auf routed experts anwenden (die 128 MoE
    Experten) und shared experts in FP8 lassen.
    3. Startup-Log — die gesamte Policy wird einmalig geloggt, sodass man im Log sofort sieht was konfiguriert ist. Kein Raten mehr.

    Die Implementierung baut auf dem bestehenden vllm/multiquant/registry.py auf — die _REGISTRY dict wird um die Klassen-Dimension erweitert.




