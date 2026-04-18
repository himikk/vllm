# MultiQuant JIT — Kernel Management ohne Autotuner

## Problem

FlashInfer's MoE Autotuner hängt 25+ Minuten auf SM121 (GB10). Wird bei jedem MoE-Modell-Start getriggert, auch wenn MultiQuant/Marlin die GEMM-Kernels bereitstellt. Inakzeptabel.

## Grundsatz

**Kein Autotuning zur Laufzeit.** Die Plattform (SM121) und das Datenformat (TQ3, RQ3, INT4) bestimmen den Kernel eindeutig. Es gibt nichts zu "tunen".

| Gewichte | GEMM Kernel | Autotuner nötig? |
|----------|-------------|-----------------|
| INT4 (GPTQ/AWQ/AutoRound) | Marlin | Nein — fest |
| Archer TQ3/RQ3 | CUDA Unpack + cuBLAS | Nein — fest |
| FP8 | cuBLAS FP8 | Nein — fest |
| BF16 | cuBLAS BF16 | Nein — fest |

| KV-Cache | Attention Kernel | Autotuner nötig? |
|----------|-----------------|-----------------|
| TQ3/RQ3 | MultiQuant Compressed Score | Nein — fest |
| FP8 | FlashInfer/Triton | FlashInfer Autotuner |
| BF16 | FlashInfer/Triton | FlashInfer Autotuner |

**Nur FP8/BF16 KV-Cache braucht FlashInfer** — und damit den Autotuner. MultiQuant KV-Cache nutzt eigene Kernels.

## JIT Strategie

### Build-Zeit (Dockerfile)

Kernels werden beim Image-Build kompiliert:

```dockerfile
# TQ Round-Trip Kernel (KV-Cache Compress)
RUN python3 -c "from torch.utils.cpp_extension import load; \
    load(name='tq_serve', sources=[...], ...)"

# Archer Unpack Kernel (Weight Decompress)
RUN python3 -c "from vllm.multiquant.weight_quant.archer_ops import _load_unpack_kernel; \
    _load_unpack_kernel()"
```

### Serve-Start

Kompilierte `.so` aus Cache geladen — kein nvcc, kein Autotuning:

```
[INFO] Archer unpack CUDA kernel compiled successfully   ← aus Cache, <1s
```

Wenn Cache fehlt (erster Start): JIT kompiliert einmalig (~30s), dann gecacht.

### Fallback

Wenn CUDA Kernel nicht verfügbar: PyTorch-Implementierung (langsamer, sofort).

## Env-Variablen

```bash
# FlashInfer Autotuner komplett deaktivieren
FLASHINFER_DISABLE_AUTOTUNER=1

# FlashInfer MoE nicht nutzen (Triton/Marlin stattdessen)
VLLM_USE_FLASHINFER_MOE_FP8=0

# FlashInfer Version-Check deaktivieren (cubin Mismatch)
FLASHINFER_DISABLE_VERSION_CHECK=1

# MLA deaktivieren (MultiQuant ersetzt MLA-Kompression)
VLLM_MLA_DISABLE=1

# Kein torch.compile (MultiQuant Kernels nicht optimierbar)
--enforce-eager
```

## Start-Script

```bash
#!/bin/bash
# start.multiquant.sh
podman run -d --name mq-serve \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -p 8011:8000 \
  -v /data/tensordata:/data/tensordata \
  -e FLASHINFER_DISABLE_AUTOTUNER=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP8=0 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_MLA_DISABLE=1 \
  localhost/vllm-multiquant \
  vllm serve <model> \
    --kv-cache-dtype tq3 \
    --enforce-eager \
    --trust-remote-code
```

## Kernel-Inventar

| Kernel | Datei | Wann kompiliert | Für was |
|--------|-------|-----------------|---------|
| `tq_serve` | `kernels/turboquant/tq_round_trip.cu` | Build-Zeit | KV Pack (TQ) |
| `archer_unpack` | `kernels/archer/archer_unpack.cu` | Erster Serve | Weight/KV Unpack |
| `archer_decompress` | `kernels/archer/archer_decompress.cu` | Erster Serve | Weight Decompress (D≤256) |
| Triton MQ Decode | `vllm/v1/attention/ops/triton_mq_decode.py` | Erster Serve | KV Decode (Fused) |
| Marlin GEMM | vLLM built-in | Build-Zeit | INT4 Weight GEMM |
| cuBLAS | PyTorch built-in | — | BF16/FP8 GEMM |
