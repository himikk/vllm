# TASK.fix.md — Root Cause Analyse MultiQuant Math-Bug

## Erkenntnisse

### Was wir wissen
- Unit Test 1-Layer cos=0.95 (TQ4) / 0.86 (TQ3) — Algorithmus korrekt
- Live-Serving produziert Müll (`15+27=54`, Mozart="(the)")
- Prefill ist korrekt (erstes Token `3+4=7` stimmt)
- Problem beginnt beim Decode (ab 2. Token)
- Fehler ist NICHT multiplikativ über Layer (jeder Layer hat eigenen KV-Cache)
- cos=0.95 pro Layer REICHT (Google TurboQuant funktioniert für KV-Cache bei tiefen Modellen)
- Post-Loop GEMV ist mathematisch identisch mit voller V-Decompression (verifiziert)
- RQ3/RQ4 sind komplett kaputt (cos≈0, Regression)

### Was wir NICHT wissen
- Warum cos=0.95 im Unit Test aber Müll im Live-System?
- Was passiert zwischen Prefill (korrekt) und Decode (kaputt)?
- Wird der KV-Cache bei Decode korrekt gelesen?

## Hypothese

Wenn cos=0.95 pro Layer reicht und der Unit Test bestätigt dass die Quant
korrekt ist — dann muss der Bug im **vLLM Integrationspfad** liegen:
- KV Cache wird nicht geschrieben (CUDA Graph Guard)
- KV Cache wird falsch adressiert (Strides, Block Table)
- Decode liest aus falschen Cache-Positionen

## Testplan: Von innen nach außen im laufenden Container

Alle Tests mit gemounteten Dateien, kein Build nötig.
Image: `8d373f2ba` (stabil).

### Test 1: Prefill-Output verifizieren
Sende Prompt, logge Output des letzten Prefill-Tokens.
Vergleich: MQ Backend vs FlashAttn Backend (gleicher Prompt).
→ Wenn identisch: Prefill OK
→ Wenn verschieden: Bug in _forward_prefill

### Test 2: Decode Step 1 verifizieren
Nach Prefill: erstes Decode-Token.
Logge: seq_len, block_table, KV-Cache Inhalt (stichprobenartig).
→ Wurde KV korrekt geschrieben? (Cache non-zero nach do_kv_cache_update?)
→ Stimmt der Decode-Output?

### Test 3: Decode Step 2+ verifizieren
Ab dem 2. Decode-Token wächst die Sequenz.
→ Wird neues K/V korrekt zum Cache hinzugefügt?
→ Stimmt seq_len für den Decode-Kernel?
→ Liest der Kernel alle bisherigen Tokens?

### Test 4: KV Cache Dump
Nach einigen Decode-Steps:
- Dumpe Cache-Block für Layer 0
- Unpacke manuell (Python)
- Vergleiche gegen die originalen K/V Werte
→ Wenn Mismatch: Pack/Write Bug
→ Wenn Match: Decode/Read Bug

## Wie testen

```bash
# Container mit gemounteten Dateien + MQ_DEBUG=1
podman run -d --name mq-test \
  -v multiquant_attn.py:/.../multiquant_attn.py:ro \
  -v triton_mq_fused_decode.py:/.../triton_mq_fused_decode.py:ro \
  -v kernels/turboquant:/opt/tq_build:ro \
  -e MQ_DEBUG=1 \
  vllm-multiquant vllm serve ... --kv-cache-dtype tq4 --enforce-eager

# Request senden
curl localhost:8011/v1/completions -d '{"prompt":"3+4=","max_tokens":5}'

# Logs analysieren
podman logs mq-test 2>&1 | grep MQ_DEBUG
```
