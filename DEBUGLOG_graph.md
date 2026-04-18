# CUDA Graph Memory Debug

## Root Cause
pack_wht allocates 104 MB temporaries per call (B=256, nkv=20).
During graph capture: 35 graphs × 47 layers × 2 KV = 3290 calls → **504 GB** pool.
FP8 uses 0.68 GB total (single C kernel, zero allocs).

## Fix Options

### Option A: Pre-allocated pack buffer (minimal change)
Pre-allocate output buffer once, write qs/qr/gamma directly via indexing.
Replace `torch.cat([qs, qr, gamma_bytes])` with indexed writes.
Saves ~50% of allocs (the cat is the biggest).

### Option B: Fused CUDA pack kernel (best perf)
Single kernel: wht_forward + amax + quantize + bitpack.
Zero Python-side allocs. Like `reshape_and_cache_flash` for FP8.
Requires new CUDA kernel `tq_wht_pack.cu`.

### Option C: Skip graph capture for KV write
`forward_includes_kv_cache_update = False` for decode.
KV write runs outside graph (eager). Only decode is graphed.
But: slot_mapping problem returns for prefill (known fix needed).

### Measurements
- FP8 graph capture: 0.68 GB, 8 seconds
- WHT graph capture: >50 GB, OOM
- pack_wht per call: 104 MB peak (B=256×20 vectors)
- unpack_wht per call: 49 MB peak
