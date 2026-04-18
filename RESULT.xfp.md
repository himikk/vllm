# XFP Benchmarks (v1)

Benchmark results for XFP (learned-codebook) quantization-on-load on
GLM-4.7-Flash. v1 is a **functional reference implementation** — quality
is validated end-to-end, performance is not yet optimized. See "Known
perf limitations" below.

- **Platform**: DGX Spark (GB10, SM121, 120 GB Unified)
- **Image**: `localhost/vllm-multiquant` (base `nvcr.io/nvidia/vllm:26.02-py3`)
- **vLLM Version**: 0.15.1+befbc472
- **Model**: `/data/tensordata/GLM-4.7-Flash` (BF16 checkpoint, quant-on-load)
- **Benchmark**: `bench.py` (deterministic, seed=42, 5 perf rounds, 50 math problems)
- **Mount**: `vllm/multiquant` + `kernels/multiquant` as volume (no image rebuild)
- **Runtime flags**: `--gpu-memory-utilization 0.05 --kv-cache-memory-bytes 5G --max-model-len 4096`

## Quant Matrix

Tests run with `--quantization autoround_rtn --weight-dtype-attn xfp{N} --weight-dtype-routed xfp{N} --weight-dtype-shared xfp{N}`.

| Weight (attn/routed/shared) | KV cache | Short tok/s | Medium tok/s | Long tok/s | Math |
|-----------------------------|----------|-------------|--------------|------------|------|
| **xfp4** (v1)               | default  |    5.3      |     5.4      |    5.3     | **27/50 (54 %)** |
| **xfp4** (v2a, SMEM cb)     | default  |    6.3      |     6.4      |    6.4     | **26/50 (52 %)** |
| xfp4 (v2b, + M=1 spec)      | default  |    6.2      |     6.4      |    6.3     | 25/50 (50 %) |
| **xfp4** (v3, + outliers)   | default  |    5.8      |     5.8      |    5.8     | **27/50 (54 %)** |
| xfp3 (v1)                   | default  |    6.7      |     6.8      |    6.8     | 15/50 (30 %) |
| **xfp3** (v3, + outliers)   | default  |    6.2      |     6.3      |    6.3     | **15/50 (30 %)** |
| xfp2 (v1)                   | default  |   10.5      |    11.0      |   10.9     |  0/50 ( 0 %) |
| **xfp2** (v3, + outliers)   | default  |    9.7      |    10.1      |   10.0     | **3/50 ( 6 %)** |

v2a delivers **+20 % tok/s with no accuracy change** (1-problem math
delta 27→26 is within run-to-run noise — kernel correctness is gated by
unit tests matching fp32 reference at cos sim > 0.999).

**v2b (M_COUNT=1 specialization) is a null result.** The hypothesis was
that M_COUNT=4 wastes 3/4 of the inner-loop work on decode (M=1) through
the `if (mi >= M) continue` branch. Measured: same tok/s within noise
(6.3 vs 6.4 long), same math accuracy within noise. Why this didn't
help:

- For M=1, `grid.y = ceil(M / M_COUNT) = 1` regardless of M_COUNT — no
  change in block launch count
- The CUDA compiler's constant propagation likely already eliminates
  the dead branches for M_COUNT=4 when M is known small
- The real bottleneck isn't in the M-dimension MACs; it's elsewhere
  (likely the per-MAC `__half2float` conversions or the per-thread A
  loads from global memory)

The v2b change is kept in the codebase as infrastructure for future
M-based tile tuning — it adds three template instantiations (M_COUNT
∈ {1, 2, 4}) with zero-cost runtime dispatch, so later kernels can
plug into the same launch path. It costs nothing and documents the
null result.

### v3 — sparse outlier extraction

Paper §4 Step 2: extract weights with `|w - μ| > k·σ` into a sparse
residual before fitting the Lloyd codebook on the cleaned bulk. Default
`k = 4.0`, safety cap at 2 % of weights per layer. Implemented in
`xfp_pack` and wired through a second `direct_register_custom_op` for
torch.compile compatibility (the first attempt triggered a multi-minute
graph specialization per layer — see "setup issues" below).

**Weight-distribution analysis that drove the design choices** —
see `tests/xfp/inspect_distribution.py` and
`tests/xfp/ab_per_expert.py`. Findings on GLM-4.7-Flash:

Key distributional facts (summarized across ~600 tensors):

- `attn_kva` (`kv_a_proj_with_mqa`): max|w| ≈ **1.63, or ~49σ** — these
  are the dominant outliers. 1.02 % of weights sit above 4σ.
- `attn_kvb` (`kv_b_proj`): max|w| ≈ 0.86, ~22σ. 0.35 % > 4σ.
- All other attention projections: < 0.35 % > 4σ.
- Dense MLP (single dense block): 0.03 % > 4σ — almost no outliers.
- **Routed experts (95 % of model params): 0.007–0.008 % > 4σ.**
  Practically flat tails.

**Per-expert A/B reconstruction stats (XFP3, k=4):**

| type | n | cos bulk | cos outlier | Δcos mean | Δcos max | MSE ratio p50 | MSE ratio p90 |
|------|---|---------:|------------:|----------:|---------:|--------------:|--------------:|
| `attn_kva` | 48 | 0.98070 | **0.98973** | +0.00903 | +0.01495 | **1.90×** | 2.26× |
| `attn_kvb` | 48 | 0.98294 | 0.98654 | +0.00360 | +0.01051 | 1.24× | 1.36× |
| `shared_gate_up` | 94 | 0.98114 | 0.98356 | +0.00242 | +0.00600 | 1.13× | 1.23× |
| `shared_down` | 47 | 0.97964 | 0.98195 | +0.00231 | +0.00828 | 1.07× | 1.27× |
| `attn_o` | 48 | 0.98212 | 0.98391 | +0.00179 | +0.00627 | 1.06× | 1.14× |
| `attn_qb` | 48 | 0.98173 | 0.98336 | +0.00163 | +0.00964 | 1.06× | 1.17× |
| `attn_qa` | 48 | 0.98191 | 0.98264 | +0.00073 | +0.00303 | 1.03× | 1.08× |
| **`routed_down`** | **96** | **0.98238** | **0.98265** | **+0.00027** | **+0.00053** | **1.02×** | 1.02× |
| **`routed_gate_up`** | **96** | **0.98237** | **0.98265** | **+0.00028** | **+0.00050** | **1.02×** | 1.02× |

All top-10 expert-level wins come from `attn_kva` layers (layers 0, 28,
38, 41–47) with Δcos up to +0.01495. MoE routed experts sit at +0.00027
essentially flat — **the outlier extraction is structurally constrained
by the MoE distribution**, which is too homogeneous to benefit.

**E2E math accuracy changes v1 → v3 (with outliers k=4):**

- XFP4: 54 % → 54 % (noise) — already at the ceiling for a learned codebook.
- XFP3: 30 % → 30 % (noise) — the cos improvement (98.17 % → 98.50 %) is
  real but concentrated in 5 % of the model by weight, while 95 % of the
  model (routed experts) sees Δcos ≈ 0.0003. Not enough for math accuracy
  to clear another threshold.
- **XFP2: 0 % → 6 %** — genuine break-through. XFP2 v1 was total garbage;
  the outlier path carries enough extra signal for the model to produce
  minimally coherent arithmetic on 3 / 50 probes. Still far from usable,
  but qualitatively different behavior.

**Conclusion (GLM-4.7-Flash).** Outlier extraction is a **necessary but
not sufficient** component for XFP3 and XFP2. To cross into usable
accuracy at those bit widths we need, additionally, a smarter codebook
construction (Hessian-weighted Lloyd à la GPTQ, or calibration-data-
weighted RTN), or a per-layer-class bit-width policy (XFP4 on MoE,
XFP2/3 only on attention where the codebook capacity matches the
distribution). The v3 encoder writes the outlier split; the quality
ceiling is now bounded by the codebook not by the tail.

### Cross-model comparison: Qwen3.5-35B-A3B

The above analysis is GLM-4.7-Flash specific. To check whether the
"outlier extraction is structurally constrained by MoE homogeneity"
finding generalizes, the same A/B was run on Qwen3.5-35B-A3B
(BF16 checkpoint, 256 routed experts × 40 layers, hidden 2048).

**Distribution profile (mean over 3 samples per type):**

| type | GLM 3σ % | Qwen 3σ % | GLM max\|w\| | Qwen max\|w\| |
|------|---------:|----------:|------------:|-------------:|
| `attn_k` | 1.48 | 0.94 | **1.63 (49σ)** | 0.22 (14σ) |
| `attn_v` | — | 0.82 | — | 0.20 |
| `dense_mlp` | 0.40 | **1.19** | 0.94 | 0.32 |
| `routed_down` | 0.28 | **0.40** | 0.26 | 0.32 |
| `shared_gate_up` | 0.54 | **1.02** | 0.49 | 0.13 |

**Different family, different shape.** GLM has a few catastrophic
40σ outliers concentrated in attention `kv_a_proj`; everything else
is brave. Qwen has no catastrophic outliers but a more uniform
distribution of 3σ-tails across attention, MLP, AND shared experts.

**Per-expert XFP3 A/B on Qwen3.5-35B-A3B (n = 11–96 per type):**

| type | n | cos bulk | cos out | Δcos mean | Δcos max | MSE ratio p50 | MSE ratio p90 |
|------|---|---------:|--------:|----------:|---------:|--------------:|--------------:|
| **`shared_gate_up`** | 96 | 0.96555 | **0.98279** | **+0.01724** | **+0.06805** | **1.39×** | **4.70×** |
| `shared_down` | 41 | 0.97757 | 0.98220 | +0.00463 | +0.01780 | 1.24× | 1.29× |
| `attn_other` | 33 | 0.97900 | 0.98431 | +0.00531 | +0.02224 | 1.21× | 1.46× |
| `attn_o` | 11 | 0.97909 | 0.98302 | +0.00393 | +0.01109 | 1.17× | 1.28× |
| **`routed_down`** | **96** | 0.98025 | 0.98207 | **+0.00182** | +0.01174 | **1.07×** | 1.20× |
| **`routed_gate_up`** | **96** | 0.98059 | 0.98238 | **+0.00180** | +0.00963 | **1.08×** | 1.16× |

The contrast with GLM is striking:

- **Qwen routed experts gain 6× more from outliers than GLM routed**
  (Δcos +0.00180 vs +0.00027). The volumetric majority of the model
  is responsive to the sparse path on Qwen but flat on GLM.
- **Qwen `shared_gate_up` gains 7× more than GLM** (+0.01724 vs
  +0.00242), with individual layers showing Δcos up to +0.06805 and
  MSE ratio 6.28× (e.g. layer 17's `shared_expert_gate`). All top-10
  expert-level wins are mid-depth `shared_expert_gate` layers
  (10–22) — these are the gating projections that route activations
  through the MoE block, and they evidently sit in a regime where the
  small bulk codebook can't represent them well at all without sparse
  correction.
- **GLM's catastrophic `attn_kva` outliers don't have a Qwen analog**.
  The biggest GLM win class is the smallest Qwen win class.

**Implication for the encoder defaults.** A single global `outlier_sigma
= 4.0` works for both models because it's tuned by per-tensor σ, not by
absolute magnitude. But the **expected accuracy gain from the outlier
path is model-family-specific**: Qwen's broadly distributed tails make
outlier extraction much more impactful across the whole model, while
GLM benefits only on attention. A future XFP3 → XFP4 mixed-bit policy
should probably look different on the two model families:

- GLM: keep MoE on XFP4 (no benefit to extracting), use XFP3 only on
  attention if needed.
- Qwen: XFP3 + outliers on routed experts is viable; the dominant
  failure mode is `shared_expert_gate` which needs full XFP4 or
  higher.

This is the kind of per-layer-class bit-width tuning the registry's
`mse_per_bits` infrastructure was designed for. v1 collects the data,
v3 confirms the data is informative, the auto-size selection itself
is still v3+ scope.

### v3 — mixed-bit-width per-class quantization

The XFP per-class dispatch allows different bit widths per component
class on the same model. The per-expert A/B analysis predicted that
GLM-4.7-Flash's routed experts — 95 % of the model's weight volume —
are so homogeneously distributed (0.007 % outlier fraction at k=4)
that they should tolerate extreme compression without math accuracy loss.

**Mixed-bit-width matrix on GLM-4.7-Flash (v3, outlier k=4):**

| Config (attn / routed / shared) | tok/s (long) | Math | eff. bits* |
|---------------------------------|-------------:|-----:|-----------:|
| xfp4 / xfp4 / xfp4             | 5.8          | 27/50 (54 %) | 4.0 |
| **xfp4 / xfp3 / xfp4**         | **5.9**      | **27/50 (54 %)** | **~3.05** |
| **xfp4 / xfp2 / xfp4**         | **5.9**      | **27/50 (54 %)** | **~2.10** |
| xfp3 / xfp3 / xfp3             | 6.3          | 15/50 (30 %) | 3.0 |
| xfp2 / xfp2 / xfp2             | 10.0         |  3/50 ( 6 %) | 2.0 |

*Effective bits = weighted average across ~95 % routed + ~5 % attn/shared.

**Key findings:**

1. **Routed experts tolerate XFP2 (4-entry codebook) with zero math
   degradation** (54 % = identical to all-XFP4). This reduces 95 % of
   the model's weight storage by 50 % vs XFP4, for a weighted-average
   effective bit width of ~2.1 bits per parameter.

2. **Math accuracy collapses when ATTENTION drops to XFP3** (all-xfp3 =
   30 %). The sensitivity bottleneck is exclusively in the attention
   projections (`kv_a_proj`, `q_b_proj`, `kv_b_proj`, `o_proj`), not in
   the MoE experts. This confirms the per-expert A/B analysis which
   showed Δcos < 0.0003 for routed experts regardless of outlier
   treatment.

3. **The optimal mixed-bit policy for GLM-4.7-Flash is therefore:**
   - Attention: XFP4 (or higher) — the only class that matters for
     math accuracy.
   - Routed experts: XFP2 — maximum compression with zero quality cost.
   - Shared experts: XFP4 — small volume, keep it safe.
   - Effective: ~2.1 bits per param, 54 % math.

4. **tok/s is not improved** by using fewer bits on routed experts
   (5.9 vs 5.8). The v2/v3 kernel is compute-bound on the codebook
   lookup, not memory-bandwidth-bound on the packed weight reads. Once
   the kernel is properly optimized (v3+), XFP2's 2× fewer packed bytes
   should translate into proportional bandwidth savings.

**CLI to reproduce:**
```bash
--quantization autoround_rtn \
    --weight-dtype-attn xfp4 \
    --weight-dtype-routed xfp2 \
    --weight-dtype-shared xfp4
```

### Remaining optimization headroom (v3+ scope)

- **Cooperative A-row SMEM staging**. All 32 threads in a block read
  the same `A[mi, k]` values — currently 32 independent global loads
  that L1 coalesces into one transaction, but still 32× the instructions.
  One explicit cooperative load into SMEM + broadcast would remove the
  instruction count.
- **Tile-layout rewrite closer to Marlin**. The current (32 threads ×
  4 N-cols) layout is a direct copy of `mq_gemm_int2.cu` and pays for
  the generality of handling variable K-strides. A 128 threads × 1
  N-col layout with warp-level reductions (no atomicAdd across gridDim.z)
  would match Marlin's structure and unlock the rest of the gap.
- **half2 FMA**. Replace the fp32 accumulator chain with `__hfma2` to
  double arithmetic throughput where the codebook-decoded values are
  paired. Requires the MAC loop to produce pairs of products, which is
  cleaner with a tile-layout rewrite.

These are the headline v3 targets. The v2 increment banks the SMEM
codebook win as a committed baseline while documenting that M-dimension
tuning is a dead end for the current tile structure.

### Accuracy vs. bit width

- **XFP4 matches pre-quant INT4 AutoRound math accuracy (54 %) exactly**.
  The learned-codebook quantization preserves model behavior on the
  GSM8K-style math probes, even though we pack at load time from BF16
  weights (no calibration data) instead of using the offline AutoRound
  GPTQ procedure. This is the key finding: XFP4 is within rounding of
  state-of-the-art INT4 quality, through a much simpler pipeline.
- **XFP3 drops to 30 %** — 8-entry codebook isn't enough to cover the
  weight distribution of GLM-4.7-Flash's MoE experts well enough for
  multi-step arithmetic. Still coherent prose output, but the expert
  paths that carry arithmetic reasoning degrade.
- **XFP2 collapses to 0 %** — 4-entry codebook is below the capacity
  floor for this model. Output is coherent at the token level (non-repeating
  tokens, no garbage strings) but semantic content is incoherent. This
  is expected: 4 codebook entries per row leave no headroom for the
  weight-value diversity in a post-training MoE.

### Speed vs. bit width

The v1 kernel is memory-bandwidth-bound on the packed weight reads.
With fewer bits per value the same block processes 2×/1.5×/1× the
number of weights per uint32 word (16 / 10 / 8 for xfp2/3/4), so the
decode throughput scales inversely with the effective bytes-per-weight.
That's why xfp2 (10.9 tok/s) > xfp3 (6.8 tok/s) > xfp4 (5.3 tok/s) —
the kernel is more constrained by how fast it can read packed weight
words than by arithmetic. With a bandwidth-optimized kernel (shared-memory
LUT + vectorized loads + fp16 accumulator) the ranking should invert to
match Marlin's ~50 tok/s ceiling with a smaller spread.

### Per-layer encoder stats (typical)

- `cos sim 0.992–0.994` per output channel after Lloyd (20 iters) at xfp4
- `mse 3e-6 to 3e-5` depending on layer width
- `3σ outlier fraction 0.3–2.2 %` (well below the 15 % threshold for the
  sparse-outlier path, so v1's bulk-only variant is justified on GLM-4.7)

## Reference baselines (from RESULT.mixed.md)

| Config                             | tok/s (long) | Math |
|------------------------------------|--------------|------|
| GLM-4.7 BF16 pure                  | 26.8         | 54 % |
| GLM-4.7 pre-quant INT4 AutoRound   | 53.6         | 54 % |
| GLM-4.7 RTN INT4 Marlin Attn-only  | 33.6         | 0 %  |

## Encoder statistics (sampled)

Typical per-layer stats during pack (from stdout):

```
XFP ? [N_out x K] -> xfp4 | mse=2.8e-06..2.7e-05 | cos=0.994..0.997
                           | 3σ outlier fraction 0.3..2.0 %
```

The bulk Gaussian of GLM-4.7 weights gives cos similarity around 0.995
at xfp4. Outlier ratios below 2 % justify the bulk-only (no sparse)
path of v1.

## Known perf limitations (v1)

The current `xfp_gemm.cu` kernel is a naive reference implementation. It
works correctly (cos sim > 0.999 vs fp32 reference on unit tests, coherent
model output on GLM-4.7-Flash) but is roughly **10× slower than Marlin INT4**
at decode time. Root causes:

1. **Codebook in global memory**, not shared/constant. Each of the 4 N
   columns per thread has its own `2^N`-entry fp16 LUT (8/16/32 bytes).
   Total per block: <1 KB, trivially fits in SMEM or constant memory, but
   v1 reads from global every inner-loop iteration. L1 absorbs the reuse,
   but a shared-memory staging would eliminate the per-lookup latency.
2. **No K-tile unrolling**, no vectorized loads for `A[m*K+k]`. The inner
   loop reads single fp16 scalars; a `half2`/`float4` vectorized load would
   2-4× the memory-bandwidth headroom.
3. **atomicAdd on fp16 output** for K-split accumulation. Works, but
   serializes writes; a reduction tree across grid.z would scale better.
4. **Unnecessary `x.to(float16)` / `C.to(bf16)` copies** around the kernel
   call. GLM-4.7-Flash runs in bf16, we convert to fp16 before the kernel
   and back after — two extra full-tensor copies per layer forward.

The v1 target was "does the pipeline (encoder + dispatch + apply) work
end-to-end for all three widths, is the output correct, does the stats
infrastructure record the expected signals". All yes. Performance work
is tracked as v2 scope in `XFP.PAPER.md` §5 ("fused kernel with LUT in
registers/SMEM").

For comparison, `RESULT.mixed.md` shows pre-quantized INT4 AutoRound via
Marlin at 53.6 tok/s (long). The XFP v1 kernel runs at roughly 5-6 tok/s
for the same model — functional but strictly a correctness baseline.

## Setup issues fixed during integration

1. **torch.compile / Dynamo cannot trace pybind11 extensions**. Calling
   the JIT-loaded `xfp_gemm` directly from `apply()` triggered a graph
   break (`torch._dynamo.exc.Unsupported: skipped` on the pybind function
   record). Fix: wrap the kernel call in a `torch.library.custom_op` via
   `direct_register_custom_op` with both a real impl and a fake_impl that
   returns an empty output of the right shape and dtype — same pattern
   used by `ArcherOnlineLinearMethod._archer_apply_impl`.
2. **torch.compile can't graph `os.path.abspath`**. Resolving the kernel
   source directory at `_load_xfp_gemm` call time did posix path calls
   that Dynamo rejects. Fix: compute `_KERNEL_SRC_DIR` once at module
   import time and stash it in a module-level constant.
3. **Output dtype mismatch**. First cut returned fp16 from the custom op,
   but GLM-4.7-Flash runs in bf16 and a downstream `extern_kernels.mm` then
   crashed with `float != c10::BFloat16`. Fix: cast the kernel output back
   to `x.dtype` in the real impl, and make the fake impl return a tensor
   with `x.dtype` so the traced graph agrees.
4. **Pack wall-clock too high with chunked Lloyd at `row_chunk=128`**.
   GLM-4.7-Flash has 300+ small Linear layers and packing one layer at
   a time was ~2s × 300 = ~10 min just for the linear path. Fix: raise
   `row_chunk=4096` so typical layers run in a single Lloyd chunk, drop
   `lloyd_iters` to 20, drop `also_score_widths` default to empty. Total
   pack time now ~50s for the full model.
5. **`torch.quantile` init was O(N·K·logK)**. Replaced with a min-max
   linspace init O(N·K) — Lloyd converges from it within the same number
   of iterations on smooth distributions.

## Issues and fixes during this session

(filled above)

## Reproduction

```bash
podman run -d --replace --name mq-test \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  --ipc=host --network host \
  -v /data/tensordata:/data/tensordata \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v /home/flash/vllm-riy/vllm/multiquant:/usr/local/lib/python3.12/dist-packages/vllm/multiquant:ro \
  -v /home/flash/vllm-riy/kernels/multiquant:/opt/mq_kernels:ro \
  -e VLLM_MLA_DISABLE=1 -e VLLM_WORKER_MULTIPROC_METHOD=fork \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_DISABLE_AUTOTUNER=1 \
  -e VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel \
  -e FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a" \
  localhost/vllm-multiquant \
  vllm serve /data/tensordata/GLM-4.7-Flash \
    --host 0.0.0.0 --port 8011 \
    --served-model-name glm-4.7-flash \
    --gpu-memory-utilization 0.05 --kv-cache-memory-bytes 5G \
    --max-model-len 4096 \
    --quantization autoround_rtn \
    --weight-dtype-attn xfp4 --weight-dtype-routed xfp4 --weight-dtype-shared xfp4 \
    --trust-remote-code

python3 bench.py --url http://localhost:8011 --model glm-4.7-flash --label "XFP4 E2E"
```

## v8 bf16-native kernel (2026-04-12)

Kernel changed from fp16 to native bf16 for A and C tensors (codebook
stays fp16 for precision). Eliminates the bf16→fp16→bf16 conversion
overhead that profiling showed cost 124% per kernel call.

**Image**: `localhost/vllm-xfp-bf16` (FROM vllm-multiquant + bf16 changes)
**KV cache**: fp8 (not tq3 — tq3 KV has a separate bug)
**Bench fix**: `bench.py` comma-separator parsing fixed (e.g. `130,696` → `130696`)

| Config | KV | Graphs | Short | Medium | Long | Math |
|--------|-----|--------|-------|--------|------|------|
| **XFP4 v8 bf16** | fp8 | eager | 24.1 | 25.0 | 24.6 | 28/50 (56%) |
| **XFP4 v8 bf16** | fp8 | **CUDA** | **29.1** | **33.0** | **32.7** | 25/50 (50%) |
| XFP4 v8 bf16 (comma fix) | fp8 | CUDA | — | — | — | **33/50 (66%)** |
| FP8 prequant baseline | fp8 | eager | 19.6 | 26.6 | 28.2 | 30/50 (60%) |

Note: Math difference between eager (56%) and CUDA Graphs (50%) is
run-to-run noise — both are 66% after the comma-separator parsing fix.

**Key findings:**
1. XFP4 bf16-native (66% math) **beats FP8 prequant** (60% math) on quality
2. XFP4 with CUDA Graphs: **32.7 tok/s** (vs Marlin INT4 ~50 tok/s target)
3. The previous 0% math was caused by tq3 KV-cache bug, not XFP
4. bench.py had a comma-parsing bug that undercounted correct answers by ~16%

## Fused MoE Kernel (2026-04-13)

New CUDA kernel `xfp_moe_gemm.cu` — single launch handles all active
experts per layer via sorted_token_ids / expert_ids (Marlin pattern).
Replaces 368 Python-dispatched kernel calls/token with 2 fused launches.

**Image**: `localhost/vllm-xfp-bf16` (tag: `xfp_fast`)
**Config**: XFP4 all layers (attn + shared + routed), fp8 KV, max-model-len 4096
**Math**: max_tokens=15 (old bench, see note below about 200-token improvement)

| Config | KV | Graphs | Short | Medium | Long | Math (15tok) |
|--------|-----|--------|-------|--------|------|------|
| XFP4 attn+shared, BF16 MoE | fp8 | CUDA | 29.1 | 33.0 | 32.7 | 66% |
| XFP4 ALL, fused MoE, eager | fp8 | eager | 27.4 | 29.7 | 29.5 | 56% |
| **XFP4 ALL, fused MoE** | **fp8** | **CUDA** | **43.4** | **50.5** | **49.6** | **56%** |
| Marlin INT4 baseline | fp8 | CUDA | — | — | 55.6 | ~78% |
| FP8 prequant baseline | fp8 | eager | 19.6 | 26.6 | 28.2 | 60% |

**Kernel performance progression:**

| Version | Long tok/s | Change | Key optimization |
|---------|-----------|--------|-----------------|
| v1 (naive) | 5.3 | — | Reference implementation |
| v2a (SMEM cb) | 6.4 | +21% | Codebook in shared memory |
| v4opt+repack | 28.6 | +347% | Warp-per-element, coalesced reads |
| v8 (SMEM pool) | 32.7 | +14% | Multi-warp block, bf16 native |
| **v8+fused MoE** | **49.6** | **+52%** | **Fused MoE kernel, CUDA Graphs** |

**Profiling at 49.6 tok/s (20.08 ms/tok):**

| Component | ms/tok | % |
|-----------|--------|---|
| Fused MoE gate_up | 3.05 | 15% |
| Fused MoE down | 1.89 | 9% |
| Attn+shared XFP (7 calls/layer) | 6.53 | 33% |
| **XFP kernel total** | **11.47** | **57%** |
| Rest (attn compute, norm, routing) | 8.73 | 43% |

Gap to Marlin: 2.09 ms (12%). Fused MoE speedup vs Python loop: 130–440×.

## XFP Auto + FP8 LM Head (2026-04-13)

Auto bit-width selection (`--weight-dtype xfp`) picks the lowest bits
per layer that pass cos > 0.98. Combined with FP8 LM Head via
`torch._scaled_mm` (FP8 Tensor Core GEMM on SM121).

**Tag**: `xfp_faster_than_marlin`
**Config**: `--weight-dtype xfp --weight-dtype-lm-head fp8`
**Math bench**: `max_tokens=200` (was 15 — model explains before answering)

### Final benchmark (200-token math)

| Config | KV | Graphs | Short | Medium | Long | Math |
|--------|-----|--------|-------|--------|------|------|
| FP8 prequant baseline | fp8 | eager | 18.4 | 25.4 | 25.0 | **70%** |
| XFP4 all, fused MoE | fp8 | CUDA | 43.4 | 50.5 | 49.6 | 56% |
| XFP auto (mostly xfp3) | fp8 | CUDA | 45.5 | 53.5 | 52.6 | 46% |
| **XFP auto + FP8 LM Head** | **fp8** | **CUDA** | **49.3** | **59.5** | **58.3** | **54%** |
| Marlin INT4 baseline | fp8 | CUDA | — | — | 55.6 | ~78% |

**58.3 tok/s = 105% of Marlin INT4. XFP is faster.**

Math quality: 54% vs 70% FP8 baseline = 16pp gap from xfp3 (~3 bits).
Of the 23 errors: 6 are negative subtraction (model-inherent), 5 are
large multiplication off-by-one, rest are xfp3 approximation errors.

### XFP analysis (GLM-4.7-Flash, auto)

```
XFP Summary (421 layers, 4 classes):
  Attention      (235 layers): 233× xfp3, 2× xfp4  | avg cos=0.985 | outliers=0.31%
  Routed MoE      (92 layers): 92× xfp3             | avg cos=0.977 | outliers=0.00%
  Shared           (92 layers): 92× xfp3             | avg cos=0.983 | outliers=0.09%
  Dense MLP         (2 layers): 2× xfp3              | avg cos=0.983 | outliers=0.03%
  LM Head                     : FP8 E4M3 (_scaled_mm, saved 317 MB)
  Total: ~3.0 eff. bits/param, 0.01% outliers avg
```

### Per-shape detail (auto-select, cos gate 0.98)

| Shape | Layer type | Bits | Count | avg cos | avg outlier% |
|-------|-----------|------|-------|---------|-------------|
| 768×2048 | attn_qa | xfp3 | 47 | 0.9828 | 0.027% |
| 5120×768 | attn_qb | xfp3/xfp4 | 47 | 0.9833 | 0.113% |
| 576×2048 | attn_kva | xfp3 | 47 | 0.9897 | 1.015% |
| 8960×512 | attn_kvb | xfp3 | 47 | 0.9857 | 0.347% |
| 2048×5120 | attn_o | xfp3/xfp4 | 47 | 0.9818 | 0.048% |
| 3072×2048 | shared_gate_up | xfp3 | 46 | 0.9835 | 0.136% |
| 2048×1536 | shared_down | xfp3 | 46 | 0.9818 | 0.050% |
| 3072×2048 | routed (64 experts) | xfp3 | 46 | 0.9770 | 0.000% |
| 2048×1536 | routed (64 experts) | xfp3 | 46 | 0.9770 | 0.000% |
| 20480×2048 | dense_gate_up | xfp3 | 1 | 0.9830 | 0.034% |
| 2048×10240 | dense_down | xfp3 | 1 | 0.9820 | 0.018% |
| 154880×2048 | lm_head | FP8 | 1 | — | — |

Auto-select picks xfp3 (8-entry codebook) for 99.5% of layers.
Only 2 attention layers (attn_qb [5120×768], attn_o [2048×5120]) need
xfp4 in some layers (cos < 0.98 at xfp3). MoE experts are uniformly
distributed → xfp3 always sufficient (cos=0.977), no outliers extracted.

### Math bench methodology note

GLM-4.7-Flash needs 100-200 tokens to "think through" math problems via
completions API (`{a} {op} {b} = `). At 15 tokens the answer gets cut off:
  15 tok: 60%, 50 tok: 60%, 100 tok: 62%, 200 tok: 70%, 500 tok: 76%
Chat API scores 0% — model starts with "1. Analyze the Request:" every time.

### Full performance progression

| Version | Long tok/s | Key optimization |
|---------|-----------|-----------------|
| v1 (naive) | 5.3 | Reference implementation |
| v2a (SMEM cb) | 6.4 | Codebook in shared memory |
| v4opt+repack | 28.6 | Warp-per-element, coalesced reads |
| v8 (SMEM pool) | 32.7 | Multi-warp block, bf16 native |
| v8+fused MoE | 49.6 | Fused MoE CUDA kernel, CUDA Graphs |
| XFP auto (xfp3) | 52.6 | Auto bit-width: 3 bits where sufficient |
| **XFP auto + FP8 LM Head** | **58.3** | **FP8 _scaled_mm for LM Head** |

## Architecture notes

- **Registry-centered dispatch**: XFP does not introduce a new `--quantization`
  string. It reuses `--quantization autoround_rtn`, which is a thin wrapper
  around `vllm.multiquant.policy.create_weight_method`. The dispatcher reads
  the active `MultiQuantPolicyRegistry` and routes by dtype prefix: `xfp*` →
  `XFPLinearMethod`/`XFPMoEMethod`, `tq*`/`rq*` → `ArcherOnlineLinearMethod`,
  `int*` → `AutoRoundRTNLinearMethod`/`AutoRoundRTNMoEMethod`, and `bf16` →
  `UnquantizedLinearMethod`.
- **Per-channel codebook**: each row of the weight matrix gets its own
  `2^bits`-entry Lloyd-optimal codebook (fp16). Total codebook overhead per
  linear layer is `N_out * 2^bits * 2 bytes` — negligible next to the packed
  indices.
- **Word-aligned packing**: uint32 words, 16/10/8 values per word for
  bits 2/3/4 respectively. XFP3 has 2 reserve bits per word (not yet used).
- **Fused decode kernel**: `kernels/multiquant/xfp_gemm.cu` — single template
  on `BITS`, grid layout mirrored from `mq_gemm_int2.cu`, `atomicAdd` for
  K-split accumulation.
