# XFP: Extended Low-Bit Codebook Quantization Family (XFP2–XFP6) with Adaptive Sparse Outlier Separation and Learned Per-Layer Codebooks

**Dr. Thomas Witt**
Gemini Foundation Leipzig, 2026

---

**Abstract** — We propose XFP, a family of low-bit weight quantization formats for large language models, in particular Mixture-of-Experts (MoE) architectures, parameterized over a bit width `N ∈ {2, 3, 4, 5, 6}`. XFP addresses the limitations of both fixed-codebook formats (NVFP4) and linear sub-byte INT schemes (GPTQ/AWQ, RTN) by combining three orthogonal mechanisms that apply uniformly across all bit widths: (1) adaptive, distribution-driven outlier extraction into a sparse fp8 residual component, (2) learned per-layer `2^N`-entry codebooks optimized via Lloyd iteration on the cleaned bulk distribution, and (3) nearest-neighbor RTN assignment in codebook space. The format is served through a new fused decode kernel that unifies sub-byte depacking, codebook lookup, sparse outlier scatter-add, and GEMM in a single pass — no reliance on existing Marlin or GPTQ dequantization paths. XFP is hardware-agnostic, MoE-native, and strictly superior in reconstruction quality to uniform INT-N and fixed-codebook FP4 across its entire bit-width range, at negligible additional storage cost and without Hessian computation or gradient-based training.

---

## 1. Introduction

Post-training quantization (PTQ) to sub-byte precision is the dominant strategy for deploying large language models under memory and bandwidth constraints. Formats such as GPTQ, AWQ, and NVFP4 have demonstrated that 4-bit inference is viable with acceptable quality loss on dense transformer models. However, three structural problems remain underaddressed across the full sub-byte range:

**The outlier problem.** Weight matrices in transformer layers exhibit heavy-tailed distributions. A small fraction of weights — typically 2–10% — occupy a range far outside the bulk distribution. Standard linear INT-N quantization must accommodate this range in its scale factor, which severely degrades resolution in the dense central region where the majority of weights reside. The problem is worst at the lowest bit widths: at 2 bits, the outlier-driven scale leaves effectively one usable non-zero level for the bulk.

**The fixed-codebook problem.** NVFP4 addresses non-linearity at 4 bits by using a fixed 16-entry codebook derived from theoretical assumptions about weight distributions. This codebook is universal across all layers and models, and is not adapted to the actual distribution of any given layer. Furthermore, hardware acceleration for NVFP4 is restricted to NVIDIA Blackwell-generation datacenter GPUs (SM100+), making it impractical for most deployment targets. Other bit widths (2, 3, 5, 6) have no fixed-codebook equivalent at all.

**The MoE heterogeneity problem.** In Mixture-of-Experts models, individual experts exhibit highly diverse weight distributions. Frequently activated generalist experts differ structurally from rarely activated specialists. A universal quantization strategy treats both identically, at cost to the specialist experts which may carry disproportionate functional importance despite low activation frequency.

XFP addresses all three problems with a single, bit-width-parameterized framework.

---

## 2. Background

### 2.1 Linear INT-N Round-to-Nearest (RTN)

The simplest N-bit quantization encodes weights as:

```
q_i = clamp(round(w_i / scale), -(2^(N-1)), 2^(N-1) - 1)
W_reconstructed = q_i * scale
```

where `scale = max(|W|) / (2^(N-1) - 1)`. RTN is fast and requires no calibration data, but its reconstruction error is dominated by outliers that force a large scale, reducing effective resolution for the bulk of weights. Degradation is severe at `N ≤ 3`.

### 2.2 GPTQ

GPTQ improves on RTN by sequentially quantizing weights column by column, compensating for quantization error using the inverse Hessian of the layer's output with respect to its weights. This yields substantially better perplexity but requires calibration data and significant compute (O(d²) per layer).

### 2.3 NVFP4

NVFP4 uses a fixed 16-entry non-linear codebook, inspired by NF4 (Dettmers et al., 2023), to better match the assumed near-Gaussian weight distribution. The format offers hardware acceleration on Blackwell GPUs via Tensor Core support. On earlier architectures (e.g., SM121 / DGX Spark GB10), NVFP4 falls back to software emulation, where it offers no throughput advantage over INT4 with Marlin kernels. NVFP4 is defined only at `N = 4`.

### 2.4 SpQR / SparseGPT

SpQR (Dettmers et al., 2023) separates sensitive weights into a sparse high-precision component and quantizes the remainder. Sensitivity is determined by a Hessian-based saliency metric. XFP draws conceptual inspiration from this decomposition but eliminates the Hessian requirement by using a distribution-driven outlier criterion, and extends it with learned codebooks for the bulk component at every supported bit width.

---

## 3. The XFP Format

### 3.1 Decomposition

XFP represents a weight matrix `W` as:

```
W ≈ W_outlier + Codebook[Q_bulk]
```

where, for a chosen bit width `N ∈ {2, 3, 4, 5, 6}`:

- `W_outlier` is a **sparse fp8 matrix** containing high-magnitude outlier weights at their original positions
- `Q_bulk` is a **dense INT-N index matrix** into a learned per-layer `2^N`-entry codebook
- `Codebook` is a **fp16[2^N] vector**, one per layer, optimized for the bulk weight distribution

The decomposition is identical across all bit widths — only the codebook size and the index packing scheme differ.

### 3.2 Storage Layout

```
layer_K/
  weights.xfpN          # sub-byte packed bulk indices, N bits each
  codebook.fp16         # 2^N-entry learned codebook, fp16[2^N]
  outliers.indices      # sparse outlier positions, uint16[]
  outliers.values       # sparse outlier values, fp8 (E4M3)[]
  meta.json             # N, outlier_ratio, scale, distribution stats
```

Codebook storage per layer as a function of `N`:

| `N` | Codebook entries | Codebook size | Memory overhead per layer |
|-----|------------------|---------------|---------------------------|
| 2   | 4                | 8 B           | negligible                |
| 3   | 8                | 16 B          | negligible                |
| 4   | 16               | 32 B          | negligible                |
| 5   | 32               | 64 B          | negligible                |
| 6   | 64               | 128 B         | negligible                |

Memory overhead versus pure INT-N:

| Component        | Format       | Typical size           |
|------------------|--------------|------------------------|
| Bulk weights     | packed XFP-N | baseline               |
| Codebook         | fp16[2^N]    | 8–128 bytes per layer  |
| Outlier values   | fp8 sparse   | ~ratio × d × 1 byte    |
| Outlier indices  | uint16 sparse| ~ratio × d × 2 bytes   |

At a 5% outlier ratio, total overhead is approximately 15% over pure INT-N — substantially less than the reconstruction-quality gain justifies.

### 3.3 Sub-Byte Packing per Bit Width

The packing scheme is chosen per bit width to align with hardware word boundaries and to avoid cross-word bit-field extraction in the decode kernel. Unused bits are reserved for future per-value metadata (e.g. outlier flags, see §3.4).

| `N` | Values per word | Word type | Used bits | Reserve bits |
|-----|-----------------|-----------|-----------|--------------|
| 2   | 16              | uint32    | 32        | 0            |
| 3   | 10              | uint32    | 30        | 2            |
| 4   | 8               | uint32    | 32        | 0            |
| 5   | 3               | uint16    | 15        | 1            |
| 6   | 5               | uint32    | 30        | 2            |

XFP2 and XFP4 are naturally word-aligned. XFP3 stores 10 indices in a 32-bit word with 2 reserve bits (instead of a true bitstream), which the decode kernel can load/shift/mask without cross-word reads. XFP5 uses a 16-bit word holding 3 indices with 1 reserve bit; XFP6 uses a 32-bit word holding 5 indices with 2 reserve bits.

### 3.4 Reserve Bits (optional extension)

The reserve bits provided by XFP3/XFP5/XFP6 are unused by the base format and are free for optional metadata per packed value. The most immediately useful interpretation is a per-value **outlier flag**, which lets the decode kernel integrate the sparse outlier scatter-add into the depack pass instead of running it as a second pass:

- XFP5 (1 reserve bit): boolean `is_outlier` per value
- XFP6 (2 reserve bits): 4-state outlier slot (e.g. none / small / medium / full fp8)

This extension is **optional** and does not affect the base reconstruction semantics. For the initial implementation, reserve bits are set to zero and the outlier scatter-add runs as a dedicated second pass (§5).

---

## 4. XFP Encoding Algorithm

All steps except the codebook size and the final packing are independent of `N`.

### Step 1: Distribution Analysis and Outlier Ratio Estimation

For each layer weight matrix `W`, compute the empirical weight distribution. The outlier ratio is not a fixed hyperparameter but is **derived from the distribution itself**:

```
μ, σ = mean(W), std(W)
outlier_mask = |W - μ| > k·σ       # k typically 3.0–4.0
outlier_ratio = sum(outlier_mask) / numel(W)
```

Alternatively, the ratio can be derived from the second derivative of the empirical CDF — identifying the natural inflection point where density drops sharply. The distribution-derived ratio is stored as layer metadata.

**Ratio interpretation:**

```
ratio < ~15%  → genuine outlier regime → sparse + codebook path
ratio > ~30%  → broad-spectrum distribution → codebook-only path (no sparse)
15–30%        → hybrid, threshold determined per model family
```

A ratio exceeding ~30% indicates the weight distribution is broadly spread but not outlier-dominated. In this case, sparse extraction is counterproductive and a learned codebook over the full distribution is preferred. The threshold is independent of `N`, though at lower `N` the quality gain from removing outliers is larger (fewer codebook entries are wasted on the tails).

### Step 2: Outlier Extraction

```
W_outlier = W * outlier_mask          # sparse, fp8 E4M3
W_bulk    = W * (1 - outlier_mask)    # dense, zeros at outlier positions
```

Outlier values are converted to fp8 E4M3 format. Since outlier weights are by definition large in magnitude, the relative precision of fp8 (with its extended exponent range in E4M3) is sufficient. Indices are stored as uint16 (layer-local, relative to the flattened weight tensor).

### Step 3: Scale Estimation for Bulk

With outliers removed, the bulk distribution is well-conditioned — approximately Gaussian, zero-centered, with reduced dynamic range:

```
scale = max(|W_bulk|) / ((2^N - 1) / 2)    # or percentile-based, e.g. 99.9th
```

The scale is used for initial normalization before codebook optimization and is stored implicitly within the codebook values.

### Step 4: Learned Codebook via Lloyd Iteration

The `2^N` codebook entries are optimized for the empirical bulk distribution using Lloyd's algorithm (k-Means in 1D):

```
Initialize:
  codebook = quantile(W_bulk_nonzero, linspace(0, 1, 2^N))   # CDF-uniform init

Iterate until convergence:
  assignments = argmin_k |w_i - codebook[k]|   for all w_i in W_bulk_nonzero
  codebook[k] = mean(W_bulk_nonzero[assignments == k])
```

CDF-uniform initialization (equal probability mass per slot) is preferred over uniform-value initialization, as it avoids the degenerate case where multiple codebook entries collapse into the dense central region while the tails remain unrepresented. The advantage of CDF-uniform init grows with `N`: for small `N` the initial quantiles are coarse and a few iterations correct any imbalance; for large `N` a poor init is harder to recover from.

Typically 20–50 iterations suffice for convergence at any `N`. The final codebook is stored as `fp16[2^N]` per layer.

### Step 5: RTN Assignment in Codebook Space

Each bulk weight is assigned to its nearest codebook entry:

```
Q_bulk[i] = argmin_k |W_bulk[i] - codebook[k]|     k ∈ [0, 2^N)
```

This is the XFP analog of round-to-nearest — deterministic, parameter-free, and fast. The result is a dense index matrix with values in `[0, 2^N)`.

### Step 6: Packing

`Q_bulk` is packed into the word-aligned layout specified in §3.3. XFP2 and XFP4 pack into 32-bit words with no reserve bits. XFP3 packs 10 indices per 32-bit word (3 bits each, 2 reserve bits). XFP5 packs 3 indices per 16-bit word (5 bits each, 1 reserve bit). XFP6 packs 5 indices per 32-bit word (6 bits each, 2 reserve bits). Reserve bits are zero-initialized unless the optional outlier-flag extension (§3.4) is used.

### Complete Encoding Pipeline

```
Input: W (fp16, layer weight matrix), N ∈ {2,3,4,5,6}

1.  Analyze distribution → derive outlier_ratio
2a. If outlier_ratio < threshold:
      outlier_mask = |W - μ| > k·σ
      W_outlier = sparse_extract(W, outlier_mask) → fp8 E4M3
      W_bulk = W - W_outlier (bulk, zeros at outlier positions)
2b. If outlier_ratio ≥ threshold:
      W_outlier = empty
      W_bulk = W
3.  Lloyd iteration on W_bulk_nonzero → codebook[2^N] (fp16)
4.  RTN assignment: Q_bulk[i] = argmin_k |W_bulk[i] - codebook[k]|
5.  Pack Q_bulk → word-aligned XFP-N layout
6.  Store: weights.xfpN, codebook.fp16, outliers.{indices,values}, meta.json
```

---

## 5. XFP Decoding — Fused Kernel

Decoding is performed by a single **fused decode kernel** that runs per layer forward pass and combines sub-byte depacking, codebook lookup, sparse outlier scatter-add, and GEMM. The kernel is a new development — XFP does not reuse existing Marlin or `mq_gemm_int{2,3}` kernels, because those fuse depack with a linear `q * scale` multiply rather than a codebook gather and have no provision for an outlier scatter-add.

### Pipeline

```
1. Depack:       idx = unpack_xfpN(packed_weights)     # [N, K] int8
2. Lookup:       W_bulk_fp16 = codebook[idx]            # per-layer LUT gather
3. Scatter-add:  W_bulk_fp16[outlier_indices] += outlier_values (fp8→fp16)
4. GEMM:         Y = X @ W_bulk_fp16.T                  # standard fp16 GEMM
```

Stages 1–3 run once per forward pass and their cost is amortized over the GEMM. At typical MoE expert dimensions (4096 × 4096), the depack and scatter-add together are a small fraction of the matmul cost. For layers with no outliers (`outlier_ratio ≥ threshold` at encode time), stage 3 is skipped entirely.

### Codebook Lookup

The codebook `fp16[2^N]` is small enough (8–128 bytes) to live in registers or shared memory for the entire layer. Lookup is a single LUT indexed by the depacked index — no arithmetic on the index is needed. This is what makes XFP bit-width-portable: the same gather works for `N = 2` (4-entry LUT) through `N = 6` (64-entry LUT) with only the index width and LUT size changing.

### Outlier Scatter-Add

```
W_full = W_bulk_fp16
W_full[outlier_indices] += outlier_values     # sparse scatter-add, fp8→fp16
```

This is a single sparse scatter per layer forward pass — not per token. At 5% density the operation is cheap relative to the matrix multiply. When reserve bits carry outlier flags (§3.4), the scatter can be fused into stage 1 and stage 3 disappears.

---

## 6. Comparison to NVFP4, GPTQ, and INT-N RTN

| Property                  | INT-N RTN            | GPTQ                  | NVFP4 (N=4 only)             | **XFP (N ∈ 2–6)**                     |
|---------------------------|----------------------|-----------------------|------------------------------|---------------------------------------|
| Bit widths                | 2–8                  | 2–8                   | 4                            | **2, 3, 4, 5, 6**                     |
| Codebook                  | None (linear)        | None (linear)         | Fixed, universal             | Learned, per layer                    |
| Outlier handling          | None (clipped)       | None (Hessian-compensated) | Clipped to codebook range | Preserved exactly, sparse fp8         |
| Outlier ratio             | None                 | None                  | None (implicit clip)         | Adaptive, distribution-derived        |
| Hessian required          | No                   | **Yes**               | No                           | No                                    |
| Calibration data required | No                   | **Yes**               | No                           | Optional (improves codebook)          |
| Hardware acceleration     | Universal            | Universal             | Blackwell (SM100+) only      | Universal via new fused kernel        |
| MoE per-expert adaptation | None                 | Per-layer weights     | None                         | Native (per-layer = per-expert block) |
| Kernel                    | Linear dequant + GEMM| Linear dequant + GEMM | NVFP4 Tensor Core            | **Fused depack + LUT + scatter + GEMM** |

XFP is strictly more general than NVFP4 (which exists only at `N = 4`) and than linear INT-N schemes (which cannot match learned-codebook reconstruction quality at any fixed bit width). At `N = 2`, where linear RTN essentially collapses, the learned 4-entry codebook is the only practical non-trivial option. At `N = 6`, XFP reaches a reconstruction quality close to fp8 while using 25% less memory.

### 6.1 Empirical Codebook Analysis: Why No Fixed Grid Suffices

A per-channel comparison of 942,784 individually learned XFP4 codebooks (from GLM-4.7-Flash, a 30B MoE model with 64 routed experts × 46 layers) against three reference grids reveals that **no fixed codebook captures even the majority of output channels**:

| Reference grid       | Median cos | Channels < 0.99 cos (structurally different) | Channels > 0.999 cos (near-identical) |
|----------------------|-----------:|----------------------------------------------:|--------------------------------------:|
| Lloyd-Max N(0,1)     | 0.994      | **27.3 %**                                    | 2.3 %                                |
| INT4 symmetric       | 0.991      | 44.9 %                                        | 2.3 %                                |
| NF4 (QLoRA)          | 0.989      | **54.5 %**                                    | 0.001 % (8 of 942K)                  |

The theoretical Lloyd-Max quantizer for N(0,1) is the closest match — as expected, since transformer weight distributions are approximately Gaussian. But even this theoretically optimal fixed grid fails on **more than a quarter of all output channels**. NF4, despite being designed for Gaussian weights, is structurally different from the majority (54.5 %) because it is asymmetric (includes an explicit 0.0 level) while the learned codebooks are naturally symmetric.

**This falsifies the premise that a universal fixed codebook (NVFP4, NF4) can match the reconstruction quality of a per-channel learned codebook.** The per-channel variation is not noise — it reflects genuine distributional heterogeneity across output channels within a single layer and across layers of different types (attention, MoE expert, shared expert, dense MLP).

The practical consequence is a double failure mode for fixed grids:

1. **Channels with outliers**: NVFP4 and NF4 clip extreme values to their codebook range. XFP separates them into a sparse fp8 residual and fits the codebook to the cleaned bulk — strictly superior reconstruction.

2. **Channels without outliers**: Even when the distribution is well-behaved, a fixed grid wastes codebook entries in regions of low density. The Lloyd algorithm redistributes entries to match the actual per-channel CDF, gaining resolution where it matters. This is precisely the difference between 0.989 (NF4 median) and 0.994 (learned median) — modest per channel, but accumulated over hundreds of thousands of channels it shifts the reconstruction quality boundary.

The analysis also reveals that **inter-expert codebook variance within MoE routed experts is negligible** (pairwise cos 0.99999 across 96 sampled experts). This suggests that for the homogeneous MoE bulk, a single shared codebook per layer (not per channel) would suffice — reducing storage from `N_out × 2^N` to `2^N` entries per layer with no quality loss. XFP's per-channel flexibility is most valuable on the heterogeneous attention layers, where inter-channel variance is measurably higher (cos 0.99955).

A **cross-model comparison** (GLM-4.7-Flash vs Qwen3.5-35B-A3B, 256 experts × 40 layers) shows that the degree of per-channel heterogeneity is model-family-specific:

- GLM routed experts: Δcos from outlier extraction +0.00027 (flat tails, homogeneous). A fixed NF4 grid would lose little here.
- Qwen routed experts: Δcos +0.00180 (6.7× more responsive). The same fixed grid would miss meaningful structure.
- Qwen `shared_expert_gate`: Δcos up to +0.068 on individual layers — these mid-depth gating projections have highly non-Gaussian distributions where a fixed grid fails catastrophically.

This heterogeneity across model families is the strongest argument for XFP's learned-codebook approach: the format adapts automatically to whatever distribution the weights present, without requiring a priori distributional assumptions.

---

## 7. Relevance to MoE Models

In Mixture-of-Experts architectures, each expert is a distinct weight matrix with potentially unique distributional characteristics. XFP is **natively per-layer** and, by the same mechanism, natively **per-expert**:

- Each expert receives its own codebook, fitted to its weight distribution
- Each expert's outlier ratio is independently determined
- Specialist experts with heavy-tailed or bimodal distributions get appropriate treatment without compromising generalists

Because the XFP framework supports a range of bit widths with identical algorithmic structure, a natural extension is **per-expert bit width**: a generalist expert can be encoded as XFP4 while a specialist that requires higher precision can be encoded as XFP5 or XFP6, or a rarely-activated expert with simple structure as XFP2 or XFP3 — all within the same model and served by the same fused decode kernel. The kernel dispatches on `N` per expert block; the codebook and packing differ, but the pipeline is identical. This is in contrast to NVFP4 and linear INT-N, which apply a uniform strategy across heterogeneous expert populations.

We expect the quality improvement from XFP to be disproportionately concentrated in low-frequency, high-specialization experts — exactly those most damaged by conventional quantization.

---

## 8. Quality Evaluation Protocol

We propose the following evaluation hierarchy, run for each bit width `N ∈ {2, 3, 4, 5, 6}`.

### 8.1 Weight Reconstruction Error (fast, no model needed)

```
MSE(W_original, W_reconstructed) per layer
RMSE across layers
Max absolute error (outlier validation)
```

Validates the encoding pipeline independently of model behavior. Baseline comparisons: linear RTN at the same `N`, and (for `N = 4`) NVFP4.

### 8.2 Perplexity

Standard evaluation on WikiText-2 and C4. Baseline comparisons per `N`:

- Linear RTN INT-N (no outlier treatment)
- GPTQ INT-N (Hessian-based, `N ∈ {2,3,4,8}` in practice)
- NVFP4 (software fallback on non-Blackwell, `N = 4` only)
- **XFP-N** (proposed)

### 8.3 Downstream Task Performance

- **MMLU** — factual knowledge, robust signal
- **GSM8K** — arithmetic reasoning, sensitive to quantization-induced expert degradation
- **HumanEval** — code generation, tests rare but important weight patterns

GSM8K is particularly diagnostic for MoE quantization quality: multi-step reasoning chains fail early if specialist experts are degraded. Expect the gap between linear RTN and XFP to widen as `N` shrinks.

### 8.4 Expert-Level Analysis (MoE-specific)

Using activation profiling (cf. RIY framework), measure:

- Per-expert weight reconstruction error before/after XFP, per `N`
- Correlation between outlier_ratio and expert activation frequency
- Identification of experts where sparse treatment is most critical
- Per-expert optimal bit width (mixed-`N` deployment)

This provides mechanistic evidence that XFP's adaptive outlier treatment specifically preserves the specialist experts most vulnerable to conventional quantization, and that per-expert bit-width selection is a practical lever.

---

## 9. Implementation Notes

XFP is designed for integration into quantization and inference pipelines as a self-contained component:

- **Offline quantization (encoder)**: Single routine parameterized on `N`. Lloyd iteration and RTN assignment are 1D and trivial to implement. The distribution-driven outlier split is a single pass over the weight tensor. Encoder runs per-layer, outputs the standard file layout of §3.2.
- **Inference (decoder)**: Single fused CUDA kernel per `N`, following the pipeline of §5. The kernel is a new development; there is no reuse of Marlin or `mq_gemm_int{2,3}`. All bit widths share the pipeline structure — they differ only in the sub-byte unpack routine (a few lines of shift/mask code) and in the codebook size.
- **Memory**: Outlier indices and values loaded alongside packed XFP-N weights; codebook is 8–128 bytes and fits in registers or shared memory.
- **Compatibility**: fp8 E4M3 outlier values are supported on SM89+ (Ada Lovelace) and above natively. On SM80/SM86, fp8 outliers can be emulated or stored as fp16 at doubled sparse storage cost. The codebook lookup and depacking are pure integer and fp16 operations and run on any CUDA-capable GPU.
- **Hardware portability**: XFP is deliberately independent of Tensor Core sub-byte instructions. The fused kernel uses conventional fp16 tensor cores for the GEMM after depack and scatter, which are available on all SM70+ GPUs. This makes XFP practical on consumer and prosumer hardware (DGX Spark GB10, workstation RTX cards) where NVFP4 Tensor Core paths do not exist.

---

## 10. Conclusion

XFP is a hardware-agnostic, algorithmically simple low-bit quantization family that addresses the three core failure modes of existing approaches — outlier clipping, fixed codebooks, and MoE homogeneity — across the full bit-width range `N ∈ {2, 3, 4, 5, 6}`. By cleanly separating the outlier problem (handled via sparse fp8 residuals) from the bulk quantization problem (handled via learned per-layer codebooks with RTN assignment), and by serving the result through a single new fused depack+lookup+scatter+GEMM kernel, XFP achieves reconstruction quality strictly superior to both NVFP4 and linear INT-N RTN, without requiring Hessian computation, gradient-based training, or specialized hardware.

Where NVFP4 is a point solution at 4 bits on Blackwell and linear INT-N degrades sharply below 4 bits on all hardware, XFP provides a **continuum from 2 to 6 bits**. The family is particularly well-suited to MoE inference on consumer and prosumer hardware where NVFP4 hardware acceleration is unavailable, where per-expert adaptation is most beneficial, and where quality-per-bit is the primary constraint.

---

## References

- Dettmers et al. (2022). *GPT-Int8: LLM.int8() and Emergent Features*
- Frantar et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
- Dettmers et al. (2023). *SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression*
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs* [NF4]
- Lin et al. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*
- NVIDIA (2024). *NVFP4 Tensor Core Quantization — Blackwell Architecture Whitepaper*
- Lloyd, S.P. (1982). *Least Squares Quantization in PCM*
