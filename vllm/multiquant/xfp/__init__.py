# SPDX-License-Identifier: Apache-2.0
"""XFP: Extended Low-Bit Codebook Quantization Family (XFP2–XFP4 in v1).

Quant-on-load pipeline:
  BF16 weight [N_out, K] → Lloyd-optimal per-channel codebook + packed indices
  → fused depack + LUT + GEMM kernel at inference.

See XFP.PAPER.md for the mathematical framework. Dispatched via the central
MultiQuantPolicyRegistry in vllm/multiquant/policy.py — user activates through
any dispatcher-aware quant_method string (autoround_rtn or multiquant) and sets
per-class target via --weight-dtype-* xfp{2,3,4}.
"""
