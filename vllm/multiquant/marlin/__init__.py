# SPDX-License-Identifier: Apache-2.0
"""Marlin INT4 adapter for MultiQuant framework.

Thin wrapper around vLLM's existing Marlin GEMM kernel.
Used as reference/baseline for comparing MultiQuant quantizers.

NOTE: Marlin uses PRE-QUANTIZED models (GPTQ, AWQ, AutoRound).
It does NOT do online quantization like Archer (TQ/RQ).
This adapter just makes Marlin accessible through the MultiQuant
registry for unified benchmarking.
"""
