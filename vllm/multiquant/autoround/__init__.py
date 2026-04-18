# SPDX-License-Identifier: Apache-2.0
"""AutoRound RTN adapter for MultiQuant framework.

Fast INT4 quantization at load time using AutoRound's --iters 0 mode.
Requires minimal calibration data (128 samples, auto-downloaded).
Output is Marlin-compatible GPTQ format.

Comparison with Archer (calibration-free):
  Archer:  No dataset, analytical centroids, seconds, cos ~0.79 (3bit)
  RTN:     128 samples, optimized rounding, 2-3 min (7B), better quality
"""
