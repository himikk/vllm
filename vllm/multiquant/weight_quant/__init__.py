# SPDX-License-Identifier: Apache-2.0
"""Archer: MultiQuant online weight quantization.

Load BF16/FP8 models and quantize to 2-4 bits at load time using
RotorQuant or TurboQuant. Decompress on-the-fly during inference.
"""
