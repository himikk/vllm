# SPDX-License-Identifier: Apache-2.0
"""Shared routines for MultiQuant quantizers.

These are algorithm-agnostic building blocks used by TurboQuant and
RotorQuant. Future quantizers with different algorithms should NOT
be forced to use these — they exist because TQ and RQ share the same
underlying math (Lloyd-Max + QJL + bit-packing).
"""
