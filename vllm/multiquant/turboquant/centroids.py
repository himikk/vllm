# SPDX-License-Identifier: Apache-2.0
"""Compat shim — delegates to shared/centroids.py."""

from vllm.multiquant.shared.centroids import (  # noqa: F401
    get_boundaries,
    get_centroids,
    solve_lloyd_max,
)
