# SPDX-License-Identifier: Apache-2.0
"""Lloyd-Max optimal scalar quantizer (shared by TQ + RQ).

After rotating a d-dimensional unit vector by a random orthogonal matrix,
each coordinate approximately follows N(0, 1/d) for d >= 64.
We solve the Lloyd-Max conditions to find optimal centroids.

Based on: turboquant-pytorch/lloyd_max.py (Zandieh et al.)
"""

import math
from functools import lru_cache

import torch


def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(
        -x * x / (2 * sigma2)
    )


def solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d) distribution.

    Args:
        d: Vector dimension (determines variance = 1/d).
        bits: Number of quantization bits.
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance.

    Returns:
        centroids: Sorted tensor of 2^bits optimal centroids.
        boundaries: Sorted tensor of 2^bits - 1 decision boundaries.
    """
    from scipy import integrate

    n_levels = 2**bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x):
        return _gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])

        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
    ]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


# Precomputed centroids for common configs (avoids scipy at runtime)
_PRECOMPUTED = {
    # (d, bits): [centroids]  — computed via solve_lloyd_max
    (128, 1): [-0.07054, 0.07054],
    (128, 2): [-0.13353, -0.04001, 0.04001, 0.13353],
    (128, 3): [-0.19023, -0.11174, -0.05171, -0.01131,
                0.01131,  0.05171,  0.11174,  0.19023],
    (256, 1): [-0.04989, 0.04989],
    (256, 2): [-0.09441, -0.02829, 0.02829, 0.09441],
    (256, 3): [-0.13450, -0.07903, -0.03657, -0.00800,
                0.00800,  0.03657,  0.07903,  0.13450],
}


# ── WHT Block-Compression Centroids ──────────────────────────
# Lloyd-Max optimal centroids for N(0,1) distribution.
# Used with Walsh-Hadamard Transform (WHT) block compression.
# These are UNIVERSAL — same for all head_dim, all layers.
# Reference: github.com/animehacker/llama-turboquant

WHT_CENTROIDS = {
    # bits: [centroids] — Lloyd-Max optimal for N(0,1)
    1: [-0.7979, 0.7979],
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1519, -1.3439, -0.7560, -0.2451,
         0.2451,  0.7560,  1.3439,  2.1519],
    4: [-2.7326, -2.0690, -1.6181, -1.2563,
        -0.9424, -0.6568, -0.3881, -0.1284,
         0.1284,  0.3881,  0.6568,  0.9424,
         1.2563,  1.6181,  2.0690,  2.7326],
}

WHT_THRESHOLDS = {
    # bits: [decision boundaries between centroids]
    1: [0.0],
    2: [-0.9816, 0.0, 0.9816],
    3: [-1.7479, -1.0500, -0.5005, 0.0, 0.5005, 1.0500, 1.7479],
    4: [-2.4008, -1.8436, -1.4372, -1.0993, -0.7996, -0.5224, -0.2582,
         0.0, 0.2582, 0.5224, 0.7996, 1.0993, 1.4372, 1.8436, 2.4008],
}


def get_wht_centroids(bits: int) -> torch.Tensor:
    """Get WHT-mode centroids (N(0,1) Lloyd-Max, universal)."""
    if bits not in WHT_CENTROIDS:
        raise ValueError(f"No WHT centroids for {bits} bits. "
                         f"Available: {list(WHT_CENTROIDS.keys())}")
    return torch.tensor(WHT_CENTROIDS[bits], dtype=torch.float32)


def get_wht_thresholds(bits: int) -> torch.Tensor:
    """Get WHT-mode decision thresholds."""
    if bits not in WHT_THRESHOLDS:
        raise ValueError(f"No WHT thresholds for {bits} bits.")
    return torch.tensor(WHT_THRESHOLDS[bits], dtype=torch.float32)


# ── Original per-dimension centroids ─────────────────────────

@lru_cache(maxsize=32)
def get_centroids(d: int, bits: int) -> torch.Tensor:
    """Get precomputed Lloyd-Max centroids (cached)."""
    key = (d, bits)
    if key in _PRECOMPUTED:
        return torch.tensor(_PRECOMPUTED[key], dtype=torch.float32)
    centroids, _ = solve_lloyd_max(d, bits)
    return centroids


@lru_cache(maxsize=32)
def get_boundaries(d: int, bits: int) -> torch.Tensor:
    """Get precomputed Lloyd-Max boundaries (cached)."""
    _, boundaries = solve_lloyd_max(d, bits)
    return boundaries
