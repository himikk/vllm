# SPDX-License-Identifier: Apache-2.0
"""Walsh-Hadamard Transform (WHT) for TurboQuant v2 block compression.

The WHT on N elements (N = power of 2) Gaussianizes arbitrary input
distributions via the Central Limit Theorem, making Lloyd-Max centroids
for N(0,1) optimal regardless of the actual input distribution.

Reference: github.com/animehacker/llama-turboquant (tq3_wht32_forward)
"""

import math

import torch

# Fixed sign-flip pattern from the reference implementation.
# Applied before WHT to break input structure and improve Gaussianization.
# Pattern for block_size=32 (from llama-turboquant tq3_signs[32]):
WHT_SIGNS_32 = [
    +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
    +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1,
]

# Precomputed sign tensors (lazily initialized per device)
_sign_cache: dict[tuple[int, torch.device], torch.Tensor] = {}


def _get_signs(block_size: int, device: torch.device) -> torch.Tensor:
    """Get sign-flip tensor for given block_size, cached per device."""
    key = (block_size, device)
    if key not in _sign_cache:
        if block_size == 32:
            signs = torch.tensor(WHT_SIGNS_32, dtype=torch.float32, device=device)
        else:
            # For other block sizes: generate deterministic sign pattern
            # Use a fixed seed to ensure reproducibility
            gen = torch.Generator(device="cpu")
            gen.manual_seed(0x574854)  # "WHT" in hex
            signs = torch.sign(torch.randn(block_size, generator=gen))
            signs[signs == 0] = 1.0
            signs = signs.to(dtype=torch.float32, device=device)
        _sign_cache[key] = signs
    return _sign_cache[key]


def wht_forward(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Walsh-Hadamard Transform on blocks of `block_size` elements.

    Args:
        x: [..., D] tensor where D is divisible by block_size.
        block_size: Block size (must be power of 2). Default 32.

    Returns:
        WHT-transformed tensor, same shape as input.
        Each block of `block_size` elements is independently transformed.
    """
    D = x.shape[-1]
    assert D % block_size == 0, f"D={D} not divisible by block_size={block_size}"
    assert block_size & (block_size - 1) == 0, f"block_size={block_size} not power of 2"

    # Reshape to blocks
    orig_shape = x.shape
    x = x.reshape(*orig_shape[:-1], D // block_size, block_size).clone()

    # Step 1: Apply sign flips
    signs = _get_signs(block_size, x.device)
    x = x * signs

    # Step 2: Butterfly stages (log2(block_size) stages)
    n_stages = int(math.log2(block_size))
    for stage in range(n_stages):
        half = 1 << stage
        # Split into pairs and add/subtract
        x_view = x.reshape(*x.shape[:-1], block_size // (2 * half), 2, half)
        lo = x_view[..., 0, :].clone()
        hi = x_view[..., 1, :].clone()
        x_view[..., 0, :] = lo + hi
        x_view[..., 1, :] = lo - hi
        x = x_view.reshape(*x.shape)

    # Step 3: Normalize
    x = x * (1.0 / math.sqrt(block_size))

    return x.reshape(orig_shape)


def wht_inverse(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Inverse WHT. With proper normalization, WHT is self-inverse
    except for the sign-flip order (apply signs AFTER butterfly).

    Args:
        x: [..., D] WHT-transformed tensor.
        block_size: Must match the forward transform.

    Returns:
        Reconstructed tensor, same shape as input.
    """
    D = x.shape[-1]
    assert D % block_size == 0
    assert block_size & (block_size - 1) == 0

    orig_shape = x.shape
    x = x.reshape(*orig_shape[:-1], D // block_size, block_size).clone()

    # Step 1: Butterfly stages (same as forward)
    n_stages = int(math.log2(block_size))
    for stage in range(n_stages):
        half = 1 << stage
        x_view = x.reshape(*x.shape[:-1], block_size // (2 * half), 2, half)
        lo = x_view[..., 0, :].clone()
        hi = x_view[..., 1, :].clone()
        x_view[..., 0, :] = lo + hi
        x_view[..., 1, :] = lo - hi
        x = x_view.reshape(*x.shape)

    # Step 2: Normalize
    x = x * (1.0 / math.sqrt(block_size))

    # Step 3: Remove sign flips (multiply by same signs = undo)
    signs = _get_signs(block_size, x.device)
    x = x * signs

    return x.reshape(orig_shape)
