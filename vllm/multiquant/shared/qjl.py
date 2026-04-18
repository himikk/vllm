# SPDX-License-Identifier: Apache-2.0
"""QJL (Quantized Johnson-Lindenstrauss) projection — shared by TQ + RQ.

1-bit sign quantization on residuals for unbiased inner product estimation.
The QJL correction term: ||r|| * sqrt(π/2)/m * <S@q, sign(S@r)>
"""

import math

import torch


def generate_qjl_matrix(
    d: int, seed: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate i.i.d. N(0,1) projection matrix for QJL."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    S = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    return S.to(device)


@torch.no_grad()
def qjl_correction(
    residual: torch.Tensor,
    S: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute QJL residual correction.

    Args:
        residual: (..., d) unit-normalized residual vectors
        S: (d, d) random projection matrix
        head_dim: original vector dimension (for scaling)

    Returns:
        signs: (..., d) sign quantized projections
        res_norms: (...) residual norms
        correction: (..., d) QJL correction vectors
    """
    res_norms = residual.norm(dim=-1, keepdim=True)
    projected = residual @ S.T
    signs = torch.sign(projected)
    signs[signs == 0] = 1.0

    correction_scale = math.sqrt(math.pi / 2) / head_dim
    correction = correction_scale * res_norms * (signs @ S)

    return signs, res_norms.squeeze(-1), correction
