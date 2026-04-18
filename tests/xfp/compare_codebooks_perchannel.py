#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-channel codebook comparison against INT4/NF4 reference grids.

Each output channel of each weight tensor has its OWN Lloyd codebook.
We compare each of those individually against the reference grids and
report the distribution of cos similarities — no averaging, no grand mean.

Reports:
  - Histogram of cos similarities (per-channel vs INT4 and vs NF4)
  - Per-type percentiles (p10, p50, p90, min, max)
  - Count of "close" (cos > 0.999), "similar" (0.99-0.999), "different" (<0.99)
  - Top outlier codebooks that deviate most from both grids

Usage (inside container):
  python3 /opt/xfp_tests/compare_codebooks_perchannel.py /data/tensordata/GLM-4.7-Flash [bits=4] [n_tensors=64]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.multiquant.xfp.xfp_pack import xfp_pack  # noqa: E402


def classify(name: str) -> str:
    n = name.lower()
    if ".experts." in n or "routed_expert" in n:
        if "down" in n or "w2" in n:
            return "routed_down"
        return "routed_gate_up"
    if "shared_expert" in n:
        if "down" in n:
            return "shared_down"
        return "shared_gate_up"
    if "self_attn" in n or ".attention." in n:
        return "attn"
    if ".mlp." in n and "expert" not in n:
        return "dense_mlp"
    return "other"


def make_reference_grids(bits: int) -> dict[str, torch.Tensor]:
    """Reference codebooks, all normalized to max-abs = 1."""
    n = 1 << bits
    grids = {}

    # INT symmetric: {-(2^(b-1)), ..., -1, 0, 1, ..., 2^(b-1)-1} / (2^(b-1)-1)
    half = n // 2
    int_grid = torch.arange(-half, half, dtype=torch.float32) / (half - 1)
    grids["INT_sym"] = int_grid

    # Lloyd-Max for N(0,1) — from centroids.py
    lloyd_max = {
        2: [-1.5104, -0.4528, 0.4528, 1.5104],
        3: [-2.1519, -1.3439, -0.7560, -0.2451,
             0.2451,  0.7560,  1.3439,  2.1519],
        4: [-2.7326, -2.0690, -1.6181, -1.2563,
            -0.9424, -0.6568, -0.3881, -0.1284,
             0.1284,  0.3881,  0.6568,  0.9424,
             1.2563,  1.6181,  2.0690,  2.7326],
    }
    if bits in lloyd_max:
        lm = torch.tensor(lloyd_max[bits], dtype=torch.float32)
        grids["LloydMax_N01"] = lm / lm.abs().max()

    # NF4 (only for bits=4)
    if bits == 4:
        nf4 = torch.tensor([
            -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
             0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
        ])
        grids["NF4"] = nf4 / nf4.abs().max()

    return grids


def main():
    if len(sys.argv) < 2:
        print("usage: compare_codebooks_perchannel.py <model_dir> [bits=4] [n_tensors=64]")
        sys.exit(1)
    model_dir = Path(sys.argv[1])
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    max_tensors = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    st_files = sorted(model_dir.glob("*.safetensors"))
    grids = make_reference_grids(bits)

    # Collect: per type → list of per-channel cos values vs each grid
    # type → grid_name → list[float]
    results: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, int] = {}
    total_channels = 0
    processed = 0

    for f in st_files:
        with safe_open(f, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.endswith(".weight"):
                    continue
                t = handle.get_tensor(key)
                if t.dim() != 2:
                    continue
                lt = classify(key)
                if lt == "other":
                    continue
                if counts.get(lt, 0) >= max_tensors:
                    continue

                W = t.float()
                _, codebook, _, _, _ = xfp_pack(W, bits=bits, outlier_sigma=4.0)
                cb = codebook.float()  # [N_out, n_entries]

                # Normalize each row to max-abs = 1
                row_max = cb.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
                cb_norm = cb / row_max  # [N_out, n_entries]

                for gname, gvec in grids.items():
                    gvec_expanded = gvec.unsqueeze(0).expand_as(cb_norm)
                    cos_per_row = F.cosine_similarity(cb_norm, gvec_expanded, dim=1)
                    cos_list = cos_per_row.tolist()
                    results.setdefault(lt, {}).setdefault(gname, []).extend(cos_list)

                total_channels += cb.shape[0]
                counts[lt] = counts.get(lt, 0) + 1
                processed += 1
                if processed % 40 == 0:
                    print(f"  ... {processed} tensors, {total_channels} channels",
                          flush=True)

    print()
    print(f"XFP{bits} per-channel codebook vs reference grids — {model_dir.name}")
    print(f"{processed} tensors, {total_channels} total channels")
    print()

    # Per-type, per-grid: percentiles and cluster counts
    for gname in sorted(next(iter(results.values())).keys()):
        print(f"=== vs {gname} ===")
        print(f"{'type':<18} {'n_ch':>8} "
              f"{'min':>8} {'p10':>8} {'p50':>8} {'p90':>8} {'max':>8} "
              f"{'>.999':>7} {'.99-.999':>9} {'<.99':>7}")
        print("-" * 105)

        all_cos = []
        for lt in sorted(results):
            vals = results[lt][gname]
            all_cos.extend(vals)
            n = len(vals)
            s = sorted(vals)
            def at(p): return s[min(n-1, int(n*p))]
            close = sum(1 for v in vals if v > 0.999)
            similar = sum(1 for v in vals if 0.99 <= v <= 0.999)
            diff = sum(1 for v in vals if v < 0.99)
            print(f"{lt:<18} {n:>8} "
                  f"{s[0]:>8.5f} {at(0.1):>8.5f} {at(0.5):>8.5f} "
                  f"{at(0.9):>8.5f} {s[-1]:>8.5f} "
                  f"{close:>7} {similar:>9} {diff:>7}")

        # Overall
        n = len(all_cos)
        s = sorted(all_cos)
        def at(p): return s[min(n-1, int(n*p))]
        close = sum(1 for v in all_cos if v > 0.999)
        similar = sum(1 for v in all_cos if 0.99 <= v <= 0.999)
        diff = sum(1 for v in all_cos if v < 0.99)
        print("-" * 105)
        print(f"{'OVERALL':<18} {n:>8} "
              f"{s[0]:>8.5f} {at(0.1):>8.5f} {at(0.5):>8.5f} "
              f"{at(0.9):>8.5f} {s[-1]:>8.5f} "
              f"{close:>7} {similar:>9} {diff:>7}")
        pct_close = 100 * close / n
        pct_diff = 100 * diff / n
        print(f"  → {pct_close:.1f}% channels >.999 (near-identical), "
              f"{pct_diff:.1f}% channels <.99 (structurally different)")
        print()


if __name__ == "__main__":
    main()
