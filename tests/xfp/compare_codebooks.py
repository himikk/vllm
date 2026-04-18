#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare XFP learned codebooks across experts and against INT4 linear grid.

Hypothesis: If all MoE expert codebooks converge to the same shape, and that
shape matches INT4 AutoRound's uniform symmetric grid, then XFP4 is doing
"INT4 with extra steps" — explaining why both achieve identical 54 % math.

Usage (inside container):
  python3 /opt/xfp_tests/compare_codebooks.py /data/tensordata/GLM-4.7-Flash [bits=4] [n=128]
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


def int4_symmetric_grid(scale: float) -> torch.Tensor:
    """INT4 symmetric: 16 levels = {-8, -7, ..., -1, 0, 1, ..., 7} * scale."""
    return torch.arange(-8, 8, dtype=torch.float32) * scale


def main():
    if len(sys.argv) < 2:
        print("usage: compare_codebooks.py <model_dir> [bits=4] [n=128]")
        sys.exit(1)
    model_dir = Path(sys.argv[1])
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    max_n = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    n_entries = 1 << bits
    st_files = sorted(model_dir.glob("*.safetensors"))

    # Collect per-expert codebooks (mean codebook per tensor, not per channel)
    type_codebooks: dict[str, list[torch.Tensor]] = {}
    type_scales: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
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
                if counts.get(lt, 0) >= max_n:
                    continue

                W = t.float()
                _, codebook, _, _, _ = xfp_pack(W, bits=bits, outlier_sigma=4.0)
                # codebook: [N_out, n_entries] fp16
                cb = codebook.float()
                # Mean codebook across all output channels (the "average shape")
                mean_cb = cb.mean(dim=0)  # [n_entries]

                type_codebooks.setdefault(lt, []).append(mean_cb)
                # Also store the per-row scale (max abs codebook entry / 7 for INT4)
                type_scales.setdefault(lt, []).append(
                    mean_cb.abs().max().item()
                )
                counts[lt] = counts.get(lt, 0) + 1
                processed += 1
                if processed % 50 == 0:
                    print(f"  ... processed {processed}", flush=True)

    print()
    print(f"XFP{bits} codebook analysis — {model_dir.name}")
    print(f"Collected {processed} tensors across {len(type_codebooks)} types")
    print()

    # 1. Inter-expert codebook similarity: how similar are codebooks across
    #    different experts of the same type?
    print("=== Inter-expert codebook similarity (cos sim between mean codebooks) ===")
    print(f"{'type':<18} {'n':>4} {'mean cos':>10} {'min cos':>10} {'std cos':>10}")
    print("-" * 60)

    for lt in sorted(type_codebooks):
        cbs = type_codebooks[lt]
        n = len(cbs)
        if n < 2:
            print(f"{lt:<18} {n:>4} (too few)")
            continue
        stack = torch.stack(cbs)  # [n, n_entries]
        # Pairwise cos sim (sample up to 500 pairs)
        pairs = min(500, n * (n - 1) // 2)
        cos_vals = []
        import random
        random.seed(0)
        idxs = list(range(n))
        for _ in range(pairs):
            i, j = random.sample(idxs, 2)
            c = F.cosine_similarity(
                stack[i].unsqueeze(0), stack[j].unsqueeze(0), dim=1
            ).item()
            cos_vals.append(c)
        mean_c = sum(cos_vals) / len(cos_vals)
        min_c = min(cos_vals)
        std_c = (sum((x - mean_c) ** 2 for x in cos_vals) / len(cos_vals)) ** 0.5
        print(f"{lt:<18} {n:>4} {mean_c:>10.5f} {min_c:>10.5f} {std_c:>10.5f}")

    # 2. Grand mean codebook per type (normalized to [-1, 1])
    print()
    print(f"=== Grand mean codebook per type (normalized to max=1) ===")
    grand_means: dict[str, torch.Tensor] = {}
    for lt in sorted(type_codebooks):
        cbs = type_codebooks[lt]
        stack = torch.stack(cbs)
        grand = stack.mean(dim=0)  # [n_entries]
        grand_norm = grand / grand.abs().max()
        grand_means[lt] = grand_norm
        vals = ", ".join(f"{v:.4f}" for v in grand_norm.tolist())
        print(f"{lt:<18} [{vals}]")

    # 3. Compare to INT4 symmetric grid (normalized)
    int4_grid = int4_symmetric_grid(1.0 / 7.0)  # [-8/7, ..., 7/7]
    # Normalize so max = 1
    int4_norm = int4_grid / int4_grid.abs().max()

    print()
    print(f"INT4 symmetric    [{', '.join(f'{v:.4f}' for v in int4_norm.tolist())}]")
    print()
    print("=== Codebook vs INT4 symmetric grid (cos sim) ===")
    print(f"{'type':<18} {'cos vs INT4':>12} {'L2 dist':>10}")
    print("-" * 45)
    for lt in sorted(grand_means):
        cb = grand_means[lt]
        cos = F.cosine_similarity(
            cb.unsqueeze(0), int4_norm.unsqueeze(0), dim=1
        ).item()
        l2 = (cb - int4_norm).norm().item()
        print(f"{lt:<18} {cos:>12.5f} {l2:>10.4f}")

    # 4. NF4 (NormalFloat4) grid comparison
    # NF4 values from Dettmers (QLoRA): asymmetric, designed for N(0,1)
    nf4_grid = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0
    ])
    nf4_norm = nf4_grid / nf4_grid.abs().max()

    print()
    print(f"NF4 (QLoRA)       [{', '.join(f'{v:.4f}' for v in nf4_norm.tolist())}]")
    print()
    print("=== Codebook vs NF4 grid (cos sim) ===")
    print(f"{'type':<18} {'cos vs NF4':>12} {'L2 dist':>10}")
    print("-" * 45)
    for lt in sorted(grand_means):
        cb = grand_means[lt]
        cos = F.cosine_similarity(
            cb.unsqueeze(0), nf4_norm.unsqueeze(0), dim=1
        ).item()
        l2 = (cb - nf4_norm).norm().item()
        print(f"{lt:<18} {cos:>12.5f} {l2:>10.4f}")

    # 5. Summary
    print()
    print("=== Interpretation ===")
    routed_cbs = [
        grand_means[lt] for lt in grand_means
        if lt.startswith("routed")
    ]
    if len(routed_cbs) >= 2:
        rc = F.cosine_similarity(
            routed_cbs[0].unsqueeze(0), routed_cbs[1].unsqueeze(0), dim=1
        ).item()
        print(f"routed_down vs routed_gate_up grand mean cos: {rc:.5f}")

    all_cos_int4 = []
    all_cos_nf4 = []
    for lt, cb in grand_means.items():
        all_cos_int4.append(F.cosine_similarity(
            cb.unsqueeze(0), int4_norm.unsqueeze(0), dim=1).item())
        all_cos_nf4.append(F.cosine_similarity(
            cb.unsqueeze(0), nf4_norm.unsqueeze(0), dim=1).item())
    print(f"Grand mean cos vs INT4 across all types: {sum(all_cos_int4)/len(all_cos_int4):.5f}")
    print(f"Grand mean cos vs NF4  across all types: {sum(all_cos_nf4)/len(all_cos_nf4):.5f}")


if __name__ == "__main__":
    main()
