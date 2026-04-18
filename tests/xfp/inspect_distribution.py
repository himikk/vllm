#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inspect weight distributions of a real model — what do the tails
actually look like? Used to calibrate outlier extraction thresholds
before blindly picking k=3.0 from the paper.

Usage (inside mq-test container):
    python3 /opt/xfp_tests/inspect_distribution.py /data/tensordata/GLM-4.7-Flash
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open


def analyze(tensor: torch.Tensor, name: str, layer_type: str) -> dict:
    W = tensor.float()
    flat = W.reshape(-1)
    N = flat.numel()
    mu = flat.mean().item()
    sd = flat.std().item()
    absmax = flat.abs().max().item()

    # Percentile and moment stats — torch.quantile is limited to
    # ~16M elements, so we compute via sort on a subsample if the
    # tensor is larger.
    centered = (flat - mu).abs()
    mean_abs = centered.mean().item()
    if centered.numel() > 10_000_000:
        # subsample 4M elements for quantile estimates
        perm = torch.randperm(centered.numel(), generator=torch.Generator().manual_seed(0))[:4_000_000]
        c_sample = centered[perm]
    else:
        c_sample = centered
    c_sorted, _ = torch.sort(c_sample)
    m = c_sorted.numel()
    def at(p): return c_sorted[min(m - 1, int(m * p))].item()
    q = [at(0.5), at(0.9), at(0.99), at(0.999), at(0.9999), c_sorted[-1].item()]

    # k-sigma mass counts
    ks = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    k_counts = {k: float((centered > k * sd).sum().item()) / N for k in ks}

    # Cluster-like statistic: fraction of mass above each quantile
    #   how much of |Σ w²| lives in the top 0.1%, 1%, 10%?
    # Sort on the same subsample used for quantiles.
    sq_sample = c_sample * c_sample
    sorted_sq, _ = torch.sort(sq_sample, descending=True)
    total_energy = sorted_sq.sum().item()
    mass_frac = {}
    for pct in (0.001, 0.01, 0.05, 0.1):
        k = max(1, int(sorted_sq.numel() * pct))
        mass_frac[pct] = sorted_sq[:k].sum().item() / max(total_energy, 1e-30)

    return {
        "name": name,
        "type": layer_type,
        "shape": tuple(W.shape),
        "numel": N,
        "mean": mu,
        "std": sd,
        "abs_max": absmax,
        "mean_abs": mean_abs,
        "q50": q[0],
        "q90": q[1],
        "q99": q[2],
        "q999": q[3],
        "q9999": q[4],
        "max_centered": q[5],
        "k_counts": k_counts,
        "mass_frac": mass_frac,
    }


def classify(name: str) -> str:
    n = name.lower()
    if "self_attn" in n or ".attention." in n:
        if "q_proj" in n or "q_a_proj" in n:
            return "attn_q"
        if "k_proj" in n or "kv_a" in n:
            return "attn_k"
        if "v_proj" in n:
            return "attn_v"
        if "o_proj" in n:
            return "attn_o"
        return "attn_other"
    if "shared_expert" in n:
        if "gate" in n or "up" in n:
            return "shared_gate_up"
        if "down" in n:
            return "shared_down"
        return "shared_other"
    if ".experts." in n or "routed_expert" in n:
        if "gate_up" in n or "w13" in n:
            return "routed_gate_up"
        if "down" in n or "w2" in n:
            return "routed_down"
        return "routed_other"
    if ".mlp." in n:
        return "dense_mlp"
    if "embed" in n:
        return "embed"
    if "norm" in n:
        return "norm"
    return "other"


def main():
    if len(sys.argv) < 2:
        print("usage: inspect_distribution.py <model_dir> [n_samples=32]")
        sys.exit(1)
    model_dir = Path(sys.argv[1])
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"no .safetensors in {model_dir}")
        sys.exit(1)

    # Collect weight param names from all shards, pick the first n_samples
    # that are actual 2-D weight matrices (skip norms, embeds, biases).
    samples: list[dict] = []
    seen_types: dict[str, int] = {}

    for f in st_files:
        with safe_open(f, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.endswith(".weight"):
                    continue
                t = handle.get_tensor(key)
                if t.dim() != 2:
                    continue
                lt = classify(key)
                if lt in ("embed", "norm", "other"):
                    continue
                if seen_types.get(lt, 0) >= 3:
                    # keep at most 3 examples per type so we see variety
                    continue
                samples.append(analyze(t, key, lt))
                seen_types[lt] = seen_types.get(lt, 0) + 1
                if len(samples) >= n_samples:
                    break
        if len(samples) >= n_samples:
            break

    # Group by type, print per-type summary
    groups: dict[str, list[dict]] = {}
    for s in samples:
        groups.setdefault(s["type"], []).append(s)

    print(f"Analyzed {len(samples)} weight matrices in "
          f"{len(groups)} classes from {model_dir.name}")
    print()

    # Global summary table
    print(f"{'type':<18} {'count':>5} "
          f"{'mean |w|':>10} {'std':>10} {'absmax':>10} "
          f"{'q99':>10} {'q999':>10} "
          f"{'>3σ %':>7} {'>4σ %':>7} {'>5σ %':>7}")
    print("-" * 120)

    for lt in sorted(groups):
        items = groups[lt]
        n = len(items)
        mean_abs = sum(x["mean_abs"] for x in items) / n
        mean_std = sum(x["std"] for x in items) / n
        mean_absmax = sum(x["abs_max"] for x in items) / n
        q99 = sum(x["q99"] for x in items) / n
        q999 = sum(x["q999"] for x in items) / n
        k3 = 100 * sum(x["k_counts"][3.0] for x in items) / n
        k4 = 100 * sum(x["k_counts"][4.0] for x in items) / n
        k5 = 100 * sum(x["k_counts"][5.0] for x in items) / n
        print(f"{lt:<18} {n:>5} "
              f"{mean_abs:>10.4g} {mean_std:>10.4g} {mean_absmax:>10.4g} "
              f"{q99:>10.4g} {q999:>10.4g} "
              f"{k3:>7.3f} {k4:>7.3f} {k5:>7.3f}")

    # Tail mass — how much L2 energy sits in the heaviest weights?
    print()
    print(f"{'type':<18} "
          f"{'mass@0.1%':>10} {'mass@1%':>10} {'mass@5%':>10} {'mass@10%':>10}")
    print("-" * 75)
    for lt in sorted(groups):
        items = groups[lt]
        m01 = sum(x["mass_frac"][0.001] for x in items) / len(items)
        m1 = sum(x["mass_frac"][0.01] for x in items) / len(items)
        m5 = sum(x["mass_frac"][0.05] for x in items) / len(items)
        m10 = sum(x["mass_frac"][0.1] for x in items) / len(items)
        print(f"{lt:<18} "
              f"{m01*100:>9.2f}% {m1*100:>9.2f}% {m5*100:>9.2f}% {m10*100:>9.2f}%")

    # Per-sample detail for the first few tensors
    print()
    print("=== Individual samples ===")
    print()
    for s in samples[:8]:
        print(f"{s['type']:<18} {s['name']}")
        print(f"  shape={s['shape']}  numel={s['numel']}")
        print(f"  μ={s['mean']:+.4e}  σ={s['std']:.4e}  max|w|={s['abs_max']:.4e}")
        print(f"  mean|w|={s['mean_abs']:.4e}  q50={s['q50']:.4e}  q99={s['q99']:.4e}  "
              f"q999={s['q999']:.4e}  q9999={s['q9999']:.4e}")
        kc = s["k_counts"]
        print("  k-sigma tail: " + "  ".join(
            f"{k}σ={v*100:.3f}%" for k, v in kc.items()
            if k in (2.0, 3.0, 4.0, 5.0, 6.0)
        ))
        mf = s["mass_frac"]
        print(f"  L2 mass: top 0.1%={mf[0.001]*100:.1f}%  "
              f"1%={mf[0.01]*100:.1f}%  "
              f"5%={mf[0.05]*100:.1f}%  "
              f"10%={mf[0.1]*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
