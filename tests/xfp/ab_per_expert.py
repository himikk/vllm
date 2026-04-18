#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-expert A/B for XFP outlier extraction.

Samples many expert weight tensors from a MoE model and reports the
distribution of reconstruction cos (not just the mean). Two hypotheses:

  H1: Per-expert codebook already works well on MoE. Confirmed by
      running xfp_pack on individual expert matrices and showing cos
      similarity.
  H2: Per-expert outlier σ is tighter than a globally pooled σ — so
      a pooled outlier_sigma (which is what online_moe.py's batched
      path uses) can miss expert-local outliers.

Usage (inside the container):
  python3 /opt/xfp_tests/ab_per_expert.py /data/tensordata/GLM-4.7-Flash [bits=3] [max_per_type=64]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

from vllm.multiquant.xfp.xfp_pack import xfp_pack, dequant_xfp  # noqa: E402


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
        if "q_a_proj" in n:
            return "attn_qa"
        if "kv_a" in n:
            return "attn_kva"
        if "q_b" in n:
            return "attn_qb"
        if "kv_b" in n:
            return "attn_kvb"
        if "o_proj" in n:
            return "attn_o"
        return "attn_other"
    return "other"


def reconstruct(packed, codebook, K, bits, o_idx, o_val):
    base = dequant_xfp(packed, codebook, K=K, bits=bits).to(torch.float32)
    if o_idx is not None and o_val is not None:
        flat = base.reshape(-1).clone()
        flat[o_idx] = o_val.to(torch.float32)
        return flat.reshape(base.shape)
    return base


def measure(W: torch.Tensor, bits: int) -> dict:
    """Single weight: A (bulk) vs B (outlier k=4 per-tensor σ)."""
    Wf = W.float()
    K = Wf.shape[1]

    pA, cbA, _, _, _ = xfp_pack(Wf, bits=bits, outlier_sigma=None)
    rA = reconstruct(pA, cbA, K=K, bits=bits, o_idx=None, o_val=None)
    cosA = F.cosine_similarity(
        Wf.reshape(-1).unsqueeze(0), rA.reshape(-1).unsqueeze(0), dim=1
    ).item()
    mseA = ((Wf - rA) ** 2).mean().item()

    pB, cbB, iB, vB, sB = xfp_pack(Wf, bits=bits, outlier_sigma=4.0)
    rB = reconstruct(pB, cbB, K=K, bits=bits, o_idx=iB, o_val=vB)
    cosB = F.cosine_similarity(
        Wf.reshape(-1).unsqueeze(0), rB.reshape(-1).unsqueeze(0), dim=1
    ).item()
    mseB = ((Wf - rB) ** 2).mean().item()

    return {
        "cos_bulk": cosA, "cos_out": cosB,
        "mse_bulk": mseA, "mse_out": mseB,
        "mse_ratio": mseA / max(mseB, 1e-30),
        "outlier_frac": sB.outlier_fraction,
        "outlier_count": sB.outlier_count,
    }


def percentiles(xs: list[float]) -> tuple[float, float, float, float, float]:
    """(min, p10, p50, p90, max)"""
    s = sorted(xs)
    n = len(s)
    def at(p): return s[min(n - 1, int(n * p))]
    return s[0], at(0.1), at(0.5), at(0.9), s[-1]


def main():
    if len(sys.argv) < 2:
        print("usage: ab_per_expert.py <model_dir> [bits=3] [max_per_type=64]")
        sys.exit(1)
    model_dir = Path(sys.argv[1])
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_per_type = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"no .safetensors in {model_dir}")
        sys.exit(1)

    groups: dict[str, list[dict]] = {}
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
                if counts.get(lt, 0) >= max_per_type:
                    continue
                r = measure(t, bits)
                r["name"] = key
                r["type"] = lt
                groups.setdefault(lt, []).append(r)
                counts[lt] = counts.get(lt, 0) + 1
                processed += 1
                if processed % 40 == 0:
                    print(f"  … processed {processed} tensors", flush=True)

    print()
    print(f"XFP{bits} per-expert distribution (A/B bulk vs outlier k=4)")
    print(f"model: {model_dir.name}")
    print()
    print(f"{'type':<15} {'n':>4} "
          f"{'cos_bulk mean':>14} {'cos_out mean':>14} "
          f"{'Δcos mean':>11} {'Δcos p90':>10} {'Δcos max':>10} "
          f"{'mse ratio p50':>15} {'mse ratio p90':>15} "
          f"{'out_frac':>10}")
    print("-" * 140)

    for lt in sorted(groups):
        items = groups[lt]
        n = len(items)
        cos_bulk = [x["cos_bulk"] for x in items]
        cos_out = [x["cos_out"] for x in items]
        dcos = [b - a for a, b in zip(cos_bulk, cos_out)]
        ratios = [x["mse_ratio"] for x in items]
        outs = [x["outlier_frac"] for x in items]

        mean_cb = sum(cos_bulk) / n
        mean_co = sum(cos_out) / n
        mean_dc = sum(dcos) / n
        _, _, _, p90_dc, max_dc = percentiles(dcos)
        _, _, p50_r, p90_r, _ = percentiles(ratios)
        mean_out = sum(outs) / n

        print(f"{lt:<15} {n:>4} "
              f"{mean_cb:>14.5f} {mean_co:>14.5f} "
              f"{mean_dc:>+11.5f} {p90_dc:>+10.5f} {max_dc:>+10.5f} "
              f"{p50_r:>14.2f}x {p90_r:>14.2f}x "
              f"{100*mean_out:>9.3f}%")

    # Top 10 expert wins across all types
    print()
    print("=== Top 10 expert-level wins by Δcos ===")
    all_items = [x for g in groups.values() for x in g]
    all_items.sort(key=lambda x: -(x["cos_out"] - x["cos_bulk"]))
    for x in all_items[:10]:
        dcos = x["cos_out"] - x["cos_bulk"]
        print(f"  {x['type']:<15} {x['name']:<70} "
              f"cos {x['cos_bulk']:.5f} -> {x['cos_out']:.5f} "
              f"(Δ={dcos:+.5f}, ratio {x['mse_ratio']:.2f}x, "
              f"out {100*x['outlier_frac']:.3f}%)")


if __name__ == "__main__":
    main()
