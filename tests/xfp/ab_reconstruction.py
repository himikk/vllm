#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A/B comparison of XFP reconstruction with and without outlier extraction.

Runs xfp_pack twice on each sampled weight matrix:
  - bulk-only (outlier_sigma=None, v1 baseline)
  - outlier-split (outlier_sigma=4.0, v3)

Reports per-layer-type cos sim, MSE, and the delta. Helps calibrate the
outlier_sigma parameter and shows which layer classes benefit.

Usage (inside the mq-test container):
  python3 /opt/xfp_tests/ab_reconstruction.py /data/tensordata/GLM-4.7-Flash
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


# Make vllm.multiquant importable from the mounted source tree
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

from vllm.multiquant.xfp.xfp_pack import xfp_pack, dequant_xfp  # noqa: E402


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
        if "q_b" in n or "kv_b" in n:
            return "attn_proj_b"
        return "attn_other"
    if "shared_expert" in n:
        if "down" in n:
            return "shared_down"
        return "shared_gate_up"
    if ".experts." in n or "routed_expert" in n:
        if "down" in n or "w2" in n:
            return "routed_down"
        return "routed_gate_up"
    if ".mlp." in n and "expert" not in n:
        return "dense_mlp"
    return "other"


def reconstruct(packed, codebook, K, bits, o_idx, o_val):
    base = dequant_xfp(packed, codebook, K=K, bits=bits).to(torch.float32)
    if o_idx is not None and o_val is not None:
        flat = base.reshape(-1).clone()
        flat[o_idx] = o_val.to(torch.float32)
        return flat.reshape(base.shape)
    return base


def a_b_compare(W: torch.Tensor, bits: int) -> dict:
    Wf = W.float()
    N, K = Wf.shape
    numel = Wf.numel()

    # A: bulk only
    pA, cbA, _, _, sA = xfp_pack(Wf, bits=bits, outlier_sigma=None)
    W_recA = reconstruct(pA, cbA, K=K, bits=bits, o_idx=None, o_val=None)
    mseA = ((Wf - W_recA) ** 2).mean().item()
    cosA = F.cosine_similarity(
        Wf.reshape(-1).unsqueeze(0),
        W_recA.reshape(-1).unsqueeze(0), dim=1,
    ).item()

    # B: outlier split k=4
    pB, cbB, oIdx, oVal, sB = xfp_pack(Wf, bits=bits, outlier_sigma=4.0)
    W_recB = reconstruct(pB, cbB, K=K, bits=bits, o_idx=oIdx, o_val=oVal)
    mseB = ((Wf - W_recB) ** 2).mean().item()
    cosB = F.cosine_similarity(
        Wf.reshape(-1).unsqueeze(0),
        W_recB.reshape(-1).unsqueeze(0), dim=1,
    ).item()

    return {
        "numel": numel,
        "shape": tuple(Wf.shape),
        "mse_bulk": mseA,
        "cos_bulk": cosA,
        "mse_out": mseB,
        "cos_out": cosB,
        "mse_ratio": mseA / max(mseB, 1e-30),  # > 1 means outlier path wins
        "outlier_frac": sB.outlier_fraction,
        "outlier_count": sB.outlier_count,
    }


def main():
    if len(sys.argv) < 2:
        print("usage: ab_reconstruction.py <model_dir> [bits=4] [n_per_type=2]")
        sys.exit(1)
    model_dir = Path(sys.argv[1])
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_per_type = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"no .safetensors in {model_dir}")
        sys.exit(1)

    groups: dict[str, list[dict]] = {}
    type_counts: dict[str, int] = {}

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
                if type_counts.get(lt, 0) >= n_per_type:
                    continue
                r = a_b_compare(t, bits)
                r["name"] = key
                r["type"] = lt
                groups.setdefault(lt, []).append(r)
                type_counts[lt] = type_counts.get(lt, 0) + 1

    print(f"XFP{bits} reconstruction A/B: bulk-only vs outlier-split (k=4)")
    print(f"model: {model_dir.name}")
    print()
    print(f"{'type':<18} {'n':>3} "
          f"{'cos_bulk':>10} {'cos_out':>10} {'Δcos':>10} "
          f"{'mse_bulk':>12} {'mse_out':>12} {'mse ratio':>10} "
          f"{'outliers %':>11}")
    print("-" * 110)

    for lt in sorted(groups):
        items = groups[lt]
        n = len(items)
        cb = sum(x["cos_bulk"] for x in items) / n
        co = sum(x["cos_out"] for x in items) / n
        dcos = co - cb
        mb = sum(x["mse_bulk"] for x in items) / n
        mo = sum(x["mse_out"] for x in items) / n
        mr = sum(x["mse_ratio"] for x in items) / n
        of = 100.0 * sum(x["outlier_frac"] for x in items) / n
        print(f"{lt:<18} {n:>3} "
              f"{cb:>10.5f} {co:>10.5f} {dcos:>+10.5f} "
              f"{mb:>12.4e} {mo:>12.4e} {mr:>9.2f}x "
              f"{of:>10.3f}%")

    # Overall
    all_items = [x for g in groups.values() for x in g]
    n = len(all_items)
    cb = sum(x["cos_bulk"] for x in all_items) / n
    co = sum(x["cos_out"] for x in all_items) / n
    mb = sum(x["mse_bulk"] for x in all_items) / n
    mo = sum(x["mse_out"] for x in all_items) / n
    mr = sum(x["mse_ratio"] for x in all_items) / n
    print("-" * 110)
    print(f"{'OVERALL':<18} {n:>3} "
          f"{cb:>10.5f} {co:>10.5f} {co - cb:>+10.5f} "
          f"{mb:>12.4e} {mo:>12.4e} {mr:>9.2f}x")

    print()
    print("=== Top wins (layers where outlier extraction helps most) ===")
    sorted_all = sorted(all_items, key=lambda x: -x["mse_ratio"])
    for x in sorted_all[:8]:
        print(f"  {x['type']:<18} {x['name']}")
        print(f"    shape={x['shape']}  "
              f"cos {x['cos_bulk']:.5f} -> {x['cos_out']:.5f}  "
              f"mse {x['mse_bulk']:.3e} -> {x['mse_out']:.3e}  "
              f"({x['mse_ratio']:.1f}x)  "
              f"outliers: {100*x['outlier_frac']:.3f}%")


if __name__ == "__main__":
    main()
