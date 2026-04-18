#!/usr/bin/env python3
"""
Profile CUDA Graph replay: measure exact kernel times for tq3w vs fp8.
Runs offline inference (no HTTP), captures torch.profiler trace.

Usage (inside container):
  python3 /opt/profile_graph.py --model /data/tensordata/GLM-4.7-Flash-FP8 --kv fp8
  python3 /opt/profile_graph.py --model /data/tensordata/GLM-4.7-Flash-FP8 --kv tq3w
"""
import argparse
import os
import time
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv", default="fp8", help="KV cache dtype: fp8, tq3w, tq3, etc.")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--prompt", default="Schreibe eine Python-Funktion die prüft ob eine Zahl prim ist. Erkläre den Code Schritt für Schritt.")
    parser.add_argument("--trace-dir", default="/tmp/traces")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_MLA_DISABLE", "1")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "CutlassFP8ScaledMMLinearKernel")

    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"Profile: kv={args.kv}, model={os.path.basename(args.model)}")
    print(f"{'='*60}")

    llm = LLM(
        model=args.model,
        kv_cache_dtype=args.kv,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.33,
        kv_cache_memory_bytes=16 * 1024 * 1024 * 1024,  # 16 GiB
        trust_remote_code=True,
        enforce_eager=False,
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0)

    # Warmup (captures CUDA graphs)
    print("\nWarmup (graph capture)...")
    out = llm.generate(["Hello"], sp)
    print(f"Warmup: {len(out[0].outputs[0].token_ids)} tokens")

    # Second warmup to ensure all graphs are warm
    out = llm.generate(["Explain Python in one sentence."], sp)
    print(f"Warmup2: {len(out[0].outputs[0].token_ids)} tokens")

    # Timed run without profiler
    print("\nTimed run (no profiler)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([args.prompt], sp)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    n_tok = len(out[0].outputs[0].token_ids)
    elapsed = t1 - t0
    print(f"  {n_tok} tokens in {elapsed:.3f}s = {n_tok/elapsed:.1f} tok/s")
    print(f"  Output: {out[0].outputs[0].text[:80]}...")

    # Profiled run
    print("\nProfiled run...")
    os.makedirs(args.trace_dir, exist_ok=True)
    trace_path = os.path.join(args.trace_dir, f"profile_{args.kv}")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        out = llm.generate([args.prompt], sp)

    n_tok2 = len(out[0].outputs[0].token_ids)

    # Print CUDA kernel table sorted by CUDA time
    print(f"\n{'='*60}")
    print(f"CUDA Kernel Summary (kv={args.kv}, {n_tok2} tokens)")
    print(f"{'='*60}")
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=50,
        max_name_column_width=80,
    )
    print(table)

    # Export chrome trace
    prof.export_chrome_trace(trace_path + ".json")
    print(f"\nChrome trace: {trace_path}.json")

if __name__ == "__main__":
    main()
