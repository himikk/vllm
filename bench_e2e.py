#!/usr/bin/env python3
"""
End-to-end A/B benchmark: tq3w vs fp8, multiple runs.
"""
import argparse
import os
import time
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv", default="fp8")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_MLA_DISABLE", "1")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "CutlassFP8ScaledMMLinearKernel")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        kv_cache_dtype=args.kv,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.05,
        kv_cache_memory_bytes=16 * 1024 * 1024 * 1024,
        trust_remote_code=True,
        enforce_eager=False,
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0)

    # Warmup
    for _ in range(3):
        llm.generate(["Hello world"], sp)

    prompts = [
        "Schreibe eine Python-Funktion die prüft ob eine Zahl prim ist. Erkläre den Code.",
        "Was ist 664 + 124? Antworte nur mit der Zahl.",
        "Erkläre was ein Transformer-Modell ist, Schritt für Schritt.",
        "Schreibe eine Fibonacci-Funktion in Python mit Memoization.",
        "Wer hat die Zauberflöte komponiert? Antworte in einem Satz.",
    ]

    results = []
    for i in range(args.runs):
        prompt = prompts[i % len(prompts)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        n = len(out[0].outputs[0].token_ids)
        elapsed = t1 - t0
        tps = n / elapsed
        results.append(tps)
        text = out[0].outputs[0].text[:60].replace('\n', ' ')
        print(f"  Run {i+1}: {n:3d} tok in {elapsed:.3f}s = {tps:.1f} tok/s | {text}...")

    avg = sum(results) / len(results)
    print(f"\n  AVERAGE ({args.kv}): {avg:.1f} tok/s ({args.runs} runs)")
    print(f"  MIN: {min(results):.1f}, MAX: {max(results):.1f}")

if __name__ == "__main__":
    main()
