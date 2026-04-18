#!/usr/bin/env python3
"""
Minimal profiling script — just generates tokens and exits.
Use with nsys: nsys profile -o /tmp/tq3w podman run ... python3 /opt/profile_nsys.py --kv tq3w
Or use NVTX markers for decode-only timing.
"""
import argparse
import os
import time
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv", default="fp8")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--max-model-len", type=int, default=32768)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_MLA_DISABLE", "1")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "CutlassFP8ScaledMMLinearKernel")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        kv_cache_dtype=args.kv,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.33,
        kv_cache_memory_bytes=16 * 1024 * 1024 * 1024,
        trust_remote_code=True,
        enforce_eager=False,
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0)

    # Warmup
    llm.generate(["Hello"], sp)
    llm.generate(["Test warmup"], sp)

    # Timed generation
    prompt = "Schreibe eine Python-Funktion die prüft ob eine Zahl prim ist."
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    n = len(out[0].outputs[0].token_ids)
    print(f"RESULT: kv={args.kv} {n} tokens in {t1-t0:.3f}s = {n/(t1-t0):.1f} tok/s")

if __name__ == "__main__":
    main()
