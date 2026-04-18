#!/usr/bin/env python3
"""TurboQuant KV-Cache Benchmark.

Measures throughput + KV-cache usage at varying context lengths.
Reads cache metrics from vLLM /metrics endpoint.

Usage:
  python3 bench_kvcache.py --url http://localhost:8011 --model qwen35-35b-tq3 \
    --label "TQ3" --contexts 0 512 2048 8192 32768 65536 131072
"""
import argparse
import json
import re
import sys
import time
import urllib.request

sys.stdout.reconfigure(line_buffering=True)


def get_kv_cache_usage(url):
    """Read KV-cache usage from vLLM /metrics endpoint."""
    try:
        resp = urllib.request.urlopen(f"{url}/metrics", timeout=5)
        text = resp.read().decode()
        # Parse kv_cache_usage_perc
        m = re.search(r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.]+)', text)
        usage_pct = float(m.group(1)) if m else -1

        # Parse cache config
        m2 = re.search(r'num_gpu_blocks="(\d+)"', text)
        num_blocks = int(m2.group(1)) if m2 else -1
        m3 = re.search(r'block_size="(\d+)"', text)
        block_size = int(m3.group(1)) if m3 else -1
        m4 = re.search(r'kv_cache_memory_bytes="(\d+)"', text)
        total_bytes = int(m4.group(1)) if m4 else -1

        return {
            "usage_pct": usage_pct,
            "num_blocks": num_blocks,
            "block_size": block_size,
            "total_bytes": total_bytes,
            "used_bytes": int(usage_pct * total_bytes) if total_bytes > 0 else -1,
        }
    except Exception as e:
        return {"error": str(e)}


def chat(url, model, prompt, max_tokens=150, temperature=0):
    """Send chat request, return (content, prompt_tokens, completion_tokens, elapsed)."""
    data = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    elapsed = time.time() - t0
    msg = body["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning") or ""
    usage = body["usage"]
    return content, usage.get("prompt_tokens", 0), usage["completion_tokens"], elapsed


def build_filler(n_tokens):
    """Build deterministic filler text of ~n_tokens."""
    import random
    random.seed(42)
    words = ("the quick brown fox jumps over a lazy dog near the river bank while "
             "clouds drift slowly across the wide blue sky and birds sing softly in "
             "the tall green trees beside the old stone wall that runs along the path "
             "through the quiet valley where flowers bloom in spring and autumn leaves "
             "fall gently to the ground covering everything in golden light ").split()
    n_words = int(n_tokens / 1.3)
    return " ".join(words[i % len(words)] for i in range(n_words))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--contexts", type=int, nargs="+",
                        default=[0, 512, 2048, 8192, 32768, 65536])
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Completion tokens per test")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"KV-Cache Benchmark: {args.label}")
    print(f"URL: {args.url}  Model: {args.model}")
    print(f"Contexts: {args.contexts}  Rounds: {args.rounds}")
    print(f"{'='*70}")

    # Initial cache state
    cache_info = get_kv_cache_usage(args.url)
    print(f"\nCache config: {cache_info.get('num_blocks','')} blocks × "
          f"{cache_info.get('block_size','')} tokens, "
          f"{cache_info.get('total_bytes',0)/1e9:.1f} GB total")

    prompt_base = "Erkläre kurz was ein Transformer ist."

    results = []
    print(f"\n{'ctx':>8s} {'prompt_tok':>10s} {'gen_tok':>8s} {'tok/s':>8s} "
          f"{'time_s':>8s} {'kv_usage':>10s} {'kv_MB':>8s}")
    print("-" * 70)

    for ctx in args.contexts:
        if ctx > 0:
            filler = build_filler(ctx)
            prompt = f"Context:\n{filler}\n\nQuestion: {prompt_base}"
        else:
            prompt = prompt_base

        times = []
        prompt_toks = []
        gen_toks = []

        for r in range(args.rounds):
            try:
                _, pt, gt, elapsed = chat(
                    args.url, args.model, prompt,
                    max_tokens=args.max_tokens, temperature=0)
                times.append(elapsed)
                prompt_toks.append(pt)
                gen_toks.append(gt)
            except Exception as e:
                print(f"  ctx={ctx}: FAILED ({e})")
                break

        if not times:
            results.append({"context": ctx, "error": "failed"})
            continue

        # Read cache usage after generation
        cache = get_kv_cache_usage(args.url)

        avg_pt = sum(prompt_toks) / len(prompt_toks)
        avg_gt = sum(gen_toks) / len(gen_toks)
        avg_time = sum(times) / len(times)
        avg_toks = avg_gt / avg_time if avg_time > 0 else 0
        kv_pct = cache.get("usage_pct", -1) * 100
        kv_mb = cache.get("used_bytes", 0) / 1e6

        print(f"{ctx:>8,d} {avg_pt:>10.0f} {avg_gt:>8.0f} {avg_toks:>8.1f} "
              f"{avg_time:>8.2f} {kv_pct:>9.1f}% {kv_mb:>7.0f}")

        results.append({
            "context": ctx,
            "prompt_tokens": round(avg_pt),
            "gen_tokens": round(avg_gt),
            "tok_s": round(avg_toks, 1),
            "time_s": round(avg_time, 2),
            "kv_usage_pct": round(kv_pct, 1),
            "kv_used_mb": round(kv_mb),
        })

    print(f"\n{'='*70}")
    print(f"JSON: {json.dumps({'label': args.label, 'results': results})}")


if __name__ == "__main__":
    main()
