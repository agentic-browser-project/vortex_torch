"""TPOT micro-benchmark for the block_size_sweep variants.

For each submission config:
  1. Boots the sglang engine via get_engine_from_json()
  2. Warms up with a short prompt
  3. Runs N timed generations of fixed length
  4. Reports TPOT (ms/token) and decode throughput (tok/s)

Way faster than AIME24 (~3-5 min/variant) and directly measures the
bandwidth-sensitive quantity. Use after RULER quality has been validated.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# silence noisy deprecation warning early
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import vortex_torch  # noqa: F401  registers attention backend
from vortex_torch.engine.sgl import get_engine_from_json


PROMPT_TEMPLATE = (
    "You are a helpful assistant. Below is a passage I will quiz you on later.\n\n"
    "Passage: {filler}\n\nNow please write a long, detailed essay on a related topic. "
    "Make sure your essay is at least 800 words and includes specific examples."
)


def make_prompt(approx_input_tokens: int) -> str:
    # ~4 chars/token average; build filler to hit the target.
    target_chars = approx_input_tokens * 4
    base = PROMPT_TEMPLATE.format(filler="")
    overhead = len(base)
    filler_words_needed = max(1, (target_chars - overhead) // 6)
    filler = " ".join(["sample sentence with several words."] * (filler_words_needed // 5 + 1))
    return PROMPT_TEMPLATE.format(filler=filler)


def bench_one(config_path: str, input_tokens: int, out_tokens: int, repeats: int) -> dict:
    print(f"\n========================================")
    print(f"=== bench: {config_path}")
    print(f"=== input~={input_tokens} tok, out={out_tokens} tok, repeats={repeats}")
    print(f"========================================", flush=True)

    engine = get_engine_from_json(config_path)
    prompt = make_prompt(input_tokens)
    # measure actual prompt tokenized length
    try:
        tok = engine.tokenizer_manager.tokenizer
        actual_input = len(tok.encode(prompt))
    except Exception:
        actual_input = -1
    print(f"[bench] actual tokenized input: {actual_input} tok")

    sampling = {
        "max_new_tokens": out_tokens,
        "temperature": 0.0,         # greedy for reproducibility
        "top_p": 1.0,
        "top_k": -1,
        "ignore_eos": True,         # never stop early, always emit full out_tokens
    }

    # Warmup: one short run to JIT-compile any lazy paths.
    print("[bench] warming up...", flush=True)
    _ = engine.generate([prompt], sampling_params={**sampling, "max_new_tokens": 8})

    results = []
    for i in range(repeats):
        t0 = time.perf_counter()
        out = engine.generate([prompt], sampling_params=sampling)
        t1 = time.perf_counter()
        wall = t1 - t0

        # Determine actually generated tokens
        # sglang offline returns a list of dicts
        if isinstance(out, list) and out:
            o = out[0]
            gen_text = o.get("text", "") if isinstance(o, dict) else ""
            meta = o.get("meta_info", {}) if isinstance(o, dict) else {}
            n_out = meta.get("completion_tokens", -1)
            n_in  = meta.get("prompt_tokens", actual_input)
        else:
            gen_text = ""
            n_out = -1
            n_in = actual_input

        # TPOT estimate: wall time minus (rough) prefill / out_tokens.
        # Without a prefill timer breakout from sglang, we report the full
        # generate() wall divided by output tokens as a conservative TPOT.
        tpot_ms = (wall / max(1, n_out)) * 1000.0 if n_out > 0 else float("nan")
        throughput = (n_out / wall) if wall > 0 and n_out > 0 else float("nan")

        results.append({
            "rep": i,
            "wall_sec": wall,
            "n_in": n_in,
            "n_out": n_out,
            "tpot_ms_per_tok": tpot_ms,
            "throughput_tok_per_sec": throughput,
        })
        print(f"[bench] rep {i}: wall={wall:.2f}s  in={n_in}  out={n_out}  "
              f"tpot={tpot_ms:.2f}ms  throughput={throughput:.1f} tok/s", flush=True)

    # Median across repeats
    sorted_t = sorted(r["tpot_ms_per_tok"] for r in results if r["tpot_ms_per_tok"] == r["tpot_ms_per_tok"])
    sorted_th = sorted(r["throughput_tok_per_sec"] for r in results if r["throughput_tok_per_sec"] == r["throughput_tok_per_sec"])
    median_tpot = sorted_t[len(sorted_t)//2] if sorted_t else float("nan")
    median_th = sorted_th[len(sorted_th)//2] if sorted_th else float("nan")

    summary = {
        "config": config_path,
        "input_tokens_target": input_tokens,
        "output_tokens": out_tokens,
        "repeats": repeats,
        "results": results,
        "median_tpot_ms_per_tok": median_tpot,
        "median_throughput_tok_per_sec": median_th,
    }
    print(f"[bench] >>> median TPOT: {median_tpot:.2f} ms/tok  ({median_th:.1f} tok/s)")

    # Tear down
    try:
        engine.shutdown()
    except Exception:
        pass
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="paths to submission JSON configs")
    ap.add_argument("--input-tokens", type=int, default=2048)
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-json", default="tpot_microbench_results.json")
    args = ap.parse_args()

    all_summaries = []
    for cfg in args.configs:
        try:
            s = bench_one(cfg, args.input_tokens, args.out_tokens, args.repeats)
            all_summaries.append(s)
        except Exception as e:
            import traceback
            print(f"[ERROR] {cfg} failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            all_summaries.append({"config": cfg, "error": str(e)})

    print("\n========================================")
    print("=== FINAL SUMMARY")
    print("========================================")
    print(f"{'config':<70} {'tpot (ms/tok)':>15} {'throughput (tok/s)':>20}")
    print("-" * 110)
    for s in all_summaries:
        if "error" in s:
            print(f"{Path(s['config']).name:<70} {'ERROR: '+s['error'][:30]:>40}")
        else:
            print(f"{Path(s['config']).name:<70} {s['median_tpot_ms_per_tok']:>15.2f} {s['median_throughput_tok_per_sec']:>20.1f}")

    with open(args.out_json, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\n[saved] full results -> {args.out_json}")


if __name__ == "__main__":
    main()
