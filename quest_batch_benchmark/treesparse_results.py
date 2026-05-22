#!/usr/bin/env python3
"""Convert a TreeSparseAttention batch-benchmark JSON into rows matching the
quest benchmark's tpot_vs_batchsize.csv schema (attention='treesparse').

TreeSparse's harness (`benchmark_batch.py`) runs `repeat` repetitions with NO
separate untimed warmup, so its first repetition absorbs one-time FlashInfer
JIT / cold-cache cost and inflates `tpot_mean_ms`. The quest harness, by
contrast, runs an untimed warmup per batch and reports a warm mean-of-3. To
keep the comparison fair we use TreeSparse's `tpot_median_ms` -- the median of
3 reps discards the single cold repetition -- as the representative TPOT, and
store it in the `tpot_ms_mean` column (the column the comparison table reads).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Must stay identical to aggregate_results.OUT_FIELDS.
OUT_FIELDS = [
    "attention", "batch_size", "model", "topk_val", "input_tokens",
    "max_tokens", "repeat", "status", "tpot_ms_mean", "tpot_ms_std",
    "tpot_ms_min", "tpot_ms_max", "ttft_ms_mean", "decode_time_ms_mean",
    "total_time_ms_mean", "throughput_tok_s_mean",
]

MODEL_TAG = "Qwen3-VL-8B-Instruct"


def treesparse_rows(raw: dict, batch_sizes: list[int], top_k: int) -> list[dict]:
    """One OUT_FIELDS row per batch size, sorted ascending.

    `raw` is the TreeSparse results JSON (top-level keys are batch-size
    strings). A batch size missing from `raw` -- benchmark_batch.py stops at
    the first OOM/error -- becomes a status='error' row with blank metrics.
    """
    rows = []
    for bs in sorted(batch_sizes):
        entry = raw.get(str(bs))
        base = {
            "attention": "treesparse",
            "batch_size": bs,
            "model": MODEL_TAG,
            "topk_val": top_k,
        }
        if entry is None:
            blanks = {k: "" for k in OUT_FIELDS
                      if k.endswith(("_mean", "_std", "_min", "_max"))}
            rows.append({**base, "input_tokens": "", "max_tokens": "",
                         "repeat": 0, "status": "error", **blanks})
            continue
        row = {
            **base,
            "input_tokens": entry["prefill_len"],
            "max_tokens": entry["num_decode_tokens"],
            "repeat": entry["repetitions"],
            "status": "ok",
            # median-of-3 -> drops the cold first rep; see module docstring
            "tpot_ms_mean": float(entry["tpot_median_ms"]),
            "tpot_ms_std": float(entry["tpot_std_ms"]),
            "tpot_ms_min": "",   # TreeSparse JSON reports only mean/median/std
            "tpot_ms_max": "",
            "ttft_ms_mean": "",          # TreeSparse harness reports no TTFT
            "decode_time_ms_mean": "",
            "total_time_ms_mean": "",
            "throughput_tok_s_mean": float(entry["throughput_median"]),  # median, same warmup-consistency reason as TPOT
        }
        rows.append({k: (f"{v:.6f}" if isinstance(v, float) else v)
                     for k, v in row.items()})
    return rows


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Convert a TreeSparse results JSON to tpot_vs_batchsize rows")
    p.add_argument("--treesparse-json",
                   default=str(here / "results" / "treesparse_raw.json"))
    p.add_argument("--out-csv",
                   default=str(here / "results" / "treesparse_rows.csv"))
    p.add_argument("--batch-sizes",
                   type=lambda s: [int(x) for x in s.split(",")],
                   default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--top-k", type=int, default=128,
                   help="TreeSparse top-k chunks (run_batch_experiments.sh default).")
    args = p.parse_args()

    with open(args.treesparse_json, encoding="utf-8") as f:
        raw = json.load(f)
    rows = treesparse_rows(raw, args.batch_sizes, args.top_k)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[treesparse] {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
