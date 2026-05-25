"""Analyze real sparse_kv_indices traces dumped from vortex_torch and compute
Block Fetch / Method 1 / Method 2 metrics on each (request, kv_head) selection.

Input: jsonl trace files (one per layer) produced by the dump hook in
       flashinfer.py, each line = one decode step's full CSR selection.

For each trace row (one (request, kv_head) selection):
  1. Expand selected block_ids into a "needed tokens" set
  2. For each (P, threshold) policy config, compute:
       coverage, waste, loaded_MB
  3. Aggregate across all rows: mean / median / p90 / count

Output: per-(method, P, threshold) summary table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

KV_BYTES_PER_TOKEN = 8 * 128 * 2 * 2  # num_kv_heads_total isn't right here — see below
# Actually: per (request, kv_head), bytes = block_size * 1 * head_dim * 2 (K+V) * 2 (bf16)
# So BYTES_PER_TOKEN_PER_HEAD = head_dim * 2 * 2 = 128 * 4 = 512
BYTES_PER_TOKEN_PER_HEAD = 128 * 2 * 2  # head_dim * 2(K+V) * 2(bf16)


def analyze_row(selected_blocks, block_size, page_sizes, thresholds):
    """For one (request, kv_head) selection, compute metrics for every policy."""
    # Expand block ids to token ids
    needed = set()
    for b in selected_blocks:
        for tok in range(b * block_size, (b + 1) * block_size):
            needed.add(tok)
    N = len(needed)
    if N == 0:
        return []

    # bytes/token (per single kv_head, since traces are per-kv_head)
    btok = BYTES_PER_TOKEN_PER_HEAD

    out = []
    # Block Fetch
    out.append({
        "method": "Block Fetch", "P": block_size, "threshold": None,
        "N": N, "loaded_tok": N, "needed_loaded": N,
        "coverage": 1.0, "waste": 0.0,
        "loaded_MB": N * btok / 1e6,
    })
    for P in page_sizes:
        if P < block_size or P % block_size != 0:
            continue
        # Method 1: any-hit
        pages_hit = {tok // P for tok in needed}
        loaded_tok_m1 = len(pages_hit) * P
        out.append({
            "method": "Method 1", "P": P, "threshold": None,
            "N": N, "loaded_tok": loaded_tok_m1, "needed_loaded": N,
            "coverage": 1.0, "waste": (loaded_tok_m1 - N) / loaded_tok_m1 if loaded_tok_m1 > 0 else 0.0,
            "loaded_MB": loaded_tok_m1 * btok / 1e6,
        })
        # Method 2: threshold (drop below)
        counts = defaultdict(int)
        for tok in needed:
            counts[tok // P] += 1
        for thr in thresholds:
            min_hits = max(1, int(thr * P + 0.999999))
            kept = [p for p, c in counts.items() if c >= min_hits]
            loaded_tok_m2 = len(kept) * P
            needed_kept = sum(counts[p] for p in kept)
            cov = needed_kept / N
            waste = (loaded_tok_m2 - needed_kept) / loaded_tok_m2 if loaded_tok_m2 > 0 else 0.0
            out.append({
                "method": "Method 2", "P": P, "threshold": thr,
                "N": N, "loaded_tok": loaded_tok_m2, "needed_loaded": needed_kept,
                "coverage": cov, "waste": waste,
                "loaded_MB": loaded_tok_m2 * btok / 1e6,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="logs/traces")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="restrict to specific layer ids (default: all)")
    ap.add_argument("--page-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.10, 0.25, 0.50, 0.75])
    ap.add_argument("--out-csv", default="logs/trace_policy_analysis.csv")
    ap.add_argument("--max-rows-per-layer", type=int, default=200,
                    help="cap per-layer rows for tractability")
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    trace_files = sorted(trace_dir.glob("trace_layer*.jsonl"))
    print(f"Found {len(trace_files)} trace files in {trace_dir}")

    all_results = []
    layer_stats = {}
    total_rows_processed = 0

    for tf in trace_files:
        # parse layer id from filename
        layer_id = int(tf.stem.replace("trace_layer", ""))
        if args.layers is not None and layer_id not in args.layers:
            continue

        with open(tf) as f:
            lines = f.readlines()
        # Subsample to max_rows_per_layer
        if len(lines) > args.max_rows_per_layer:
            step = len(lines) // args.max_rows_per_layer
            lines = lines[::step][:args.max_rows_per_layer]

        layer_count = 0
        for line in lines:
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            block_size = trace["block_size"]
            indptr = trace["indptr"]
            indices = trace["indices"]

            for row_idx in range(len(indptr) - 1):
                start, end = indptr[row_idx], indptr[row_idx + 1]
                if start == end:
                    continue
                selected_blocks = indices[start:end]
                row_results = analyze_row(selected_blocks, block_size,
                                          args.page_sizes, args.thresholds)
                for r in row_results:
                    r["layer_id"] = layer_id
                    r["row_idx"] = row_idx
                    all_results.append(r)
                layer_count += 1
                total_rows_processed += 1
        layer_stats[layer_id] = layer_count

    print(f"\nProcessed {total_rows_processed} (request, kv_head) selections")
    print(f"Per-layer counts: {layer_stats}")

    # ==== aggregate by (method, P, threshold) ====
    groups = defaultdict(list)
    for r in all_results:
        key = (r["method"], r["P"], r["threshold"])
        groups[key].append(r)

    print(f"\n{'method':<12} {'P':>4} {'thr':>5} {'count':>6}  "
          f"{'cov_mean':>9} {'cov_p50':>8} {'cov_p10':>8}  "
          f"{'waste_mean':>10} {'waste_p50':>9}  "
          f"{'N_mean':>7} {'loadMB_mean':>11}")
    print("-" * 115)

    def fmt(lst, fn):
        try: return fn(lst)
        except (statistics.StatisticsError, ValueError): return float("nan")
    def p(lst, q):
        if not lst: return float("nan")
        s = sorted(lst)
        return s[int(len(s) * q)]

    summary_rows = []
    # custom sort: Block Fetch first, then Method 1 by P, then Method 2 by (P, threshold)
    def sort_key(k):
        method, P, thr = k
        if method == "Block Fetch": return (0, P, 0)
        if method == "Method 1": return (1, P, 0)
        return (2, P, thr if thr is not None else 0)

    for key in sorted(groups.keys(), key=sort_key):
        method, P, thr = key
        rows = groups[key]
        covs = [r["coverage"] for r in rows]
        wasts = [r["waste"] for r in rows]
        Ns = [r["N"] for r in rows]
        mbs = [r["loaded_MB"] for r in rows]
        s = {
            "method": method, "P": P, "threshold": thr if thr is not None else "—",
            "count": len(rows),
            "cov_mean": fmt(covs, statistics.mean),
            "cov_p50":  p(covs, 0.5),
            "cov_p10":  p(covs, 0.10),
            "waste_mean": fmt(wasts, statistics.mean),
            "waste_p50":  p(wasts, 0.5),
            "N_mean":   fmt(Ns, statistics.mean),
            "loadMB_mean": fmt(mbs, statistics.mean),
        }
        summary_rows.append(s)
        thr_str = f"{thr:.2f}" if thr is not None else "—"
        print(f"{method:<12} {P:>4} {thr_str:>5} {s['count']:>6}  "
              f"{s['cov_mean']:>9.3f} {s['cov_p50']:>8.3f} {s['cov_p10']:>8.3f}  "
              f"{s['waste_mean']:>10.3f} {s['waste_p50']:>9.3f}  "
              f"{s['N_mean']:>7.0f} {s['loadMB_mean']:>11.3f}")

    # write summary csv
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[saved] summary -> {out_csv}")

    # Also save the full per-row results
    detail_csv = out_csv.with_suffix(".detail.csv")
    with open(detail_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"[saved] detail -> {detail_csv}  ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
