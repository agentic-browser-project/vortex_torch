"""Compare three KV-fetch policies for sparse attention.

Given an "ideal" sparse selection of N needed tokens from a length-T pool,
we evaluate three policies:

  A0 (ideal)   : fetch exactly the N needed tokens (page_size=1)
                 — recall=1.0, waste=0, but smallest pages = highest kernel overhead
  A1 (any-hit) : pages of size P. Load every page that has ≥1 needed token.
                 — recall=1.0, waste = over-fetched / total fetched (lossless coarsening)
  A2 (ratio≥x) : pages of size P, threshold x ∈ [0, 1]. Load page only if
                 (needed_in_page / P) ≥ x. Drop other pages entirely.
                 — recall ≤ 1.0 (lossy), waste typically lower than A1

For each scenario we:
  1. Generate a "needed" mask of N tokens (random or clustered).
  2. Compute the page set each policy actually fetches.
  3. Compute (effective_fetched, recall, waste, MB).
  4. Invoke bench_decode_bandwidth_notc.py with (page_size=P, seq_len=eff)
     to get achieved GB/s + us/call.

Output: a per-row CSV with all columns, plus a one-line summary print.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Set

BENCH_SCRIPT = Path(__file__).resolve().parent.parent / "bench_decode_bandwidth_notc.py"
ENV_ROOT = os.path.expanduser("~/miniforge3/envs/vortex_v04")
PYTHON = f"{ENV_ROOT}/bin/python"

# ---------------------------------------------------------------------------
# Selection generators
# ---------------------------------------------------------------------------

def gen_random_uniform(N: int, T: int, rng: random.Random) -> Set[int]:
    """N distinct token indices, uniform random over [0, T)."""
    return set(rng.sample(range(T), N))

def gen_clustered(N: int, T: int, rng: random.Random, n_clusters: int = 8,
                  cluster_std: int = 64) -> Set[int]:
    """N indices distributed around `n_clusters` centers (Gaussian-ish)."""
    centers = rng.sample(range(T), n_clusters)
    chosen: Set[int] = set()
    attempts = 0
    while len(chosen) < N and attempts < N * 20:
        attempts += 1
        c = centers[rng.randrange(n_clusters)]
        offset = int(rng.gauss(0, cluster_std))
        idx = c + offset
        if 0 <= idx < T:
            chosen.add(idx)
    # fall-back top-up if we couldn't reach N
    while len(chosen) < N:
        chosen.add(rng.randrange(T))
    return chosen


# ---------------------------------------------------------------------------
# Page-set computation per policy
# ---------------------------------------------------------------------------

def compute_pages_a0(needed: Set[int], T: int) -> tuple[int, int, int, float, float]:
    """A0: per-token (page_size=1). Loads exactly the needed tokens."""
    N = len(needed)
    return (1, N, N, 1.0, 0.0)  # page_size, eff_seq_len, recall_num, recall, waste

def compute_pages_a1(needed: Set[int], T: int, P: int) -> tuple[int, int, int, float, float]:
    """A1: any-hit. Load every page with ≥1 needed token."""
    N = len(needed)
    pages_hit: Set[int] = {idx // P for idx in needed}
    eff = len(pages_hit) * P
    recall = 1.0  # never drops a needed token
    waste = (eff - N) / eff if eff > 0 else 0.0
    return (P, eff, N, recall, waste)

def compute_pages_a2(needed: Set[int], T: int, P: int, x: float) -> tuple[int, int, int, float, float]:
    """A2: threshold x. Load page if (needed_in_page / P) ≥ x; else drop."""
    N = len(needed)
    # count needed per page
    counts: dict[int, int] = {}
    for idx in needed:
        p = idx // P
        counts[p] = counts.get(p, 0) + 1
    # min hits needed in a page to keep it
    min_hits = max(1, int((x * P) + 0.999_999))   # ceil(x * P), at least 1
    if x <= 0.0:
        min_hits = 1
    kept_pages = [p for p, c in counts.items() if c >= min_hits]
    eff = len(kept_pages) * P
    needed_kept = sum(counts[p] for p in kept_pages)
    recall = needed_kept / N if N > 0 else 0.0
    waste = (eff - needed_kept) / eff if eff > 0 else 0.0
    return (P, eff, needed_kept, recall, waste)


# ---------------------------------------------------------------------------
# Bench invocation
# ---------------------------------------------------------------------------

BW_RE = re.compile(r"^wrapper\s+([\d.]+)\s*us\s+([\d.]+)\s*GB/s", re.M)

def run_bench(page_size: int, eff_seq: int, total_seq: int,
              num_qo: int, num_kv: int, head_dim: int,
              iters: int = 50, warmup: int = 5) -> Optional[tuple[float, float]]:
    """Run the patched bench, return (us/call, GB/s) or None on failure."""
    if eff_seq <= 0:
        return (0.0, 0.0)  # nothing to fetch — degenerate
    # round eff_seq down to multiple of page_size (bench requirement)
    eff = (eff_seq // page_size) * page_size
    if eff < page_size:
        eff = page_size
    if eff > total_seq:
        eff = (total_seq // page_size) * page_size

    cmd = [
        PYTHON, str(BENCH_SCRIPT),
        "--batch-size", "1",
        "--total-seq-len", str(total_seq),
        "--seq-len", str(eff),
        "--num-qo-heads", str(num_qo),
        "--num-kv-heads", str(num_kv),
        "--head-dim", str(head_dim),
        "--page-size", str(page_size),
        "--kv-dtype", "bf16",
        "--warmup", str(warmup), "--iters", str(iters),
    ]
    env = os.environ.copy()
    env["CUDA_HOME"] = ENV_ROOT
    env["PATH"] = f"{ENV_ROOT}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{ENV_ROOT}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        m = BW_RE.search(r.stdout + "\n" + r.stderr)
        if m:
            return (float(m.group(1)), float(m.group(2)))
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-seq", type=int, default=32768)
    ap.add_argument("--needed-list", type=int, nargs="+", default=[256, 2048, 8192])
    ap.add_argument("--page-list", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--x-list", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--distributions", nargs="+", default=["random", "clustered"])
    ap.add_argument("--num-qo-heads", type=int, default=16)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default="logs/page_policy_sweep.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # KV bytes per token: num_kv_heads * head_dim * 2 (K+V) * 2 (bf16)
    bytes_per_tok = args.num_kv_heads * args.head_dim * 2 * 2

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fields = ["algo", "dist", "N", "P", "x", "eff_seq", "MB",
              "recall", "waste", "us_per_call", "GB_per_s"]

    print(f"{'algo':<8} {'dist':>10} {'N':>6} {'P':>3} {'x':>5} "
          f"{'eff':>6} {'MB':>7} {'recall':>7} {'waste':>7} {'us':>7} {'GB/s':>8}")
    print("-" * 92)

    def emit(row):
        rows.append(row)
        print(f"{row['algo']:<8} {row['dist']:>10} {row['N']:>6} "
              f"{row['P']:>3} {row['x']:>5} "
              f"{row['eff_seq']:>6} {row['MB']:>7.2f} "
              f"{row['recall']:>7.3f} {row['waste']:>7.3f} "
              f"{row['us_per_call']:>7.1f} {row['GB_per_s']:>8.2f}")

    # ---- iterate all scenarios ----
    for dist_name in args.distributions:
        for N in args.needed_list:
            if N > args.total_seq:
                continue
            # generate the needed set once per (dist, N)
            if dist_name == "random":
                needed = gen_random_uniform(N, args.total_seq, rng)
            elif dist_name == "clustered":
                needed = gen_clustered(N, args.total_seq, rng)
            else:
                raise ValueError(dist_name)
            assert len(needed) == N

            # --- A0 baseline (page_size=1) ---
            P, eff, _, recall, waste = compute_pages_a0(needed, args.total_seq)
            bench = run_bench(P, eff, args.total_seq,
                              args.num_qo_heads, args.num_kv_heads, args.head_dim)
            if bench is None: bench = (-1.0, -1.0)
            emit({"algo": "A0", "dist": dist_name, "N": N, "P": P, "x": 0.0,
                  "eff_seq": eff, "MB": eff * bytes_per_tok / 1e6,
                  "recall": recall, "waste": waste,
                  "us_per_call": bench[0], "GB_per_s": bench[1]})

            for P in args.page_list:
                # --- A1 any-hit ---
                _, eff, _, recall, waste = compute_pages_a1(needed, args.total_seq, P)
                bench = run_bench(P, eff, args.total_seq,
                                  args.num_qo_heads, args.num_kv_heads, args.head_dim)
                if bench is None: bench = (-1.0, -1.0)
                emit({"algo": "A1", "dist": dist_name, "N": N, "P": P, "x": 0.0,
                      "eff_seq": eff, "MB": eff * bytes_per_tok / 1e6,
                      "recall": recall, "waste": waste,
                      "us_per_call": bench[0], "GB_per_s": bench[1]})

                # --- A2 threshold ---
                for x in args.x_list:
                    _, eff, _, recall, waste = compute_pages_a2(needed, args.total_seq, P, x)
                    bench = run_bench(P, eff, args.total_seq,
                                      args.num_qo_heads, args.num_kv_heads, args.head_dim)
                    if bench is None: bench = (-1.0, -1.0)
                    emit({"algo": "A2", "dist": dist_name, "N": N, "P": P, "x": x,
                          "eff_seq": eff, "MB": eff * bytes_per_tok / 1e6,
                          "recall": recall, "waste": waste,
                          "us_per_call": bench[0], "GB_per_s": bench[1]})

    # write CSV
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
