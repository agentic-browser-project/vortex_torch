#!/usr/bin/env python3
"""Compare TPOT between the no-CUDA-graph and CUDA-graph sweeps.

Reads two aggregated TPOT-vs-batch-size CSVs (produced by aggregate_results.py)
-- one from the no-graph baseline, one from the `--enable-cuda-graph` sweep --
joins on (attention, batch_size), and writes a side-by-side table with
absolute and relative deltas. Outputs both CSV (machine-readable) and
markdown (the table you paste into the README). Mirrors compare_engine_apis.py.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


_OUT_FIELDS = [
    "attention", "batch_size",
    "tpot_ms_nograph", "tpot_ms_cudagraph",
    "abs_diff_ms", "speedup_cudagraph_over_nograph",
]


def _load(path: str) -> Dict[tuple, Dict[str, str]]:
    """Return {(attention, batch_size_int): row} from an aggregated CSV.

    Skips rows whose status is 'error' or whose tpot_ms_mean is blank --
    aggregate_results.py emits a row with empty metric columns for any
    (attention, batch_size) group whose repeats all errored. Treating those
    as missing measurements (drop, don't crash) is symmetric with the
    inner-join "missing pair" semantics in build_comparison_rows.
    """
    out: Dict[tuple, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "error" or not r.get("tpot_ms_mean"):
                continue
            out[(r["attention"], int(r["batch_size"]))] = r
    return out


def build_comparison_rows(nograph_csv: str, cudagraph_csv: str) -> List[Dict[str, str]]:
    """Join two aggregated CSVs on (attention, batch_size); drop unmatched rows."""
    nograph = _load(nograph_csv)
    cudagraph = _load(cudagraph_csv)
    rows: List[Dict[str, str]] = []
    for key in sorted(nograph.keys() & cudagraph.keys(),
                      key=lambda kv: (kv[0], kv[1])):
        attention, batch_size = key
        tn = float(nograph[key]["tpot_ms_mean"])
        tc = float(cudagraph[key]["tpot_ms_mean"])
        rows.append({
            "attention": attention,
            "batch_size": str(batch_size),
            "tpot_ms_nograph": f"{tn:.3f}",
            "tpot_ms_cudagraph": f"{tc:.3f}",
            "abs_diff_ms": f"{tc - tn:.6f}",
            # speedup > 1 means CUDA graph is faster than no-graph
            "speedup_cudagraph_over_nograph": (
                f"{tn / tc:.6f}" if tc > 0 else "nan"
            ),
        })
    return rows


def render_markdown(rows: List[Dict[str, str]]) -> str:
    header = ("| attention | batch size | no-graph TPOT (ms) | CUDA-graph TPOT (ms) "
              "| abs diff (ms) | speedup (no-graph / CUDA-graph) |")
    sep = "|---|---:|---:|---:|---:|---:|"
    body = []
    for r in rows:
        body.append(
            f"| {r['attention']} | {r['batch_size']} | "
            f"{r['tpot_ms_nograph']} | {r['tpot_ms_cudagraph']} | "
            f"{float(r['abs_diff_ms']):+.3f} | "
            f"{r['speedup_cudagraph_over_nograph']} |"
        )
    return "\n".join([header, sep, *body])


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nograph-csv",
                   default=str(here / "results" / "tpot_vs_batchsize.csv"),
                   help="No-graph aggregated CSV. Default: the headline "
                        "results/tpot_vs_batchsize.csv (matches the README).")
    p.add_argument("--cudagraph-csv",
                   default=str(here / "results" / "tpot_vs_batchsize_cudagraph.csv"),
                   help="CUDA-graph aggregated CSV.")
    p.add_argument("--out-csv",
                   default=str(here / "results" / "cuda_graph_comparison.csv"))
    p.add_argument("--out-md",
                   default=str(here / "results" / "cuda_graph_comparison.md"))
    args = p.parse_args()

    rows = build_comparison_rows(args.nograph_csv, args.cudagraph_csv)
    if not rows:
        raise SystemExit(
            f"no joined rows -- check that {args.nograph_csv} and "
            f"{args.cudagraph_csv} share (attention, batch_size) keys"
        )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    md = render_markdown(rows)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    print(f"[compare] wrote {args.out_csv}")
    print(f"[compare] wrote {args.out_md}")
    print(md)


if __name__ == "__main__":
    main()
