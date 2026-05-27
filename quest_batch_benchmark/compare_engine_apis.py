#!/usr/bin/env python3
"""Compare TPOT between the two engine-construction paths.

Reads two aggregated TPOT-vs-batch-size CSVs (produced by aggregate_results.py)
-- one from `--engine-api direct`, one from `--engine-api get_engine` -- joins
on (attention, batch_size), and writes a side-by-side table with absolute and
relative deltas. Outputs both CSV (machine-readable) and markdown (the table
you paste into a report).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


_OUT_FIELDS = [
    "attention", "batch_size",
    "tpot_ms_direct", "tpot_ms_get_engine",
    "abs_diff_ms", "ratio_get_engine_over_direct",
]


def _load(path: str) -> Dict[tuple, Dict[str, str]]:
    """Return {(attention, batch_size_int): row} from an aggregated CSV."""
    out: Dict[tuple, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["attention"], int(r["batch_size"]))] = r
    return out


def build_comparison_rows(direct_csv: str, get_engine_csv: str) -> List[Dict[str, str]]:
    """Join two aggregated CSVs on (attention, batch_size); drop unmatched rows."""
    direct = _load(direct_csv)
    gengine = _load(get_engine_csv)
    rows: List[Dict[str, str]] = []
    for key in sorted(direct.keys() & gengine.keys(),
                      key=lambda kv: (kv[0], kv[1])):
        attention, batch_size = key
        td = float(direct[key]["tpot_ms_mean"])
        tg = float(gengine[key]["tpot_ms_mean"])
        rows.append({
            "attention": attention,
            "batch_size": str(batch_size),
            "tpot_ms_direct": f"{td:.3f}",
            "tpot_ms_get_engine": f"{tg:.3f}",
            "abs_diff_ms": f"{tg - td:.6f}",
            "ratio_get_engine_over_direct": (
                f"{tg / td:.6f}" if td > 0 else "nan"
            ),
        })
    return rows


def render_markdown(rows: List[Dict[str, str]]) -> str:
    header = ("| attention | batch size | direct TPOT (ms) | get_engine TPOT (ms) | "
              "abs diff (ms) | get_engine / direct |")
    sep = "|---|---:|---:|---:|---:|---:|"
    body = []
    for r in rows:
        body.append(
            f"| {r['attention']} | {r['batch_size']} | "
            f"{r['tpot_ms_direct']} | {r['tpot_ms_get_engine']} | "
            f"{float(r['abs_diff_ms']):+.3f} | "
            f"{r['ratio_get_engine_over_direct']} |"
        )
    return "\n".join([header, sep, *body])


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct-csv",
                   default=str(here / "results" / "tpot_vs_batchsize_direct.csv"))
    p.add_argument("--get-engine-csv",
                   default=str(here / "results" / "tpot_vs_batchsize_get_engine.csv"))
    p.add_argument("--out-csv",
                   default=str(here / "results" / "engine_api_comparison.csv"))
    p.add_argument("--out-md",
                   default=str(here / "results" / "engine_api_comparison.md"))
    args = p.parse_args()

    rows = build_comparison_rows(args.direct_csv, args.get_engine_csv)
    if not rows:
        raise SystemExit(
            f"no joined rows -- check that {args.direct_csv} and "
            f"{args.get_engine_csv} share (attention, batch_size) keys"
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
