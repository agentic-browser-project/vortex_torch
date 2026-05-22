#!/usr/bin/env python3
"""Merge dense + quest (tpot_vs_batchsize.csv) with TreeSparse
(treesparse_raw.json) into one three-way decode-TPOT comparison.

Writes `tpot_three_way.csv` (one OUT_FIELDS row per method x batch size) and
`comparison_table.md` (the human-readable TPOT table), and prints the table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aggregate_results import OUT_FIELDS
from treesparse_results import treesparse_rows

# render order within each batch size
_METHOD_ORDER = {"dense": 0, "quest": 1, "treesparse": 2}


def load_aggregated(csv_path: str) -> list[dict]:
    """dense + quest rows from the quest benchmark's aggregated CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if r["attention"] in ("dense", "quest")]


def merge(quest_csv: str, treesparse_json: str, batch_sizes: list[int],
          top_k: int) -> list[dict]:
    """All three methods' rows, sorted by (batch_size, method)."""
    rows = load_aggregated(quest_csv)
    with open(treesparse_json, encoding="utf-8") as f:
        raw = json.load(f)
    rows = rows + treesparse_rows(raw, batch_sizes, top_k)
    rows.sort(key=lambda r: (int(r["batch_size"]),
                             _METHOD_ORDER.get(r["attention"], 9)))
    return rows


def _tpot(rows: list[dict], attention: str, bs: int):
    """tpot_ms_mean for one (attention, batch size), or None if missing/blank."""
    for r in rows:
        if r["attention"] == attention and int(r["batch_size"]) == bs:
            v = r.get("tpot_ms_mean", "")
            return float(v) if v not in ("", None) else None
    return None


def format_table(rows: list[dict], batch_sizes: list[int]) -> str:
    """Markdown TPOT table: dense / quest / treesparse + speedups vs dense."""
    header = (
        "| batch size | dense TPOT (ms) | quest TPOT (ms) "
        "| treesparse TPOT (ms) | quest vs dense | treesparse vs dense |"
    )
    sep = (
        "|-----------:|----------------:|----------------:"
        "|---------------------:|---------------:|--------------------:|"
    )
    lines = [header, sep]
    for bs in sorted(batch_sizes):
        d = _tpot(rows, "dense", bs)
        q = _tpot(rows, "quest", bs)
        t = _tpot(rows, "treesparse", bs)
        ds = f"{d:.2f}" if d is not None else "—"
        qs = f"{q:.2f}" if q is not None else "—"
        ts = f"{t:.2f}" if t is not None else "—"
        qsp = f"{d / q:.2f}x" if (d is not None and q) else "—"
        tsp = f"{d / t:.2f}x" if (d is not None and t) else "—"
        lines.append(f"| {bs} | {ds} | {qs} | {ts} | {qsp} | {tsp} |")
    return "\n".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Three-way TPOT comparison")
    p.add_argument("--quest-csv",
                   default=str(here / "results" / "tpot_vs_batchsize.csv"))
    p.add_argument("--treesparse-json",
                   default=str(here / "results" / "treesparse_raw.json"))
    p.add_argument("--out-csv",
                   default=str(here / "results" / "tpot_three_way.csv"))
    p.add_argument("--out-md",
                   default=str(here / "results" / "comparison_table.md"))
    p.add_argument("--batch-sizes",
                   type=lambda s: [int(x) for x in s.split(",")],
                   default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--top-k", type=int, default=128)
    args = p.parse_args()

    rows = merge(args.quest_csv, args.treesparse_json, args.batch_sizes,
                 args.top_k)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    table = format_table(rows, args.batch_sizes)
    Path(args.out_md).write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n[compare] {len(rows)} rows -> {args.out_csv}")
    print(f"[compare] table   -> {args.out_md}")


if __name__ == "__main__":
    main()
