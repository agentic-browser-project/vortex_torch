#!/usr/bin/env python3
"""Collapse the per-repeat raw CSV into a per-config TPOT-vs-batch-size CSV."""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

OUT_FIELDS = [
    "attention", "batch_size", "model", "topk_val", "input_tokens",
    "max_tokens", "repeat", "status", "tpot_ms_mean", "tpot_ms_std",
    "tpot_ms_min", "tpot_ms_max", "ttft_ms_mean", "decode_time_ms_mean",
    "total_time_ms_mean", "throughput_tok_s_mean",
]

# raw-CSV metric column -> processed-CSV column (aggregated as a plain mean)
_MEAN_COLS = {
    "ttft_ms": "ttft_ms_mean",
    "decode_time_ms": "decode_time_ms_mean",
    "total_time_ms": "total_time_ms_mean",
    "throughput_tok_s": "throughput_tok_s_mean",
}


def _floats(rows, col):
    """Parsed float values of `col` over `rows`, skipping blank cells."""
    out = []
    for r in rows:
        v = r.get(col, "")
        if v != "" and v is not None:
            out.append(float(v))
    return out


def aggregate(raw_csv: str, out_csv: str) -> None:
    """Read the raw per-repeat CSV; write one aggregated row per (attention, batch_size)."""
    with open(raw_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["attention"], int(r["batch_size"]))].append(r)

    out_rows = []
    for (attention, batch_size), grp in sorted(groups.items(),
                                               key=lambda kv: (kv[0][0], kv[0][1])):
        sample = grp[0]
        base = {
            "attention": attention,
            "batch_size": batch_size,
            "model": sample["model"],
            "topk_val": sample["topk_val"],
            "input_tokens": sample["input_tokens"],
            "max_tokens": sample["max_tokens"],
        }
        measured = [r for r in grp if r["status"] in ("ok", "capped")]
        tpots = _floats(measured, "tpot_ms")
        # An error row anywhere in the group fails the whole config -- even if
        # some repeats succeeded, a later one crashing means the config is not
        # a trustworthy measurement, so don't let the ok rows mask it.
        has_error = any(r["status"] == "error" for r in grp)
        if tpots and not has_error:
            # capped if any measured repeat was wave-serialized, else ok
            status = "capped" if any(r["status"] == "capped" for r in measured) else "ok"
            row = {
                **base,
                "repeat": len(measured),
                "status": status,
                "tpot_ms_mean": statistics.fmean(tpots),
                "tpot_ms_std": statistics.pstdev(tpots) if len(tpots) > 1 else 0.0,
                "tpot_ms_min": min(tpots),
                "tpot_ms_max": max(tpots),
            }
            for src, dst in _MEAN_COLS.items():
                vals = _floats(measured, src)
                row[dst] = statistics.fmean(vals) if vals else ""
            row = {k: (f"{v:.6f}" if isinstance(v, float) else v)
                   for k, v in row.items()}
            out_rows.append(row)
        else:
            # error -- a failure row in the group, or no usable measurement;
            # blank the statistic columns
            blanks = {k: "" for k in OUT_FIELDS
                      if k.endswith(("_mean", "_std", "_min", "_max"))}
            out_rows.append({**base, "repeat": 0, "status": "error", **blanks})

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[aggregate] {len(out_rows)} config rows -> {out_csv}")


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Aggregate raw decode latencies into TPOT-vs-batch-size")
    p.add_argument("--raw-csv", default=str(here / "results" / "raw_results.csv"))
    p.add_argument("--out-csv", default=str(here / "results" / "tpot_vs_batchsize.csv"))
    args = p.parse_args()
    aggregate(args.raw_csv, args.out_csv)


if __name__ == "__main__":
    main()
