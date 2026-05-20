#!/usr/bin/env python3
"""Collapse the per-step raw CSV into a per-config TPOT-vs-batch-size CSV."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from benchmark_quest_tpot import summarize_step_latencies

OUT_FIELDS = [
    "attention", "batch_size", "model", "topk_val", "input_tokens",
    "measured_steps", "status", "tpot_ms_mean", "tpot_ms_p50", "tpot_ms_p90",
    "tpot_ms_std", "tpot_ms_min", "tpot_ms_max",
]


def aggregate(raw_csv: str, out_csv: str) -> None:
    """Read the raw per-step CSV; write one aggregated row per (attention, batch_size)."""
    with open(raw_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["attention"], int(r["batch_size"]))].append(r)

    out_rows = []
    for (attention, batch_size), grp in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sample = grp[0]
        statuses = {r["status"] for r in grp}
        base = {
            "attention": attention,
            "batch_size": batch_size,
            "model": sample["model"],
            "topk_val": sample["topk_val"],
            "input_tokens": sample["input_tokens"],
            "measured_steps": sample["measured_steps"],
        }
        ok_latencies = [
            float(r["step_latency_ms"])
            for r in grp
            if r["status"] == "ok" and r["step_latency_ms"] != ""
        ]
        if ok_latencies:
            stats = summarize_step_latencies(ok_latencies)
            out_rows.append({**base, "status": "ok", **{k: f"{v:.6f}" for k, v in stats.items()}})
        else:
            # oom / error — no latencies; leave stat columns blank
            bad = "oom" if "oom" in statuses else "error"
            blanks = {k: "" for k in OUT_FIELDS if k.startswith("tpot_ms_")}
            out_rows.append({**base, "status": bad, **blanks})

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[aggregate] {len(out_rows)} config rows -> {out_csv}")


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Aggregate raw decode latencies into TPOT-vs-batch-size")
    p.add_argument("--raw-csv", default=str(here / "results" / "raw_results.csv"))
    p.add_argument("--out-csv", default=str(here / "results" / "tpot_vs_batchsize.csv"))
    args = p.parse_args()
    aggregate(args.raw_csv, args.out_csv)


if __name__ == "__main__":
    main()
