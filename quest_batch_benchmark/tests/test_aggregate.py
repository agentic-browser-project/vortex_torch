import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aggregate_results import aggregate

RAW_FIELDS = [
    "run_timestamp", "attention", "batch_size", "model", "topk_val",
    "input_tokens", "warmup_steps", "measured_steps", "step_idx",
    "step_latency_ms", "status",
]


def _write_raw(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _ok_row(attention, batch_size, step_idx, latency):
    return {
        "run_timestamp": "t", "attention": attention, "batch_size": batch_size,
        "model": "Qwen3-8B", "topk_val": 64, "input_tokens": 8000,
        "warmup_steps": 16, "measured_steps": 2, "step_idx": step_idx,
        "step_latency_ms": latency, "status": "ok",
    }


def test_aggregate_computes_per_config_mean(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "tpot.csv"
    _write_raw(raw, [
        _ok_row("dense", 1, 0, "10.0"), _ok_row("dense", 1, 1, "20.0"),
        _ok_row("quest", 1, 0, "4.0"), _ok_row("quest", 1, 1, "6.0"),
    ])
    aggregate(str(raw), str(out))
    got = {(r["attention"], r["batch_size"]): r for r in csv.DictReader(open(out))}
    assert float(got[("dense", "1")]["tpot_ms_mean"]) == 15.0
    assert float(got[("quest", "1")]["tpot_ms_mean"]) == 5.0
    assert got[("dense", "1")]["status"] == "ok"


def test_aggregate_marks_oom(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "tpot.csv"
    oom = _ok_row("dense", 64, -1, "")
    oom["status"] = "oom"
    _write_raw(raw, [_ok_row("dense", 1, 0, "10.0"), oom])
    aggregate(str(raw), str(out))
    got = {(r["attention"], r["batch_size"]): r for r in csv.DictReader(open(out))}
    assert got[("dense", "64")]["status"] == "oom"
    assert got[("dense", "64")]["tpot_ms_mean"] == ""
