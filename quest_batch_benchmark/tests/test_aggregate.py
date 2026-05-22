import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aggregate_results import aggregate

RAW_FIELDS = [
    "run_timestamp", "attention", "batch_size", "model", "topk_val",
    "input_tokens", "max_tokens", "repeat_idx", "tokens_generated",
    "ttft_ms", "tpot_ms", "decode_time_ms", "total_time_ms",
    "throughput_tok_s", "status",
]


def _write_raw(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(attention, batch_size, repeat_idx, tpot, status="ok", **over):
    r = {
        "run_timestamp": "t", "attention": attention, "batch_size": batch_size,
        "model": "Qwen3-8B", "topk_val": 64, "input_tokens": 9661,
        "max_tokens": 256, "repeat_idx": repeat_idx, "tokens_generated": 256,
        "ttft_ms": "800.0", "tpot_ms": tpot, "decode_time_ms": "3000.0",
        "total_time_ms": "3800.0", "throughput_tok_s": "70.0", "status": status,
    }
    r.update(over)
    return r


def test_aggregate_computes_per_config_mean(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "tpot.csv"
    _write_raw(raw, [
        _row("dense", 1, 0, "10.0"), _row("dense", 1, 1, "20.0"),
        _row("quest", 1, 0, "4.0"), _row("quest", 1, 1, "6.0"),
    ])
    aggregate(str(raw), str(out))
    got = {(r["attention"], r["batch_size"]): r for r in csv.DictReader(open(out))}
    assert float(got[("dense", "1")]["tpot_ms_mean"]) == 15.0
    assert float(got[("quest", "1")]["tpot_ms_mean"]) == 5.0
    assert got[("dense", "1")]["status"] == "ok"
    assert got[("dense", "1")]["repeat"] == "2"
    # non-tpot metric columns are aggregated as a plain mean
    assert float(got[("dense", "1")]["ttft_ms_mean"]) == 800.0


def test_aggregate_marks_capped(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "tpot.csv"
    _write_raw(raw, [
        _row("dense", 32, 0, "50.0", status="capped"),
        _row("dense", 32, 1, "52.0", status="capped"),
    ])
    aggregate(str(raw), str(out))
    got = {(r["attention"], r["batch_size"]): r for r in csv.DictReader(open(out))}
    # a capped config still carries real TPOT statistics
    row = got[("dense", "32")]
    assert row["status"] == "capped"
    assert float(row["tpot_ms_mean"]) == 51.0
    assert float(row["tpot_ms_std"]) == 1.0       # population std of [50, 52]
    assert float(row["tpot_ms_min"]) == 50.0
    assert float(row["tpot_ms_max"]) == 52.0


def test_aggregate_marks_error(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "tpot.csv"
    err = _row("dense", 64, -1, "", status="error",
               tokens_generated="", ttft_ms="", decode_time_ms="",
               total_time_ms="", throughput_tok_s="")
    _write_raw(raw, [err])
    aggregate(str(raw), str(out))
    got = {(r["attention"], r["batch_size"]): r for r in csv.DictReader(open(out))}
    assert got[("dense", "64")]["status"] == "error"
    assert got[("dense", "64")]["tpot_ms_mean"] == ""


def test_aggregate_error_row_dominates_ok_rows(tmp_path):
    # a partial-failure group (some repeats ok, then one errored) must be
    # reported as error -- the ok rows must not mask the failure
    raw = tmp_path / "raw.csv"
    out = tmp_path / "tpot.csv"
    err = _row("dense", 8, -1, "", status="error",
               tokens_generated="", ttft_ms="", decode_time_ms="",
               total_time_ms="", throughput_tok_s="")
    _write_raw(raw, [
        _row("dense", 8, 0, "12.0"), _row("dense", 8, 1, "12.5"), err,
    ])
    aggregate(str(raw), str(out))
    got = {(r["attention"], r["batch_size"]): r for r in csv.DictReader(open(out))}
    assert got[("dense", "8")]["status"] == "error"
    assert got[("dense", "8")]["tpot_ms_mean"] == ""
