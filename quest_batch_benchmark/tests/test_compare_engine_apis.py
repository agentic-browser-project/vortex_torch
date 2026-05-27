import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compare_engine_apis import build_comparison_rows, render_markdown


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _agg_row(attention, batch_size, tpot_mean):
    return {
        "attention": attention, "batch_size": str(batch_size),
        "model": "Qwen3-VL-8B-Instruct", "topk_val": "64",
        "input_tokens": "9661", "max_tokens": "256", "repeat": "3",
        "status": "ok", "tpot_ms_mean": f"{tpot_mean:.6f}",
        "tpot_ms_std": "0.0", "tpot_ms_min": f"{tpot_mean:.6f}",
        "tpot_ms_max": f"{tpot_mean:.6f}", "ttft_ms_mean": "100.0",
        "decode_time_ms_mean": "0.0", "total_time_ms_mean": "0.0",
        "throughput_tok_s_mean": "0.0",
    }


_FIELDS = ["attention", "batch_size", "model", "topk_val", "input_tokens",
           "max_tokens", "repeat", "status", "tpot_ms_mean", "tpot_ms_std",
           "tpot_ms_min", "tpot_ms_max", "ttft_ms_mean", "decode_time_ms_mean",
           "total_time_ms_mean", "throughput_tok_s_mean"]


def test_join_matches_on_attention_and_batch(tmp_path):
    direct = tmp_path / "direct.csv"
    gengine = tmp_path / "gengine.csv"
    _write(direct, [_agg_row("quest", 1, 11.0), _agg_row("quest", 64, 65.0)],
           _FIELDS)
    _write(gengine, [_agg_row("quest", 1, 11.2), _agg_row("quest", 64, 65.5)],
           _FIELDS)
    rows = build_comparison_rows(str(direct), str(gengine))
    assert len(rows) == 2
    by_bs = {int(r["batch_size"]): r for r in rows}
    assert by_bs[1]["tpot_ms_direct"] == "11.000"
    assert by_bs[1]["tpot_ms_get_engine"] == "11.200"
    assert abs(float(by_bs[1]["abs_diff_ms"]) - 0.2) < 1e-9
    assert abs(float(by_bs[1]["ratio_get_engine_over_direct"]) - (11.2 / 11.0)) < 1e-5


def test_join_handles_missing_pair(tmp_path):
    """A row that exists only in one CSV is dropped (not silently zeroed)."""
    direct = tmp_path / "direct.csv"
    gengine = tmp_path / "gengine.csv"
    _write(direct, [_agg_row("quest", 1, 11.0), _agg_row("quest", 64, 65.0)],
           _FIELDS)
    _write(gengine, [_agg_row("quest", 1, 11.2)], _FIELDS)  # missing bs=64
    rows = build_comparison_rows(str(direct), str(gengine))
    assert len(rows) == 1
    assert int(rows[0]["batch_size"]) == 1


def test_markdown_table_has_expected_header(tmp_path):
    rows = [{
        "attention": "quest", "batch_size": "1",
        "tpot_ms_direct": "11.000", "tpot_ms_get_engine": "11.200",
        "abs_diff_ms": "0.200000", "ratio_get_engine_over_direct": "1.018182",
    }]
    md = render_markdown(rows)
    assert "| attention | batch size | direct TPOT (ms) | get_engine TPOT (ms) | abs diff (ms) | get_engine / direct |" in md
    assert "quest" in md
    assert "11.000" in md
