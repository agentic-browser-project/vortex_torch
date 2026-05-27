import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compare_cuda_graph import build_comparison_rows, render_markdown


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
    nograph = tmp_path / "nograph.csv"
    cudagraph = tmp_path / "cudagraph.csv"
    _write(nograph, [_agg_row("dense", 1, 9.29), _agg_row("quest", 64, 65.23)],
           _FIELDS)
    _write(cudagraph, [_agg_row("dense", 1, 7.5), _agg_row("quest", 64, 60.0)],
           _FIELDS)
    rows = build_comparison_rows(str(nograph), str(cudagraph))
    assert len(rows) == 2
    by_key = {(r["attention"], int(r["batch_size"])): r for r in rows}
    d1 = by_key[("dense", 1)]
    assert d1["tpot_ms_nograph"] == "9.290"
    assert d1["tpot_ms_cudagraph"] == "7.500"
    assert abs(float(d1["abs_diff_ms"]) - (7.5 - 9.29)) < 1e-5
    # speedup of CUDA graph over no-graph = nograph / cudagraph
    # tolerance 1e-5 (not 1e-9) because the row stores the ratio with :.6f
    # precision, matching compare_engine_apis' established formatter.
    assert abs(float(d1["speedup_cudagraph_over_nograph"]) - (9.29 / 7.5)) < 1e-5


def test_join_handles_missing_pair(tmp_path):
    """A row that exists only in one CSV is dropped (not silently zeroed)."""
    nograph = tmp_path / "nograph.csv"
    cudagraph = tmp_path / "cudagraph.csv"
    _write(nograph, [_agg_row("dense", 1, 9.29), _agg_row("quest", 64, 65.23)],
           _FIELDS)
    _write(cudagraph, [_agg_row("dense", 1, 7.5)], _FIELDS)  # quest missing
    rows = build_comparison_rows(str(nograph), str(cudagraph))
    assert len(rows) == 1
    assert rows[0]["attention"] == "dense"
    assert int(rows[0]["batch_size"]) == 1


def test_markdown_table_has_expected_header():
    rows = [{
        "attention": "quest", "batch_size": "64",
        "tpot_ms_nograph": "65.230", "tpot_ms_cudagraph": "60.000",
        "abs_diff_ms": "-5.230000",
        "speedup_cudagraph_over_nograph": "1.087167",
    }]
    md = render_markdown(rows)
    assert ("| attention | batch size | no-graph TPOT (ms) | CUDA-graph TPOT (ms) "
            "| abs diff (ms) | speedup (no-graph / CUDA-graph) |") in md
    assert "quest" in md
    assert "65.230" in md


def test_error_rows_are_dropped(tmp_path):
    """A status=error row (blank tpot_ms_mean) drops out instead of crashing."""
    nograph = tmp_path / "nograph.csv"
    cudagraph = tmp_path / "cudagraph.csv"
    bad_row = dict(_agg_row("quest", 64, 0.0))
    bad_row["status"] = "error"
    bad_row["tpot_ms_mean"] = ""  # aggregate_results.py shape for error groups
    _write(nograph, [_agg_row("dense", 1, 9.29), bad_row], _FIELDS)
    _write(cudagraph, [_agg_row("dense", 1, 7.5), bad_row], _FIELDS)
    rows = build_comparison_rows(str(nograph), str(cudagraph))
    assert len(rows) == 1
    assert rows[0]["attention"] == "dense"


def test_main_exits_on_empty_join(tmp_path, monkeypatch):
    """main() raises SystemExit if no (attention, batch_size) keys overlap."""
    import pytest
    from compare_cuda_graph import main as compare_main

    nograph = tmp_path / "nograph.csv"
    cudagraph = tmp_path / "cudagraph.csv"
    _write(nograph, [_agg_row("dense", 1, 9.29)], _FIELDS)
    _write(cudagraph, [_agg_row("quest", 1, 11.0)], _FIELDS)  # no overlap

    monkeypatch.setattr("sys.argv", [
        "compare_cuda_graph.py",
        "--nograph-csv", str(nograph),
        "--cudagraph-csv", str(cudagraph),
        "--out-csv", str(tmp_path / "out.csv"),
        "--out-md", str(tmp_path / "out.md"),
    ])
    with pytest.raises(SystemExit, match="no joined rows"):
        compare_main()
