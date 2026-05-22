import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from treesparse_results import OUT_FIELDS, treesparse_rows


def _entry(tpot_median, tpot_mean, tpot_std=0.5, prefill=9655):
    """A TreeSparse per-batch-size JSON entry (the shape benchmark_batch.py emits)."""
    return {
        "batch_size": 0, "repetitions": 3,  # "batch_size" here is not read by treesparse_rows
        "tpot_mean_ms": tpot_mean, "tpot_median_ms": tpot_median,
        "tpot_std_ms": tpot_std, "throughput_mean": 90.0,
        "throughput_median": 95.0, "throughput_std": 1.0,
        "prefill_len": prefill, "num_decode_tokens": 256,
    }


def test_uses_median_not_mean():
    # the cold first rep inflates tpot_mean_ms; the row must carry the median
    raw = {"1": _entry(tpot_median=10.5, tpot_mean=12.7)}
    rows = treesparse_rows(raw, [1], top_k=128)
    assert len(rows) == 1
    assert float(rows[0]["tpot_ms_mean"]) == 10.5
    assert float(rows[0]["throughput_tok_s_mean"]) == 95.0  # throughput_median, not throughput_mean (90.0)
    assert rows[0]["attention"] == "treesparse"
    assert rows[0]["status"] == "ok"
    assert int(rows[0]["batch_size"]) == 1
    assert rows[0]["topk_val"] == 128


def test_row_has_exact_out_fields():
    raw = {"1": _entry(tpot_median=10.5, tpot_mean=12.7)}
    rows = treesparse_rows(raw, [1], top_k=128)
    assert set(rows[0].keys()) == set(OUT_FIELDS)


def test_missing_batch_size_is_error_row():
    # benchmark_batch.py stops at the first OOM -> bs 64 absent from the JSON
    raw = {"1": _entry(tpot_median=10.5, tpot_mean=12.7)}
    rows = treesparse_rows(raw, [1, 64], top_k=128)
    by_bs = {int(r["batch_size"]): r for r in rows}
    assert by_bs[64]["status"] == "error"
    assert by_bs[64]["tpot_ms_mean"] == ""
    assert by_bs[64]["repeat"] == 0
    assert set(by_bs[64].keys()) == set(OUT_FIELDS)


def test_rows_sorted_by_batch_size():
    raw = {"1": _entry(10.0, 11.0), "8": _entry(20.0, 21.0),
           "2": _entry(12.0, 13.0)}
    rows = treesparse_rows(raw, [8, 1, 2], top_k=128)
    assert [int(r["batch_size"]) for r in rows] == [1, 2, 8]
