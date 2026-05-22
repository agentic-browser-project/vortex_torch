import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aggregate_results import OUT_FIELDS
from build_comparison import format_table, load_aggregated, merge


def _agg_row(attention, bs, tpot):
    """A row as it appears in the dense+quest aggregated CSV."""
    r = {k: "" for k in OUT_FIELDS}
    r.update(attention=attention, batch_size=bs, model="Qwen3-VL-8B-Instruct",
             topk_val=("" if attention == "dense" else 64),
             input_tokens=9661, max_tokens=256, repeat=3, status="ok",
             tpot_ms_mean=f"{tpot:.6f}")
    return r


def _write_quest_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _ts_entry(tpot_median):
    """A TreeSparse per-batch-size JSON entry."""
    return {"batch_size": 0, "repetitions": 3,  # "batch_size" inside the entry is not read by treesparse_rows
            "tpot_mean_ms": tpot_median + 2,
            "tpot_median_ms": tpot_median, "tpot_std_ms": 0.5,
            "throughput_mean": 90.0, "throughput_median": 95.0,
            "throughput_std": 1.0, "prefill_len": 9655,
            "num_decode_tokens": 256}


def test_merge_has_all_three_methods(tmp_path):
    qcsv, tjson = tmp_path / "q.csv", tmp_path / "t.json"
    _write_quest_csv(qcsv, [_agg_row("dense", 1, 9.5), _agg_row("quest", 1, 10.9)])
    tjson.write_text(json.dumps({"1": _ts_entry(11.0)}))
    rows = merge(str(qcsv), str(tjson), [1], top_k=128)
    assert sorted(r["attention"] for r in rows) == ["dense", "quest", "treesparse"]


def test_merge_sorted_by_batch_then_method(tmp_path):
    qcsv, tjson = tmp_path / "q.csv", tmp_path / "t.json"
    _write_quest_csv(qcsv, [
        _agg_row("dense", 1, 9.5), _agg_row("quest", 1, 10.9),
        _agg_row("dense", 8, 15.0), _agg_row("quest", 8, 17.8),
    ])
    tjson.write_text(json.dumps({"1": _ts_entry(11.0), "8": _ts_entry(25.6)}))
    rows = merge(str(qcsv), str(tjson), [1, 8], top_k=128)
    assert [(int(r["batch_size"]), r["attention"]) for r in rows] == [
        (1, "dense"), (1, "quest"), (1, "treesparse"),
        (8, "dense"), (8, "quest"), (8, "treesparse"),
    ]


def test_format_table_lists_three_tpots_and_speedups(tmp_path):
    qcsv, tjson = tmp_path / "q.csv", tmp_path / "t.json"
    _write_quest_csv(qcsv, [_agg_row("dense", 1, 10.0), _agg_row("quest", 1, 20.0)])
    tjson.write_text(json.dumps({"1": _ts_entry(5.0)}))
    rows = merge(str(qcsv), str(tjson), [1], top_k=128)
    table = format_table(rows, [1])
    # dense 10, quest 20, treesparse 5 -> quest 0.50x, treesparse 2.00x vs dense
    assert "| 1 | 10.00 | 20.00 | 5.00 | 0.50x | 2.00x |" in table


def test_format_table_handles_missing_treesparse_batch(tmp_path):
    qcsv, tjson = tmp_path / "q.csv", tmp_path / "t.json"
    _write_quest_csv(qcsv, [_agg_row("dense", 64, 71.8), _agg_row("quest", 64, 64.7)])
    tjson.write_text(json.dumps({}))   # TreeSparse OOMed before bs 64
    rows = merge(str(qcsv), str(tjson), [64], top_k=128)
    table = format_table(rows, [64])
    assert "| 64 | 71.80 | 64.70 | — | 1.11x | — |" in table


def test_load_aggregated_filters_other_methods(tmp_path):
    # only dense/quest rows survive; a stray third-method row is dropped
    qcsv = tmp_path / "q.csv"
    _write_quest_csv(qcsv, [
        _agg_row("dense", 1, 9.5),
        _agg_row("quest", 1, 10.9),
        _agg_row("other_method", 1, 5.0),
    ])
    rows = load_aggregated(str(qcsv))
    assert [r["attention"] for r in rows] == ["dense", "quest"]
