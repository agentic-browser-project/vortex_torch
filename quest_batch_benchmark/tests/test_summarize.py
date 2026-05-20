import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark_quest_tpot import summarize_step_latencies


def test_summarize_basic_stats():
    s = summarize_step_latencies([10.0, 20.0, 30.0, 40.0])
    assert s["tpot_ms_mean"] == 25.0
    assert s["tpot_ms_min"] == 10.0
    assert s["tpot_ms_max"] == 40.0
    assert s["tpot_ms_p50"] == 25.0  # linear-interpolated median


def test_summarize_single_value():
    s = summarize_step_latencies([7.5])
    assert s["tpot_ms_mean"] == 7.5
    assert s["tpot_ms_p50"] == 7.5
    assert s["tpot_ms_p90"] == 7.5
    assert s["tpot_ms_std"] == 0.0
