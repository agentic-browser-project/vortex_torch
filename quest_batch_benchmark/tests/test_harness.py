import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark_quest_tpot import (
    _WARMUP_TOKENS, build_engine_kwargs, classify_status, compute_metrics,
    kv_pool_capacity, warmup_batch,
)


def _args(attention, **over):
    d = dict(attention=attention, model_path="/models/Qwen3-8B", max_tokens=256,
             repeat=3, topk_val=64, enable_cuda_graph=False,
             mem_fraction_static=None, max_seq_lens=16384,
             vortex_cache_dir="/tmp/vcache")
    d.update(over)
    return SimpleNamespace(**d)


# --- build_engine_kwargs ---------------------------------------------------

def test_dense_kwargs_match_baseline():
    k = build_engine_kwargs(_args("dense"), n_input_tokens=9661)
    assert k["disable_cuda_graph"] is True
    assert k["disable_radix_cache"] is True
    assert k["disable_overlap_schedule"] is True   # vortex is not overlap-safe
    assert k["attention_backend"] == "flashinfer"
    assert k["page_size"] == 16                    # multiple of vortex_block_size
    assert k["kv_cache_dtype"] == "auto"
    # chunked prefill disabled: budget holds one whole request, is a multiple
    # of the 16-token page, and is < 2 requests so none ever co-pack
    assert 9661 <= k["chunked_prefill_size"] < 2 * 9661
    assert k["chunked_prefill_size"] % 16 == 0
    assert k["enable_vortex_sparsity"] is False    # dense -> sparsity off
    assert "vortex_module_name" not in k           # no vortex flow in dense
    assert "mem_fraction_static" not in k          # None -> omitted


def test_quest_kwargs_add_vortex():
    k = build_engine_kwargs(_args("quest"), n_input_tokens=9661)
    assert k["enable_vortex_sparsity"] is True
    assert k["vortex_module_name"] == "gqa_quest_sparse_attention"
    assert k["vortex_attention_backend"] == "flashinfer"
    assert k["vortex_topk_val"] == 64
    assert k["vortex_block_size"] == 16
    assert k["page_size"] == 16
    assert k["vortex_max_seq_lens"] >= 9661 + 256
    assert k["vortex_compilation_cache_dir"]       # non-empty


def test_enable_cuda_graph_flag():
    k = build_engine_kwargs(_args("dense", enable_cuda_graph=True), 9661)
    assert k["disable_cuda_graph"] is False


def test_mem_fraction_static_passed_when_set():
    k = build_engine_kwargs(_args("dense", mem_fraction_static=0.8), 9661)
    assert k["mem_fraction_static"] == 0.8


# --- compute_metrics -------------------------------------------------------

def test_compute_metrics_tpot_formula():
    # prefill ends 1.0 s in; stream ends at 3.0 s; 101 tokens emitted
    m = compute_metrics(0.0, 1.0, 3.0, 101)
    assert m["ttft_ms"] == 1000.0
    assert m["decode_time_ms"] == 2000.0
    assert m["total_time_ms"] == 3000.0
    assert m["tpot_ms"] == 20.0                    # 2000 ms / (101 - 1)
    assert m["tokens_generated"] == 101


def test_compute_metrics_no_decode():
    m = compute_metrics(0.0, None, 2.0, 0)
    assert m["tpot_ms"] == 0.0
    assert m["tokens_generated"] == 0


# --- kv_pool_capacity ------------------------------------------------------

def test_kv_pool_capacity_reads_scheduler_info():
    engine = SimpleNamespace(scheduler_info={"max_total_num_tokens": 127000})
    assert kv_pool_capacity(engine) == 127000


def test_kv_pool_capacity_missing_returns_none():
    assert kv_pool_capacity(SimpleNamespace()) is None


# --- classify_status -------------------------------------------------------

def test_classify_status_ok():
    # 8 * 9917 = 79336 <= 127000 -> the batch fits in the KV pool
    assert classify_status(256, 8, 9917, 127000) == "ok"


def test_classify_status_capped():
    # 32 * 9917 = 317344 > 127000 -> wave-serialized
    assert classify_status(256, 32, 9917, 127000) == "capped"


def test_classify_status_error_on_no_decode():
    assert classify_status(1, 8, 9917, 127000) == "error"


def test_classify_status_ok_when_capacity_unknown():
    assert classify_status(256, 64, 9917, None) == "ok"


# --- warmup_batch ----------------------------------------------------------

class _FakeEngine:
    """Records generate() calls without touching a GPU."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))


def test_warmup_batch_runs_one_short_untimed_generate():
    engine = _FakeEngine()
    prompts = ["p", "p", "p", "p"]
    warmup_batch(engine, prompts)
    assert len(engine.calls) == 1                  # exactly one warmup generate
    got_prompts, kwargs = engine.calls[0]
    assert got_prompts == prompts                  # the full batch is warmed
    assert kwargs["sampling_params"]["max_new_tokens"] == _WARMUP_TOKENS
    assert not kwargs.get("stream")                # untimed -- not streamed
