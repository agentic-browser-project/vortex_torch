# Quest `get_engine` API Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in path through `vortex_torch.engine.sgl.api.get_engine` (Quest's official engine constructor) to the existing TPOT benchmark, run dense + quest through both the direct `sgl.Engine(**kwargs)` path (current) and the `get_engine` path with matched fairness flags, and report whether the two paths produce different TPOT values.

**Architecture:** Single harness, two engine constructors. Adds a `--engine-api {direct,get_engine}` flag to `benchmark_quest_tpot.py`. A new `build_get_engine_kwargs(args, n_input)` returns the kwargs to pass to `get_engine()`, with the same baseline-matching overrides (`disable_cuda_graph=True`, `disable_radix_cache=True`, `chunked_prefill_size=...`, debug logging) as the direct path. A new `compare_engine_apis.py` joins the two aggregated CSVs and emits a side-by-side TPOT table. A new `run_engine_api_comparison.sh` orchestrates the four sweeps (`{direct,get_engine} × {dense,quest}`), the two aggregations, and the final comparison. The existing three-way driver (`run_benchmark.sh`) is untouched.

**Tech Stack:** Python 3.12 + `sglang 0.5.9` + `vortex_torch v0.5` (in-process `sgl.Engine` / `get_engine`); pytest for unit tests; bash for the driver; CSV/markdown for output.

---

## Background — what `get_engine` does and why we want to test it

`vortex_torch/engine/sgl/api.py:25` defines `get_engine(**named, **kwargs)`. It sets vortex-specific defaults (`vortex_block_size=16`, `vortex_layers_skip=[0]`, `vortex_dtype="bfloat16"`, `vortex_compilation_cache_dir`, etc.) plus three engine defaults (`disable_overlap_schedule=True`, `attention_backend="flashinfer"`, `tp_size=1`), then does `engine_kwargs.update(kwargs); return sgl.Engine(**engine_kwargs)`. So:

- Anything in `**kwargs` overrides the hardcoded defaults.
- `disable_cuda_graph=False` is hardcoded — must be overridden to `True` to match the baseline.
- `enable_vortex_sparsity=True` is hardcoded — must be overridden to `False` for the dense arm.
- `disable_overlap_schedule=True` already matches what we want.

The current benchmark calls `sgl.Engine(**build_engine_kwargs(...))` directly (line 283 of `benchmark_quest_tpot.py`) and never invokes `get_engine`. The question this plan answers: **with matched fairness flags, does routing through `get_engine` change measured TPOT?** If yes, the current benchmark's bypass-the-helper choice has a measurable effect and we should reconsider; if no, the current numbers are robust.

---

## File Structure

All paths relative to the worktree root `/vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5/`.

| File | Status | Responsibility |
|---|---|---|
| `quest_batch_benchmark/benchmark_quest_tpot.py` | Modify | Add `--engine-api` flag, `build_get_engine_kwargs`, `make_engine` dispatcher. |
| `quest_batch_benchmark/compare_engine_apis.py` | Create | Read two aggregated CSVs, emit side-by-side TPOT table + ratio. |
| `quest_batch_benchmark/run_engine_api_comparison.sh` | Create | Driver: 4 sweeps, 2 aggregations, 1 comparison. |
| `quest_batch_benchmark/tests/test_harness.py` | Modify | Unit tests for `build_get_engine_kwargs` + dispatcher + equivalence vs `build_engine_kwargs`. |
| `quest_batch_benchmark/tests/test_compare_engine_apis.py` | Create | Unit tests for the comparison merge. |
| `quest_batch_benchmark/results/engine_api_comparison.{csv,md}` | Generated | Output of the comparison run. |
| `quest_batch_benchmark/results/tpot_vs_batchsize_{direct,get_engine}.csv` | Generated | Per-API aggregated outputs. |

---

## Task 1: Add `--engine-api` flag and `build_get_engine_kwargs`

**Files:**
- Modify: `quest_batch_benchmark/benchmark_quest_tpot.py`
- Test: `quest_batch_benchmark/tests/test_harness.py`

- [ ] **Step 1: Add failing tests for `build_get_engine_kwargs`**

Append to `quest_batch_benchmark/tests/test_harness.py`:

```python
# --- build_get_engine_kwargs (Quest's official get_engine API) -------------

from benchmark_quest_tpot import build_get_engine_kwargs, QUEST_MODULE


def test_get_engine_kwargs_dense_overrides_sparsity():
    """get_engine hardcodes enable_vortex_sparsity=True; dense must override."""
    k = build_get_engine_kwargs(_args("dense"), n_input_tokens=9661)
    assert k["enable_vortex_sparsity"] is False
    assert k["disable_cuda_graph"] is True       # baseline match
    assert k["disable_radix_cache"] is True
    assert k["attention_backend"] == "flashinfer"
    assert 9661 <= k["chunked_prefill_size"] < 2 * 9661
    assert k["chunked_prefill_size"] % 16 == 0


def test_get_engine_kwargs_quest_sets_vortex():
    k = build_get_engine_kwargs(_args("quest"), n_input_tokens=9661)
    assert k["enable_vortex_sparsity"] is True
    assert k["vortex_module_name"] == QUEST_MODULE
    assert k["vortex_topk_val"] == 64
    assert k["vortex_block_size"] == 16
    assert k["vortex_max_seq_lens"] >= 9661 + 256


def test_get_engine_kwargs_fairness_flags_match_direct_path():
    """Every fairness-relevant flag in build_engine_kwargs must also appear
    (with the same value) in build_get_engine_kwargs. This is the contract
    that makes the two paths comparable."""
    args = _args("quest")
    direct = build_engine_kwargs(args, n_input_tokens=9661)
    gengine = build_get_engine_kwargs(args, n_input_tokens=9661)
    for key in ("disable_cuda_graph", "disable_radix_cache",
                "disable_overlap_schedule", "attention_backend",
                "chunked_prefill_size", "page_size", "kv_cache_dtype",
                "decode_log_interval", "show_time_cost", "log_level"):
        assert gengine[key] == direct[key], (
            f"fairness flag {key!r} differs: direct={direct[key]!r} "
            f"get_engine={gengine[key]!r}"
        )


def test_get_engine_kwargs_enable_cuda_graph_flag():
    k = build_get_engine_kwargs(_args("quest", enable_cuda_graph=True), 9661)
    assert k["disable_cuda_graph"] is False
```

- [ ] **Step 2: Run tests, verify they fail with ImportError / AttributeError**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
quest_batch_benchmark/.venv/bin/python -m pytest \
  quest_batch_benchmark/tests/test_harness.py -k get_engine -v
```
Expected: `ImportError: cannot import name 'build_get_engine_kwargs'` or equivalent.

- [ ] **Step 3: Add `build_get_engine_kwargs` to `benchmark_quest_tpot.py`**

Add after the existing `build_engine_kwargs` function (right after the closing `return kwargs` of `build_engine_kwargs`, before `compute_metrics`):

```python
def build_get_engine_kwargs(args, n_input_tokens: int) -> dict:
    """Kwargs for `vortex_torch.engine.sgl.get_engine` — Quest's official
    in-process engine constructor.

    `get_engine` hardcodes a few defaults that the baseline `tpot-no-share`
    config flips (notably `disable_cuda_graph=False`, `enable_vortex_sparsity=
    True`). It also accepts `**kwargs` and applies them *after* its own
    defaults, so we override every fairness-relevant flag explicitly here.
    The only meaningful difference between the engine produced by this path
    and the engine produced by `sgl.Engine(**build_engine_kwargs(...))` is
    the constructor call itself — same model, same backend, same sparsity
    flow, same CUDA-graph / radix-cache / chunked-prefill / overlap-schedule
    posture, same debug logging.
    """
    max_seq = max(args.max_seq_lens, n_input_tokens + args.max_tokens + 64)
    chunked_prefill_size = ((n_input_tokens + 15) // 16) * 16 + 16

    kwargs = dict(
        # get_engine's named params we want to pin
        model_path=args.model_path,
        vortex_max_seq_lens=max_seq,
        vortex_block_size=16,
        vortex_topk_val=args.topk_val,
        vortex_block_reserved_bos=1,
        vortex_block_reserved_eos=2,
        vortex_workload_chunk_size=32,
        vortex_layers_skip=[0],
        vortex_module_name=QUEST_MODULE,
        # gqa_quest_sparse_attention is built-in to vortex_torch v0.5, so the
        # flow is already in the registry by the time get_engine runs;
        # vortex_module_path is only consulted if the name is unregistered.
        # Pointing at this benchmark file (which never @register's anything)
        # is harmless and avoids hardcoding submissions/-relative paths.
        vortex_module_path=str(Path(__file__).resolve()),
        kv_cache_dtype="auto",
        # Fairness overrides — passed via **kwargs tail in get_engine; they
        # land in the final sgl.Engine kwargs dict via engine_kwargs.update.
        disable_cuda_graph=not args.enable_cuda_graph,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        attention_backend="flashinfer",
        page_size=16,
        chunked_prefill_size=chunked_prefill_size,
        decode_log_interval=1,
        show_time_cost=True,
        log_level="debug",
        trust_remote_code=True,
        vortex_attention_backend="flashinfer",
        vortex_compilation_cache_dir=args.vortex_cache_dir,
    )
    if args.mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = args.mem_fraction_static

    if args.attention == "dense":
        # get_engine hardcodes enable_vortex_sparsity=True; flip it off for
        # the dense baseline. The vortex_* kwargs remain in the call but are
        # not consulted by sglang when sparsity is off (verified by the
        # smoke test in Task 5).
        kwargs["enable_vortex_sparsity"] = False
    return kwargs
```

- [ ] **Step 4: Run the new tests, verify they pass**

```bash
quest_batch_benchmark/.venv/bin/python -m pytest \
  quest_batch_benchmark/tests/test_harness.py -k get_engine -v
```
Expected: 4 tests pass.

- [ ] **Step 5: Run the full test file to verify no regressions**

```bash
quest_batch_benchmark/.venv/bin/python -m pytest \
  quest_batch_benchmark/tests/test_harness.py -v
```
Expected: all previous tests still pass.

- [ ] **Step 6: Add `--engine-api` CLI flag and `make_engine` dispatcher**

In `benchmark_quest_tpot.py`, add `make_engine` immediately above `def run(args)`:

```python
def make_engine(args, n_input_tokens: int):
    """Construct an sgl.Engine via the chosen engine API.

    `direct`     — current path: build kwargs in this file, call sgl.Engine.
    `get_engine` — Quest's official wrapper at vortex_torch.engine.sgl.api;
                   it folds in vortex defaults and ends in sgl.Engine(**...).
    The two paths are configured to produce equivalent fairness-relevant
    kwargs (see build_get_engine_kwargs); any TPOT delta between them is
    attributable to the constructor call itself.
    """
    if args.engine_api == "direct":
        return sgl.Engine(**build_engine_kwargs(args, n_input_tokens))
    if args.engine_api == "get_engine":
        from vortex_torch.engine.sgl.api import get_engine
        return get_engine(**build_get_engine_kwargs(args, n_input_tokens))
    raise ValueError(f"unknown engine_api: {args.engine_api!r}")
```

Replace line 283 (`engine = sgl.Engine(**build_engine_kwargs(args, n_input))`) with:

```python
    engine = make_engine(args, n_input)
```

In `build_parser()`, immediately after the `--attention` argument, add:

```python
    p.add_argument("--engine-api", choices=["direct", "get_engine"],
                   default="direct",
                   help="Engine constructor path. 'direct' (current default) "
                        "calls sgl.Engine(**build_engine_kwargs); 'get_engine' "
                        "routes through vortex_torch.engine.sgl.get_engine "
                        "with identical fairness flags. Used by the "
                        "engine-API comparison sweep.")
```

Add an entry for the new column to the raw CSV by extending `RAW_CSV_FIELDS` — but only if needed for downstream joins. (Decision: do not extend the schema; instead use a separate `--raw-csv` path per engine_api in the driver, so the existing aggregator is unaffected.)

- [ ] **Step 7: Add a unit test for the dispatcher**

Append to `quest_batch_benchmark/tests/test_harness.py`:

```python
from unittest.mock import patch
from benchmark_quest_tpot import make_engine


def test_make_engine_direct_calls_sgl_engine():
    with patch("benchmark_quest_tpot.sgl.Engine") as mock_eng:
        make_engine(_args("quest", engine_api="direct"), n_input_tokens=9661)
    assert mock_eng.called
    # Direct path uses our build_engine_kwargs — no vortex_module_path key.
    call_kwargs = mock_eng.call_args.kwargs
    assert call_kwargs["enable_vortex_sparsity"] is True
    assert "vortex_module_path" not in call_kwargs


def test_make_engine_get_engine_routes_through_helper():
    with patch("vortex_torch.engine.sgl.api.sgl.Engine") as mock_eng:
        make_engine(_args("quest", engine_api="get_engine"), n_input_tokens=9661)
    assert mock_eng.called
    # get_engine path always passes vortex_module_path.
    call_kwargs = mock_eng.call_args.kwargs
    assert "vortex_module_path" in call_kwargs
    assert call_kwargs["disable_cuda_graph"] is True  # baseline-matched
    assert call_kwargs["disable_radix_cache"] is True


def test_make_engine_unknown_api_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown engine_api"):
        make_engine(_args("quest", engine_api="bogus"), n_input_tokens=9661)
```

Update the `_args` helper at the top of `test_harness.py` to include `engine_api="direct"` in the default dict:

```python
def _args(attention, **over):
    d = dict(attention=attention, model_path="/models/Qwen3-8B", max_tokens=256,
             repeat=3, topk_val=64, enable_cuda_graph=False,
             mem_fraction_static=None, max_seq_lens=16384,
             vortex_cache_dir="/tmp/vcache", engine_api="direct")
    d.update(over)
    return SimpleNamespace(**d)
```

- [ ] **Step 8: Run the dispatcher tests, verify pass**

```bash
quest_batch_benchmark/.venv/bin/python -m pytest \
  quest_batch_benchmark/tests/test_harness.py -v
```
Expected: all tests pass, including 4 `get_engine_kwargs` tests and 3 dispatcher tests.

- [ ] **Step 9: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add quest_batch_benchmark/benchmark_quest_tpot.py quest_batch_benchmark/tests/test_harness.py
git commit -m "feat(bench): add --engine-api flag and get_engine kwarg builder

Adds an opt-in path that routes engine construction through Quest's
official vortex_torch.engine.sgl.get_engine helper, with the same
baseline-matching fairness flags as the direct sgl.Engine path. The
helper hardcodes disable_cuda_graph=False and enable_vortex_sparsity=
True; both are overridden via the **kwargs tail to keep the comparison
honest. Default is 'direct' so existing runs are unaffected."
```

---

## Task 2: Add the comparison script `compare_engine_apis.py`

**Files:**
- Create: `quest_batch_benchmark/compare_engine_apis.py`
- Test: `quest_batch_benchmark/tests/test_compare_engine_apis.py`

- [ ] **Step 1: Write failing tests**

Create `quest_batch_benchmark/tests/test_compare_engine_apis.py`:

```python
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
    assert abs(float(by_bs[1]["ratio_get_engine_over_direct"]) - (11.2 / 11.0)) < 1e-9


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
```

- [ ] **Step 2: Run tests, verify they fail with ImportError**

```bash
quest_batch_benchmark/.venv/bin/python -m pytest \
  quest_batch_benchmark/tests/test_compare_engine_apis.py -v
```
Expected: `ImportError: No module named 'compare_engine_apis'`.

- [ ] **Step 3: Implement `compare_engine_apis.py`**

Create `quest_batch_benchmark/compare_engine_apis.py`:

```python
#!/usr/bin/env python3
"""Compare TPOT between the two engine-construction paths.

Reads two aggregated TPOT-vs-batch-size CSVs (produced by aggregate_results.py)
— one from `--engine-api direct`, one from `--engine-api get_engine` — joins
on (attention, batch_size), and writes a side-by-side table with absolute and
relative deltas. Outputs both CSV (machine-readable) and markdown (the table
you paste into a report).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


_OUT_FIELDS = [
    "attention", "batch_size",
    "tpot_ms_direct", "tpot_ms_get_engine",
    "abs_diff_ms", "ratio_get_engine_over_direct",
]


def _load(path: str) -> Dict[tuple, Dict[str, str]]:
    """Return {(attention, batch_size_int): row} from an aggregated CSV."""
    out: Dict[tuple, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["attention"], int(r["batch_size"]))] = r
    return out


def build_comparison_rows(direct_csv: str, get_engine_csv: str) -> List[Dict[str, str]]:
    """Join two aggregated CSVs on (attention, batch_size); drop unmatched rows."""
    direct = _load(direct_csv)
    gengine = _load(get_engine_csv)
    rows: List[Dict[str, str]] = []
    for key in sorted(direct.keys() & gengine.keys(),
                      key=lambda kv: (kv[0], kv[1])):
        attention, batch_size = key
        td = float(direct[key]["tpot_ms_mean"])
        tg = float(gengine[key]["tpot_ms_mean"])
        rows.append({
            "attention": attention,
            "batch_size": str(batch_size),
            "tpot_ms_direct": f"{td:.3f}",
            "tpot_ms_get_engine": f"{tg:.3f}",
            "abs_diff_ms": f"{tg - td:.6f}",
            "ratio_get_engine_over_direct": (
                f"{tg / td:.6f}" if td > 0 else "nan"
            ),
        })
    return rows


def render_markdown(rows: List[Dict[str, str]]) -> str:
    header = ("| attention | batch size | direct TPOT (ms) | get_engine TPOT (ms) | "
              "abs diff (ms) | get_engine / direct |")
    sep = "|---|---:|---:|---:|---:|---:|"
    body = []
    for r in rows:
        body.append(
            f"| {r['attention']} | {r['batch_size']} | "
            f"{r['tpot_ms_direct']} | {r['tpot_ms_get_engine']} | "
            f"{float(r['abs_diff_ms']):+.3f} | "
            f"{r['ratio_get_engine_over_direct']} |"
        )
    return "\n".join([header, sep, *body])


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct-csv",
                   default=str(here / "results" / "tpot_vs_batchsize_direct.csv"))
    p.add_argument("--get-engine-csv",
                   default=str(here / "results" / "tpot_vs_batchsize_get_engine.csv"))
    p.add_argument("--out-csv",
                   default=str(here / "results" / "engine_api_comparison.csv"))
    p.add_argument("--out-md",
                   default=str(here / "results" / "engine_api_comparison.md"))
    args = p.parse_args()

    rows = build_comparison_rows(args.direct_csv, args.get_engine_csv)
    if not rows:
        raise SystemExit(
            f"no joined rows -- check that {args.direct_csv} and "
            f"{args.get_engine_csv} share (attention, batch_size) keys"
        )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    md = render_markdown(rows)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    print(f"[compare] wrote {args.out_csv}")
    print(f"[compare] wrote {args.out_md}")
    print(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the comparison-script tests, verify pass**

```bash
quest_batch_benchmark/.venv/bin/python -m pytest \
  quest_batch_benchmark/tests/test_compare_engine_apis.py -v
```
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add quest_batch_benchmark/compare_engine_apis.py \
        quest_batch_benchmark/tests/test_compare_engine_apis.py
git commit -m "feat(bench): add compare_engine_apis.py for side-by-side TPOT diff

Joins two aggregated TPOT-vs-batch-size CSVs (one per engine_api) on
(attention, batch_size) and emits abs_diff_ms and the get_engine /
direct ratio. Drops rows that don't appear in both CSVs rather than
silently filling zeros."
```

---

## Task 3: Add the driver `run_engine_api_comparison.sh`

**Files:**
- Create: `quest_batch_benchmark/run_engine_api_comparison.sh`

- [ ] **Step 1: Write the driver**

Create `quest_batch_benchmark/run_engine_api_comparison.sh`:

```bash
#!/usr/bin/env bash
# Drive the two-engine-API comparison sweep: dense + quest, each via the
# 'direct' sgl.Engine(...) path AND Quest's official
# vortex_torch.engine.sgl.get_engine wrapper. Aggregate each pair, then
# diff. Fairness flags are matched in build_get_engine_kwargs; any TPOT
# delta in the final table is attributable to the constructor call itself.
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH"

GPU="${GPU:-0}"
PY="$BENCH/.venv/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p results logs

for api in direct get_engine; do
    RAW="$BENCH/results/raw_results_${api}.csv"
    OUT="$BENCH/results/tpot_vs_batchsize_${api}.csv"
    rm -f "$RAW"
    for mode in dense quest; do
        echo ">>> running $mode via engine_api=$api  (GPU $GPU)"
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
            --attention "$mode" \
            --engine-api "$api" \
            --raw-csv "$RAW" \
            2>&1 | tee "logs/${mode}_${api}_${TS}.log"
        status=${PIPESTATUS[0]}
        if [ "$status" -ne 0 ]; then
            echo "!!! $mode (api=$api) failed (exit $status) -- see logs/${mode}_${api}_${TS}.log" >&2
            exit "$status"
        fi
    done
    echo ">>> aggregating $api"
    "$PY" aggregate_results.py --raw-csv "$RAW" --out-csv "$OUT" || exit 1
done

echo ">>> building engine-API comparison"
"$PY" compare_engine_apis.py \
    --direct-csv "$BENCH/results/tpot_vs_batchsize_direct.csv" \
    --get-engine-csv "$BENCH/results/tpot_vs_batchsize_get_engine.csv" \
    --out-csv "$BENCH/results/engine_api_comparison.csv" \
    --out-md "$BENCH/results/engine_api_comparison.md" \
    2>&1 | tee "logs/engine_api_comparison_${TS}.log"

echo ">>> done"
echo "    direct raw       : $BENCH/results/raw_results_direct.csv"
echo "    direct agg       : $BENCH/results/tpot_vs_batchsize_direct.csv"
echo "    get_engine raw   : $BENCH/results/raw_results_get_engine.csv"
echo "    get_engine agg   : $BENCH/results/tpot_vs_batchsize_get_engine.csv"
echo "    comparison CSV   : $BENCH/results/engine_api_comparison.csv"
echo "    comparison table : $BENCH/results/engine_api_comparison.md"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x quest_batch_benchmark/run_engine_api_comparison.sh
```

- [ ] **Step 3: Lint with `bash -n` (syntax check; no execution)**

```bash
bash -n quest_batch_benchmark/run_engine_api_comparison.sh
echo "exit=$?"
```
Expected: `exit=0` (no syntax errors).

- [ ] **Step 4: Commit**

```bash
git add quest_batch_benchmark/run_engine_api_comparison.sh
git commit -m "feat(bench): add run_engine_api_comparison.sh driver

Runs the dense+quest sweep twice (engine_api=direct then engine_api=
get_engine), aggregates each pair separately, and invokes
compare_engine_apis.py to emit the side-by-side TPOT diff. Doesn't
touch run_benchmark.sh (the three-way driver)."
```

---

## Task 4: Pre-flight smoke test on GPU (single batch, both APIs)

This is a manual GPU step. Goal: confirm both engine constructors actually boot and produce coherent decode output before committing to the full ~30–60 min sweep. The unit tests can't catch boot-time failures (e.g., `get_engine` rejecting `vortex_module_path=<this file>`).

**Files:** none modified; this is a runtime check.

- [ ] **Step 1: Smoke-test the `direct` path (already known to work — regression check)**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
mkdir -p quest_batch_benchmark/results/smoke
CUDA_VISIBLE_DEVICES=0 quest_batch_benchmark/.venv/bin/python \
    quest_batch_benchmark/benchmark_quest_tpot.py \
    --attention quest \
    --engine-api direct \
    --batch-sizes 1 \
    --repeat 1 \
    --raw-csv quest_batch_benchmark/results/smoke/direct_quest_smoke.csv \
    2>&1 | tail -40
```
Expected: a `[done]` line; the smoke CSV has a single row with `status=ok` and a finite `tpot_ms`.

- [ ] **Step 2: Smoke-test the `get_engine` path for quest**

```bash
CUDA_VISIBLE_DEVICES=0 quest_batch_benchmark/.venv/bin/python \
    quest_batch_benchmark/benchmark_quest_tpot.py \
    --attention quest \
    --engine-api get_engine \
    --batch-sizes 1 \
    --repeat 1 \
    --raw-csv quest_batch_benchmark/results/smoke/get_engine_quest_smoke.csv \
    2>&1 | tail -40
```
Expected: same — `[done]` line, single row, `status=ok`, finite `tpot_ms`.

If this fails with a `vortex_module_path` error (e.g. sglang/vortex's loader rejecting `__file__` as a non-submission), the fix is to point `vortex_module_path` at a real submission file or a dedicated stub. Replace the relevant line in `build_get_engine_kwargs` with:

```python
        # Pointing at the vortex_torch flow file is safe: gqa_quest_sparse_
        # attention is already in the registry by import time, so the loader
        # short-circuits on the name and never reads the file.
        vortex_module_path=str(Path(
            Path(__file__).resolve().parent.parent / "vortex_torch" / "flow" /
            "algorithms.py"
        )),
```

Re-run the smoke test.

- [ ] **Step 3: Smoke-test the `get_engine` path for dense**

```bash
CUDA_VISIBLE_DEVICES=0 quest_batch_benchmark/.venv/bin/python \
    quest_batch_benchmark/benchmark_quest_tpot.py \
    --attention dense \
    --engine-api get_engine \
    --batch-sizes 1 \
    --repeat 1 \
    --raw-csv quest_batch_benchmark/results/smoke/get_engine_dense_smoke.csv \
    2>&1 | tail -40
```
Expected: same — `[done]` line, single row, `status=ok`, finite `tpot_ms`. If this fails because sglang still tries to load a vortex flow when `enable_vortex_sparsity=False`, drop the vortex_* kwargs from the dense branch in `build_get_engine_kwargs` (only keep `enable_vortex_sparsity=False`, `model_path`, and the baseline-matching engine flags). Re-run.

- [ ] **Step 4: Sanity-check the smoke TPOTs are in the same order of magnitude**

For quest bs=1 the existing benchmark reports ~11 ms; for dense bs=1 ~9 ms. Both smoke runs should land in that range (±2 ms, since this is a single repeat without the warmup-then-3-rep statistical smoothing). If one path is wildly off (>2x), stop — there is a configuration bug that the full sweep will inherit.

---

## Task 5: Run the full comparison sweep

This is the experiment itself. Wall-clock estimate: ~30–60 min (4 sweeps × ~10–15 min each, dominated by quest at bs=64).

**Files:** none modified; this is a runtime step.

- [ ] **Step 1: Confirm GPU 0 is free**

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```
Expected: GPU 0 with `utilization.gpu` near 0% and `memory.used` < 1024 MiB. If not, pick another GPU and set `GPU=N` for the next step.

- [ ] **Step 2: Run the driver**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
GPU=0 bash quest_batch_benchmark/run_engine_api_comparison.sh
```
Expected on success:
- `results/raw_results_{direct,get_engine}.csv` each have `2 modes × 7 batches × 3 repeats = 42` data rows (plus header).
- `results/tpot_vs_batchsize_{direct,get_engine}.csv` each have `2 modes × 7 batches = 14` rows.
- `results/engine_api_comparison.csv` and `results/engine_api_comparison.md` exist.
- The driver tails the markdown table to stdout.

- [ ] **Step 3: Verify every row is `status=ok`**

```bash
for f in quest_batch_benchmark/results/tpot_vs_batchsize_direct.csv \
         quest_batch_benchmark/results/tpot_vs_batchsize_get_engine.csv; do
  echo "=== $f ==="
  awk -F, 'NR>1 {print $1, $2, $8}' "$f"
done
```
Expected: every row's `status` column is `ok`. Any `capped` or `error` rows mean the comparison is on shaky ground for that batch size — note it but don't necessarily reject.

- [ ] **Step 4: Inspect the comparison table**

```bash
cat quest_batch_benchmark/results/engine_api_comparison.md
```
Interpret the `abs diff (ms)` and `get_engine / direct` columns:

- **Difference dominated by noise** (e.g., `|abs diff| < 0.5 ms` for small batches, `< 2 ms` for bs=64; ratios within ~2% of 1.000): the two constructors produce indistinguishable measurements. The current bypass-the-helper choice is benign; the README can keep its current claim that the harness mirrors the baseline.
- **Systematic offset** (e.g., one API consistently 5%+ faster across all batch sizes): the constructors materially differ — probably a hardcoded default in `get_engine` we missed, or sglang behaving differently when vortex_module_path is present vs absent. Compare the final kwargs each path passes to `sgl.Engine` (see Task 1 unit tests for the technique) and diagnose.

- [ ] **Step 5: Commit the results**

```bash
git add quest_batch_benchmark/results/raw_results_direct.csv \
        quest_batch_benchmark/results/raw_results_get_engine.csv \
        quest_batch_benchmark/results/tpot_vs_batchsize_direct.csv \
        quest_batch_benchmark/results/tpot_vs_batchsize_get_engine.csv \
        quest_batch_benchmark/results/engine_api_comparison.csv \
        quest_batch_benchmark/results/engine_api_comparison.md
git commit -m "data(bench): engine-API comparison results (direct vs get_engine)

Full 7-batch-size sweep, dense + quest, each measured via both engine
constructors with matched fairness flags. See
results/engine_api_comparison.md for the side-by-side table."
```

---

## Task 6: Document the experiment in the README

**Files:**
- Modify: `quest_batch_benchmark/README.md`

- [ ] **Step 1: Add an "Engine-API comparison" section**

After the existing "Reproduce" section in `quest_batch_benchmark/README.md`, add:

```markdown
## Engine-API comparison (sanity check)

The benchmark calls `sgl.Engine(**build_engine_kwargs(...))` directly rather
than going through Quest's official wrapper
`vortex_torch.engine.sgl.get_engine`. The wrapper accepts `**kwargs` and
applies them after its own defaults, so every fairness-relevant flag the
baseline sets (`disable_cuda_graph=True`, `disable_radix_cache=True`,
`chunked_prefill_size=...`, debug logging) flows through unchanged; the
direct path was chosen because it puts every kwarg in one local file rather
than depending on the wrapper's hardcoded defaults.

To verify the choice doesn't perturb the numbers, the harness exposes
`--engine-api {direct,get_engine}` and the driver
`run_engine_api_comparison.sh` runs the full dense+quest sweep through both
paths with matched fairness flags, then writes a side-by-side table at
`results/engine_api_comparison.md`. The two paths produce equivalent TPOT
values (see that file) — confirming the current direct-construction choice
is purely a code-locality decision, not a measurement-fairness one.

```bash
GPU=0 bash quest_batch_benchmark/run_engine_api_comparison.sh
```
```

After running Task 5, replace the placeholder "(see that file)" with a brief one-sentence summary of the actual finding (e.g., "the two paths agree within 1% across all 14 configurations").

- [ ] **Step 2: Commit**

```bash
git add quest_batch_benchmark/README.md
git commit -m "docs(bench): describe the engine-API comparison sanity check

Documents the new --engine-api flag, the comparison driver, and the
finding that the two engine constructors produce equivalent TPOT
values when fairness flags are matched."
```

---

## Self-review summary

Spec coverage:
- Add `get_engine` path to the benchmark — Task 1.
- Run dense + quest both ways with matched fairness flags — Task 5.
- Check whether values differ — Tasks 5 (run) + 6 (write up finding).

Placeholder check: every step has concrete code or commands. The only `<placeholder>` is the README finding sentence in Task 6 Step 1, which is *intentionally* filled in after the run produces data.

Type consistency: `build_get_engine_kwargs` and `make_engine` are referenced consistently across Task 1's tests, the dispatcher, and Tasks 2/3 driver. `engine_api` field added to `_args` defaults in Task 1 Step 7 so existing tests don't break.
