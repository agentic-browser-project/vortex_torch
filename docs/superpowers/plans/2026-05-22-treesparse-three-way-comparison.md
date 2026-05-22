# TreeSparseAttention Three-Way Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add **TreeSparseAttention** as a third attention method alongside `dense` and `quest` in the v0.5 decode-TPOT batch benchmark, producing one table of TPOT for all three methods across batch sizes 1–64.

**Architecture:** TreeSparseAttention is **not** a vortex/sglang plugin — it is a standalone project (`/vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention`) with its own Python 3.13 venv, its own pre-built CUDA kernels, and its own HuggingFace+FlashInfer harness (`benchmark_batch.py`, driven by `run_batch_experiments.sh`). It therefore **cannot** be added as a fourth `--attention` mode inside `benchmark_quest_tpot.py` (that harness only knows `sgl.Engine`). Instead, TreeSparse runs as a **separate process in its own environment**, on the **same `request.json`** the quest/dense runs use, and its result JSON is merged with the quest/dense aggregate into a single three-way table. The proven `dense`/`quest` pipeline is left untouched; TreeSparse is a parallel branch joined at the end.

**Fairness contract** (the user's explicit requirement — *same input, same output, same TPOT setting*):
- **Same input:** all three methods run on `quest_batch_benchmark/request.json` (the established 9,661-token text-only prompt). `run_treesparse.sh` overrides TreeSparse's hardcoded WebVoyager request with this file.
- **Same output:** 256 decode tokens (`--num-decode-tokens 256`, matches quest's `--max-tokens 256`).
- **Same sweep:** batch sizes `1,2,4,8,16,32,64`, `repeat=3` — identical on both harnesses.
- **TPOT — fair representative value:** quest reports a *warm* mean-of-3 (it runs an untimed warmup per batch). TreeSparse's `benchmark_batch.py` runs **no** separate warmup, so its first repetition absorbs one-time FlashInfer JIT / cold-cache cost and inflates `tpot_mean_ms`. The merge therefore uses TreeSparse's **`tpot_median_ms`** (median of 3 — discards the single cold rep), which is the apples-to-apples match for quest's warm mean. This is documented in code and in the README.
- **Unavoidable, documented differences** (different methods *require* different engines — cannot be unified): quest/dense run through `sgl.Engine` (torch 2.9.1, streaming wall-clock TPOT); TreeSparse runs through its own harness (torch 2.11.0, per-decode-step `cuda.synchronize()` timing). Both measure mean decode-step latency excluding the first token. Sparsity operating points also differ by design: quest `topk_val=64` (1024 tokens kept); TreeSparse `top-k=128` chunks (the `run_batch_experiments.sh tpot-no-share` default). These are each method's intended setting — recorded transparently in the README, not forced equal.

**Tech Stack:** Python 3.12 (quest venv: torch 2.9.1+cu128, sglang v0.5.9) and Python 3.13 (TreeSparse venv: torch 2.11.0+cu128, flashinfer 0.6.9, transformers 5.7.0). Bash orchestration. `pytest` for unit tests. NVIDIA B200 (sm_100). Lmod modules.

**Git:** All work is committed onto the **existing branch `quest-batch-benchmark-v0.5`** in the **existing worktree** `/vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5` (per the user's choice). The TreeSparseAttention repo is **never modified** — all new code lives under `quest_batch_benchmark/`.

**Paths used throughout this plan:**
- Worktree root: `/vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5`
- Benchmark dir (`$BENCH`): `<root>/quest_batch_benchmark`
- TreeSparseAttention (`$TSA`): `/vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention`
- Quest venv python: `$BENCH/.venv/bin/python`
- TreeSparse venv: `$TSA/.venv` (Python 3.13)

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `quest_batch_benchmark/treesparse_results.py` | **Create** | Pure-Python: convert a TreeSparse results JSON into rows matching `tpot_vs_batchsize.csv`'s schema (`attention=treesparse`). Importable + CLI. |
| `quest_batch_benchmark/build_comparison.py` | **Create** | Pure-Python: merge dense+quest (`tpot_vs_batchsize.csv`) with TreeSparse (`treesparse_raw.json`) → `tpot_three_way.csv` + `comparison_table.md`; print the table. |
| `quest_batch_benchmark/run_treesparse.sh` | **Create** | Orchestrator: activate the TreeSparse env, run `run_batch_experiments.sh tpot-no-share` on the shared `request.json`, copy the result JSON to `results/treesparse_raw.json`. |
| `quest_batch_benchmark/run_benchmark.sh` | **Modify** | Append a `treesparse` stage + comparison build after the existing `dense`/`quest`/`aggregate` stages. |
| `quest_batch_benchmark/treesparse_env_notes.md` | **Create** | Lab note recording the verified TreeSparse environment (Task 1). |
| `quest_batch_benchmark/tests/test_treesparse_results.py` | **Create** | Unit tests for `treesparse_results.py`. |
| `quest_batch_benchmark/tests/test_build_comparison.py` | **Create** | Unit tests for `build_comparison.py`. |
| `quest_batch_benchmark/README.md` | **Modify** | Document the third method, the three-way table, the fairness contract. |
| `quest_batch_benchmark/PROGRESS.md` | **Modify** | Update status, commit list, methodology notes. |
| `quest_batch_benchmark/results/treesparse_raw.json` | **Generated** (Task 6) | TreeSparse benchmark output, copied in by `run_treesparse.sh`. |
| `quest_batch_benchmark/results/tpot_three_way.csv` | **Generated** (Task 6) | The merged three-method CSV (21 rows). |
| `quest_batch_benchmark/results/comparison_table.md` | **Generated** (Task 6) | The human-readable three-way TPOT table. |

The merge deliberately reuses `aggregate_results.OUT_FIELDS` as the CSV schema so the three-way CSV is a strict superset of the existing `tpot_vs_batchsize.csv`.

---

## Task 1: Verify the TreeSparseAttention environment

The TreeSparse venv and CUDA kernels already exist (built 2026-04-30; `tpot_no_share` runs as recent as 2026-05-13). This task **confirms** the environment runs end-to-end on our shared `request.json` before any integration code depends on it. No new environment is built unless verification fails.

**Files:**
- Create: `quest_batch_benchmark/treesparse_env_notes.md`

- [ ] **Step 1: Activate the TreeSparse environment and check imports**

Run (in the harness Bash tool, where Lmod's `module` function is already available):

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention
source env.sh
python -c "import sys; print('python', sys.version.split()[0], sys.executable)"
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import flashinfer, transformers; print('flashinfer', flashinfer.__version__, 'transformers', transformers.__version__)"
python -c "import sys; sys.path.insert(0,'python'); import _tree_sparse_kernels; print('tree_sparse_kernels OK')"
```

Expected: Python `3.13.x` from `.../TreeSparseAttention/.venv/bin/python`; `torch 2.11.0+cu128 cuda_avail True NVIDIA B200`; `flashinfer 0.6.9 transformers 5.7.0`; `tree_sparse_kernels OK`.

- [ ] **Step 2: Smoke-test `benchmark_batch.py` on the SHARED request.json**

This confirms TreeSparse's harness loads the quest benchmark's `request.json` (a `{"model","messages","parameters"}` JSON) and produces a valid result. Run (still in the activated TreeSparse env, from `$TSA`):

```bash
python benchmark_batch.py \
  --request-file /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5/quest_batch_benchmark/request.json \
  --batch-sizes 1 --num-decode-tokens 8 --repeat 1 --page-size 64 --no-kv-sharing \
  --output /tmp/tsa_smoke/smoke.json
python -c "import json; d=json.load(open('/tmp/tsa_smoke/smoke.json')); e=d['1']; print('prefill_len', e['prefill_len'], 'tpot_median_ms', e['tpot_median_ms'], 'num_decode_tokens', e['num_decode_tokens'])"
```

Expected: the run completes (exit 0), prints `Prefill length: ~9600 tokens (text-only)`, and the final line prints a `prefill_len` near 9,600 and a positive `tpot_median_ms`.

**Fallback if Step 1 or 2 fails** (record what failed in the notes file, then remediate):
- Missing/broken Python package → `pip install --upgrade flashinfer-python transformers` inside the activated venv.
- `_tree_sparse_kernels` import error (e.g. ABI mismatch) → rebuild the kernels: `cd $TSA && rm -f build/CMakeCache.txt && bash setup.sh` (needs `cmake`/`gcc`/`cuda` modules, which `env.sh` loads).
- `request.json` fails to load (`messages` content-block shape) → inspect `request.json`'s `messages` structure vs `Qwen3VLInference.prepare_inputs`; the WebVoyager default request is the same `messages` schema, so a failure here is unexpected — debug before proceeding.

- [ ] **Step 3: Write the environment notes file**

Create `quest_batch_benchmark/treesparse_env_notes.md` with the **observed** values from Steps 1–2:

```markdown
# TreeSparseAttention environment — verified for the three-way comparison

**Verified:** 2026-05-22 (Task 1 of the TreeSparse three-way comparison plan).

TreeSparseAttention is a separate project with its own environment. The
three-way benchmark does NOT rebuild it — it activates the existing env.

## Location & activation

- Project: `/vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention`
- venv: `<project>/.venv` (Python 3.13.2)
- Activation: `cd <project> && source env.sh` (loads Lmod modules, then
  activates `.venv`). A non-interactive script must first source the Lmod
  init — `run_treesparse.sh` does this.

## Verified stack (observed)

- python: <fill: e.g. 3.13.2>
- torch: <fill: e.g. 2.11.0+cu128>, CUDA available: <fill>, device: <fill: NVIDIA B200>
- flashinfer: <fill: e.g. 0.6.9>
- transformers: <fill: e.g. 5.7.0>
- `_tree_sparse_kernels` C++ extension: imports OK (pre-built at
  `build/_tree_sparse_kernels.cpython-313-x86_64-linux-gnu.so`)

## Smoke test

`benchmark_batch.py` on the shared `quest_batch_benchmark/request.json`
(`--batch-sizes 1 --num-decode-tokens 8 --repeat 1 --no-kv-sharing`):
PASSED — prefill_len <fill>, produced a valid result JSON with
`tpot_median_ms`.

## Note on stack divergence from the quest benchmark

TreeSparse uses torch 2.11.0; the quest/dense `sgl.Engine` path uses
torch 2.9.1. Each method requires its own engine and cannot share a venv.
This is a documented, unavoidable difference — see README.md.
```

- [ ] **Step 4: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add quest_batch_benchmark/treesparse_env_notes.md
git commit -m "quest-batch-benchmark-v0.5: verify TreeSparseAttention environment"
```

---

## Task 2: `treesparse_results.py` — TreeSparse JSON → comparison rows

Convert a TreeSparse `benchmark_batch.py` results JSON (keys are batch-size strings; each value has `tpot_mean_ms`, `tpot_median_ms`, `tpot_std_ms`, `throughput_*`, `prefill_len`, `num_decode_tokens`, `repetitions`) into rows matching `aggregate_results.OUT_FIELDS`, with `attention="treesparse"`.

**Files:**
- Create: `quest_batch_benchmark/treesparse_results.py`
- Test: `quest_batch_benchmark/tests/test_treesparse_results.py`

- [ ] **Step 1: Write the failing test**

Create `quest_batch_benchmark/tests/test_treesparse_results.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from treesparse_results import OUT_FIELDS, treesparse_rows


def _entry(tpot_median, tpot_mean, tpot_std=0.5, prefill=9655):
    """A TreeSparse per-batch-size JSON entry (the shape benchmark_batch.py emits)."""
    return {
        "batch_size": 0, "repetitions": 3,
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd quest_batch_benchmark && .venv/bin/python -m pytest tests/test_treesparse_results.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'treesparse_results'`.

- [ ] **Step 3: Write `treesparse_results.py`**

Create `quest_batch_benchmark/treesparse_results.py`:

```python
#!/usr/bin/env python3
"""Convert a TreeSparseAttention batch-benchmark JSON into rows matching the
quest benchmark's tpot_vs_batchsize.csv schema (attention='treesparse').

TreeSparse's harness (`benchmark_batch.py`) runs `repeat` repetitions with NO
separate untimed warmup, so its first repetition absorbs one-time FlashInfer
JIT / cold-cache cost and inflates `tpot_mean_ms`. The quest harness, by
contrast, runs an untimed warmup per batch and reports a warm mean-of-3. To
keep the comparison fair we use TreeSparse's `tpot_median_ms` -- the median of
3 reps discards the single cold repetition -- as the representative TPOT, and
store it in the `tpot_ms_mean` column (the column the comparison table reads).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Must stay identical to aggregate_results.OUT_FIELDS.
OUT_FIELDS = [
    "attention", "batch_size", "model", "topk_val", "input_tokens",
    "max_tokens", "repeat", "status", "tpot_ms_mean", "tpot_ms_std",
    "tpot_ms_min", "tpot_ms_max", "ttft_ms_mean", "decode_time_ms_mean",
    "total_time_ms_mean", "throughput_tok_s_mean",
]

MODEL_TAG = "Qwen3-VL-8B-Instruct"


def treesparse_rows(raw: dict, batch_sizes: list[int], top_k: int) -> list[dict]:
    """One OUT_FIELDS row per batch size, sorted ascending.

    `raw` is the TreeSparse results JSON (top-level keys are batch-size
    strings). A batch size missing from `raw` -- benchmark_batch.py stops at
    the first OOM/error -- becomes a status='error' row with blank metrics.
    """
    rows = []
    for bs in sorted(batch_sizes):
        entry = raw.get(str(bs))
        base = {
            "attention": "treesparse",
            "batch_size": bs,
            "model": MODEL_TAG,
            "topk_val": top_k,
        }
        if entry is None:
            blanks = {k: "" for k in OUT_FIELDS
                      if k.endswith(("_mean", "_std", "_min", "_max"))}
            rows.append({**base, "input_tokens": "", "max_tokens": "",
                         "repeat": 0, "status": "error", **blanks})
            continue
        row = {
            **base,
            "input_tokens": entry["prefill_len"],
            "max_tokens": entry["num_decode_tokens"],
            "repeat": entry["repetitions"],
            "status": "ok",
            # median-of-3 -> drops the cold first rep; see module docstring
            "tpot_ms_mean": float(entry["tpot_median_ms"]),
            "tpot_ms_std": float(entry["tpot_std_ms"]),
            "tpot_ms_min": "",   # TreeSparse JSON reports only mean/median/std
            "tpot_ms_max": "",
            "ttft_ms_mean": "",          # TreeSparse harness reports no TTFT
            "decode_time_ms_mean": "",
            "total_time_ms_mean": "",
            "throughput_tok_s_mean": float(entry["throughput_median"]),
        }
        rows.append({k: (f"{v:.6f}" if isinstance(v, float) else v)
                     for k, v in row.items()})
    return rows


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Convert a TreeSparse results JSON to tpot_vs_batchsize rows")
    p.add_argument("--treesparse-json",
                   default=str(here / "results" / "treesparse_raw.json"))
    p.add_argument("--out-csv",
                   default=str(here / "results" / "treesparse_rows.csv"))
    p.add_argument("--batch-sizes",
                   type=lambda s: [int(x) for x in s.split(",")],
                   default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--top-k", type=int, default=128,
                   help="TreeSparse top-k chunks (run_batch_experiments.sh default).")
    args = p.parse_args()

    with open(args.treesparse_json, encoding="utf-8") as f:
        raw = json.load(f)
    rows = treesparse_rows(raw, args.batch_sizes, args.top_k)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[treesparse] {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd quest_batch_benchmark && .venv/bin/python -m pytest tests/test_treesparse_results.py -q`
Expected: PASS — 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add quest_batch_benchmark/treesparse_results.py quest_batch_benchmark/tests/test_treesparse_results.py
git commit -m "quest-batch-benchmark-v0.5: add TreeSparse results -> comparison-row converter"
```

---

## Task 3: `build_comparison.py` — merge into the three-way table

Merge dense+quest (from `tpot_vs_batchsize.csv`) with TreeSparse (from `treesparse_raw.json`) into `tpot_three_way.csv` (21 rows) and `comparison_table.md` (the deliverable table).

**Files:**
- Create: `quest_batch_benchmark/build_comparison.py`
- Test: `quest_batch_benchmark/tests/test_build_comparison.py`

- [ ] **Step 1: Write the failing test**

Create `quest_batch_benchmark/tests/test_build_comparison.py`:

```python
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aggregate_results import OUT_FIELDS
from build_comparison import format_table, merge


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
    return {"batch_size": 0, "repetitions": 3, "tpot_mean_ms": tpot_median + 2,
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd quest_batch_benchmark && .venv/bin/python -m pytest tests/test_build_comparison.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_comparison'`.

- [ ] **Step 3: Write `build_comparison.py`**

Create `quest_batch_benchmark/build_comparison.py`:

```python
#!/usr/bin/env python3
"""Merge dense + quest (tpot_vs_batchsize.csv) with TreeSparse
(treesparse_raw.json) into one three-way decode-TPOT comparison.

Writes `tpot_three_way.csv` (one OUT_FIELDS row per method x batch size) and
`comparison_table.md` (the human-readable TPOT table), and prints the table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aggregate_results import OUT_FIELDS
from treesparse_results import treesparse_rows

# render order within each batch size
_METHOD_ORDER = {"dense": 0, "quest": 1, "treesparse": 2}


def load_aggregated(csv_path: str) -> list[dict]:
    """dense + quest rows from the quest benchmark's aggregated CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if r["attention"] in ("dense", "quest")]


def merge(quest_csv: str, treesparse_json: str, batch_sizes: list[int],
          top_k: int) -> list[dict]:
    """All three methods' rows, sorted by (batch_size, method)."""
    rows = load_aggregated(quest_csv)
    with open(treesparse_json, encoding="utf-8") as f:
        raw = json.load(f)
    rows = rows + treesparse_rows(raw, batch_sizes, top_k)
    rows.sort(key=lambda r: (int(r["batch_size"]),
                             _METHOD_ORDER.get(r["attention"], 9)))
    return rows


def _tpot(rows: list[dict], attention: str, bs: int):
    """tpot_ms_mean for one (attention, batch size), or None if missing/blank."""
    for r in rows:
        if r["attention"] == attention and int(r["batch_size"]) == bs:
            v = r.get("tpot_ms_mean", "")
            return float(v) if v not in ("", None) else None
    return None


def format_table(rows: list[dict], batch_sizes: list[int]) -> str:
    """Markdown TPOT table: dense / quest / treesparse + speedups vs dense."""
    header = (
        "| batch size | dense TPOT (ms) | quest TPOT (ms) "
        "| treesparse TPOT (ms) | quest vs dense | treesparse vs dense |"
    )
    sep = (
        "|-----------:|----------------:|----------------:"
        "|---------------------:|---------------:|--------------------:|"
    )
    lines = [header, sep]
    for bs in sorted(batch_sizes):
        d = _tpot(rows, "dense", bs)
        q = _tpot(rows, "quest", bs)
        t = _tpot(rows, "treesparse", bs)
        ds = f"{d:.2f}" if d is not None else "—"
        qs = f"{q:.2f}" if q is not None else "—"
        ts = f"{t:.2f}" if t is not None else "—"
        qsp = f"{d / q:.2f}x" if (d and q) else "—"
        tsp = f"{d / t:.2f}x" if (d and t) else "—"
        lines.append(f"| {bs} | {ds} | {qs} | {ts} | {qsp} | {tsp} |")
    return "\n".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Three-way TPOT comparison")
    p.add_argument("--quest-csv",
                   default=str(here / "results" / "tpot_vs_batchsize.csv"))
    p.add_argument("--treesparse-json",
                   default=str(here / "results" / "treesparse_raw.json"))
    p.add_argument("--out-csv",
                   default=str(here / "results" / "tpot_three_way.csv"))
    p.add_argument("--out-md",
                   default=str(here / "results" / "comparison_table.md"))
    p.add_argument("--batch-sizes",
                   type=lambda s: [int(x) for x in s.split(",")],
                   default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--top-k", type=int, default=128)
    args = p.parse_args()

    rows = merge(args.quest_csv, args.treesparse_json, args.batch_sizes,
                 args.top_k)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    table = format_table(rows, args.batch_sizes)
    Path(args.out_md).write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n[compare] {len(rows)} rows -> {args.out_csv}")
    print(f"[compare] table   -> {args.out_md}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd quest_batch_benchmark && .venv/bin/python -m pytest tests/test_build_comparison.py -q`
Expected: PASS — 4 passed.

- [ ] **Step 5: Run the full unit-test suite to confirm no regression**

Run: `cd quest_batch_benchmark && .venv/bin/python -m pytest tests/ -q`
Expected: PASS — 27 passed (19 pre-existing + 4 from Task 2 + 4 here).

- [ ] **Step 6: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add quest_batch_benchmark/build_comparison.py quest_batch_benchmark/tests/test_build_comparison.py
git commit -m "quest-batch-benchmark-v0.5: add three-way TPOT comparison merge"
```

---

## Task 4: `run_treesparse.sh` — orchestrate the TreeSparse benchmark run

A bash script that activates the TreeSparse environment, runs `run_batch_experiments.sh tpot-no-share` on the **shared** `request.json`, and copies the result JSON to `results/treesparse_raw.json`.

**Files:**
- Create: `quest_batch_benchmark/run_treesparse.sh`

- [ ] **Step 1: Write `run_treesparse.sh`**

Create `quest_batch_benchmark/run_treesparse.sh`:

```bash
#!/usr/bin/env bash
# Run the TreeSparseAttention `tpot-no-share` batch benchmark on the SAME
# request.json the quest benchmark uses, so the three-way comparison is fair.
#
# TreeSparseAttention is a separate project with its own Python 3.13 venv and
# pre-built CUDA kernels. This script activates THAT environment (never the
# quest .venv), runs the benchmark, and copies the results JSON back into this
# benchmark's results/ directory as treesparse_raw.json.
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSA="${TSA_DIR:-/vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention}"
REQUEST="$BENCH/request.json"
DEST="$BENCH/results/treesparse_raw.json"

[ -f "$REQUEST" ] || { echo "ERROR: request.json missing at $REQUEST" >&2; exit 1; }
[ -d "$TSA" ]     || { echo "ERROR: TreeSparseAttention missing at $TSA" >&2; exit 1; }

mkdir -p "$BENCH/results"

# Lmod's `module` function is installed by the system profile, which a
# non-interactive script does not load -- source it so env.sh can `module load`.
if ! command -v module >/dev/null 2>&1; then
  for init in /etc/profile.d/Z98-lmod.sh /etc/profile.d/modules.sh \
              /usr/share/lmod/lmod/init/bash; do
    # shellcheck disable=SC1090
    [ -f "$init" ] && source "$init" && break
  done
fi

# Marker so we can confirm the run produced a NEW results JSON (not a stale one).
STAMP="$(mktemp)"

echo ">>> activating TreeSparseAttention environment ($TSA)"
cd "$TSA"
# env.sh loads CUDA/toolchain modules, then activates TreeSparse's own .venv.
# shellcheck disable=SC1091
source env.sh

echo ">>> run_batch_experiments.sh tpot-no-share  (request: $REQUEST)"
# run_batch_experiments.sh forwards every argument after the mode word to
# benchmark_batch.py. The trailing --request-file is therefore appended after
# the script's hardcoded default; argparse keeps the LAST value, so TreeSparse
# runs on the SAME input as the quest/dense runs.
#
# The script's own exit code is intentionally ignored: its final (optional)
# plotting step runs AFTER the results JSON is written and may fail without
# affecting the measurement. Success is verified by locating the JSON below.
bash run_batch_experiments.sh tpot-no-share --request-file "$REQUEST" || true

LATEST="$(ls -t "$TSA"/batch_results/tpot_no_share/results_*/results_*.json \
          2>/dev/null | head -1)"
if [ -z "$LATEST" ]; then
  echo "ERROR: no TreeSparse results JSON under $TSA/batch_results/tpot_no_share/" >&2
  rm -f "$STAMP"; exit 1
fi
if [ ! "$LATEST" -nt "$STAMP" ]; then
  echo "ERROR: newest TreeSparse JSON ($LATEST) predates this run -- the" >&2
  echo "       benchmark produced no fresh results. Check the run log." >&2
  rm -f "$STAMP"; exit 1
fi
rm -f "$STAMP"

cp "$LATEST" "$DEST"
python3 -c "import json; d=json.load(open('$DEST')); \
print('[treesparse]', len(d), 'batch sizes:', sorted(int(k) for k in d))" || {
  echo "ERROR: $DEST is not valid JSON" >&2; exit 1; }

echo ">>> TreeSparse results: $LATEST"
echo "                     -> $DEST"
```

- [ ] **Step 2: Syntax-check the script**

Run: `bash -n quest_batch_benchmark/run_treesparse.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Test the guard logic without a GPU**

Run: `TSA_DIR=/nonexistent bash quest_batch_benchmark/run_treesparse.sh; echo "exit=$?"`
Expected: prints `ERROR: TreeSparseAttention missing at /nonexistent` and `exit=1`.

- [ ] **Step 4: Make the script executable and commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
chmod +x quest_batch_benchmark/run_treesparse.sh
git add quest_batch_benchmark/run_treesparse.sh
git commit -m "quest-batch-benchmark-v0.5: add run_treesparse.sh orchestrator"
```

---

## Task 5: Wire the three-way pipeline into `run_benchmark.sh`

Extend the driver so one invocation runs `dense` → `quest` → `aggregate` → `treesparse` → `build_comparison`.

**Files:**
- Modify: `quest_batch_benchmark/run_benchmark.sh`

- [ ] **Step 1: Replace the tail of `run_benchmark.sh`**

In `quest_batch_benchmark/run_benchmark.sh`, replace this final block:

```bash
echo ">>> aggregating"
"$PY" aggregate_results.py --raw-csv "$RAW" --out-csv "$OUT" || exit 1

echo ">>> done"
echo "    raw      : $RAW"
echo "    processed: $OUT"
```

with:

```bash
echo ">>> aggregating dense + quest"
"$PY" aggregate_results.py --raw-csv "$RAW" --out-csv "$OUT" || exit 1

echo ">>> running treesparse  (TreeSparseAttention's own environment)"
CUDA_VISIBLE_DEVICES="$GPU" bash "$BENCH/run_treesparse.sh" || exit 1

echo ">>> building the three-way comparison"
"$PY" build_comparison.py \
  --quest-csv "$OUT" \
  --treesparse-json "$BENCH/results/treesparse_raw.json" || exit 1

echo ">>> done"
echo "    raw (dense+quest) : $RAW"
echo "    aggregated        : $OUT"
echo "    treesparse raw    : $BENCH/results/treesparse_raw.json"
echo "    three-way CSV     : $BENCH/results/tpot_three_way.csv"
echo "    comparison table  : $BENCH/results/comparison_table.md"
```

(The first line is only reworded — `aggregating` → `aggregating dense + quest` — for clarity. `$PY`, `$RAW`, `$OUT`, `$BENCH`, `$GPU` are all already defined at the top of the existing script.)

- [ ] **Step 2: Syntax-check the modified driver**

Run: `bash -n quest_batch_benchmark/run_benchmark.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add quest_batch_benchmark/run_benchmark.sh
git commit -m "quest-batch-benchmark-v0.5: wire treesparse + comparison into run_benchmark.sh"
```

---

## Task 6: Run the full three-way benchmark

Execute the complete pipeline fresh (the user chose "re-run all three"). This is the long GPU step (~1–1.5 h: dense + quest each ~15–30 min through `sgl.Engine`, treesparse ~15–30 min). Pick a free GPU; the example uses GPU 0.

**Files:**
- Generated: `quest_batch_benchmark/results/{raw_results.csv,tpot_vs_batchsize.csv,treesparse_raw.json,tpot_three_way.csv,comparison_table.md}`

- [ ] **Step 1: Run the full benchmark**

Run (from the worktree root; long-running — run in the background and monitor):

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
GPU=0 bash quest_batch_benchmark/run_benchmark.sh
```

Expected: the driver prints `>>> running dense`, `>>> running quest`, `>>> aggregating dense + quest`, `>>> running treesparse`, `>>> building the three-way comparison`, then the markdown table and `>>> done`.

- [ ] **Step 2: Verify the three-way CSV is complete**

Run:

```bash
cd quest_batch_benchmark
.venv/bin/python -c "
import csv
rows = list(csv.DictReader(open('results/tpot_three_way.csv')))
print('rows:', len(rows))
by = {}
for r in rows:
    by.setdefault(r['attention'], []).append((int(r['batch_size']), r['status']))
for m in ('dense', 'quest', 'treesparse'):
    print(m, sorted(by.get(m, [])))
"
```

Expected: `rows: 21`; each of `dense`, `quest`, `treesparse` lists all 7 batch sizes `1,2,4,8,16,32,64`. Statuses should be `ok`. **If any `treesparse` row is `error`** (e.g. an OOM at bs 64), that is a legitimate measured outcome — it will render as `—` in the table; note it in PROGRESS.md (Task 7) and continue. Investigate only if a row that the v0.5 quest run completed (`status=ok` for dense/quest) now fails.

- [ ] **Step 3: Display the comparison table**

Run: `cat quest_batch_benchmark/results/comparison_table.md`
Expected: a markdown table with a header row and 7 data rows (batch sizes 1–64), each showing dense / quest / treesparse TPOT and the two speedup columns. This is the deliverable the user asked for.

- [ ] **Step 4: Commit the results**

`results/` is git-ignored, so force-add (consistent with the existing committed results):

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add -f quest_batch_benchmark/results/raw_results.csv \
           quest_batch_benchmark/results/tpot_vs_batchsize.csv \
           quest_batch_benchmark/results/treesparse_raw.json \
           quest_batch_benchmark/results/tpot_three_way.csv \
           quest_batch_benchmark/results/comparison_table.md
git commit -m "quest-batch-benchmark-v0.5: three-way benchmark results (dense, quest, treesparse)"
```

---

## Task 7: Update documentation

Document the third method, the three-way table, and the fairness contract in `README.md` and `PROGRESS.md`.

**Files:**
- Modify: `quest_batch_benchmark/README.md`
- Modify: `quest_batch_benchmark/PROGRESS.md`

- [ ] **Step 1: Update `README.md`**

Make these edits to `quest_batch_benchmark/README.md`:

1. Change the title and opening sentence so the benchmark is described as a **three-way** comparison:
   - Title `# Quest Batch Benchmark — v0.5` → `# Sparse-Attention Batch Benchmark — v0.5 (dense vs Quest vs TreeSparse)`
   - First paragraph: state it compares **dense**, **Quest**, and **TreeSparseAttention** decode TPOT.

2. After the existing `## Results` section, add a new section `## Three-way comparison — TreeSparseAttention` containing:
   - The **generated table**: paste the contents of `results/comparison_table.md`.
   - The fairness contract, verbatim from the **Fairness contract** bullet list at the top of this plan (same input `request.json`; same 256-token output; same `1..64 × repeat 3` sweep; TreeSparse uses `tpot_median_ms` because its harness has no warmup rep; documented unavoidable differences — separate engines/torch versions, quest `topk_val=64` vs TreeSparse `top-k=128`).
   - This explanatory paragraph:

     > **What TreeSparseAttention is.** A standalone sparse-attention library
     > (`/vast/.../sparse_attn/TreeSparseAttention`) — *not* a vortex/sglang
     > plugin. It parses the prompt into a semantic tree of chunks, scores each
     > decode query against per-chunk key centroids (FP8), selects the top-`k`
     > chunks per layer, and runs FlashInfer tensor-core paged decode on only
     > the selected pages. It has its own Python 3.13 venv, its own CUDA
     > kernels, and its own HuggingFace+FlashInfer harness, so it runs as a
     > separate process; `run_treesparse.sh` drives it on the *same*
     > `request.json` and `build_comparison.py` merges the result.

     > **Why `treesparse` is not a `--attention` mode of `benchmark_quest_tpot.py`.**
     > That harness only drives `sgl.Engine`. TreeSparse does not run under
     > sglang at all — it is a different engine end-to-end. The three-way table
     > is produced by merging two independent measurements, not by one harness.

3. In the `## Files` table, add rows for `run_treesparse.sh`, `treesparse_results.py`, `build_comparison.py`, `treesparse_env_notes.md`, `results/treesparse_raw.json`, `results/tpot_three_way.csv`, `results/comparison_table.md`.

4. In the `## Reproduce` section, note that `run_benchmark.sh` now also runs TreeSparse and that TreeSparse uses its own pre-built environment (see `treesparse_env_notes.md`); a separate environment build is **not** required.

- [ ] **Step 2: Update `PROGRESS.md`**

Make these edits to `quest_batch_benchmark/PROGRESS.md`:

1. `**Status:**` line → note the benchmark now also includes TreeSparseAttention (three-way comparison complete).
2. `**Last updated:**` → `2026-05-22`.
3. Add the three-way table (paste `results/comparison_table.md`) under a new `## Three-way result` heading.
4. Add the new commits (from Tasks 1–6) to the `## Branch commits` table.
5. Add a methodology bullet: TreeSparse runs in its own env via `run_treesparse.sh` on the shared `request.json`; the merge uses `tpot_median_ms` (TreeSparse has no warmup rep); document any `status=error` rows observed in Task 6.

- [ ] **Step 3: Run the full unit-test suite once more**

Run: `cd quest_batch_benchmark && .venv/bin/python -m pytest tests/ -q`
Expected: PASS — 27 passed.

- [ ] **Step 4: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add quest_batch_benchmark/README.md quest_batch_benchmark/PROGRESS.md
git commit -m "quest-batch-benchmark-v0.5: document the three-way TreeSparse comparison"
```

---

## Self-Review

**Spec coverage:**
- *Add TreeSparseAttention to the comparison* → Tasks 2–6 (converter, merge, orchestrator, run).
- *Fair setting — same input* → `run_treesparse.sh` overrides TreeSparse's request with the shared `request.json` (Task 4); verified by the Task 6 CSV check.
- *Same output / same TPOT setting* → 256 decode tokens, `1..64 × repeat 3` on both harnesses; `tpot_median_ms` chosen for warm-vs-warm fairness (Task 2).
- *"See if new environment has to be set up"* → Task 1 verifies the **pre-existing** TreeSparse env; a new build is needed only as a documented fallback. Answer recorded in `treesparse_env_notes.md`.
- *"Run with `run_batch_experiments.sh tpot-no-share`"* → `run_treesparse.sh` invokes exactly that command (Task 4).
- *Output: a table of TPOT for all three methods, bs 1–64* → `comparison_table.md` / `tpot_three_way.csv` (Tasks 3, 6).

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command shows expected output. The only "fill in" is the *observed* environment versions in `treesparse_env_notes.md` (Task 1) — those are runtime observations, not design placeholders.

**Type consistency:** `OUT_FIELDS` is defined once in `aggregate_results.py`, re-declared identically (with a "must stay identical" comment) in `treesparse_results.py`, and imported from `aggregate_results` by `build_comparison.py`. `treesparse_rows(raw, batch_sizes, top_k)` has the same signature in its definition (Task 2), its tests (Task 2), and its caller `merge()` (Task 3). `merge()` and `format_table()` signatures match between `build_comparison.py` and `test_build_comparison.py`.
