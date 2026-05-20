# Quest Batch Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure Quest sparse-attention decode speed (TPOT, ms/token) against a dense-attention baseline at decode batch sizes 1, 2, 4, 8, 16, 32, 64, on the `request_005` input, and emit a raw per-step CSV plus a processed `tpot_vs_batchsize` CSV.

**Architecture:** A single-process harness boots one sglang `ModelRunner` (via `sglang.bench_one_batch`) for one attention mode, replicates the benchmark request to a fixed batch size, prefills it, then times each decode step with CUDA events — the per-step latency *is* TPOT (one decode step emits exactly one token per request). A shell driver runs the harness twice (quest, dense); an aggregator collapses the raw per-step CSV into the processed curve.

**Tech Stack:** Python 3.12, a dedicated `uv` venv mirroring the project's `vortex` conda env (torch 2.7.1+cu126, sgl-kernel 0.2.4, flashinfer-python 0.2.7.post1), the `vortex_torch` sparse-attention framework, the patched sglang 0.4.9 fork (`dreaming-panda/sglang`, branch `graph`), 1× NVIDIA B200.

---

## Context & Key Findings (read before starting)

These were established by inspecting the repo and the reference conda env. They are *facts the plan depends on* — do not re-derive them.

1. **Model.** Quest runs inside `vortex_torch`, which uses a patched sglang 0.4.9 fork whose model registry has **no `qwen3_vl`** — only text `qwen3`. The original target `Qwen/Qwen3-VL-8B-Instruct` therefore cannot load. Per the user's decision, the benchmark uses **`Qwen/Qwen3-8B`** — the text twin of Qwen3-VL-8B's backbone (identical 36 layers / 32 heads / 8 KV heads / head_dim 128). The benchmark request is text-only, and decode TPOT is governed entirely by the text transformer, so the numbers transfer faithfully.

2. **Quest algorithm.** Registered as `gqa_quest_sparse_attention` in `vortex_torch/flow/algorithms.py:357` (QUEST query-envelope matching, https://arxiv.org/abs/2406.10774). It is selected with `enable_vortex_sparsity=True` + `vortex_module_name="gqa_quest_sparse_attention"`. The **dense baseline** is the same engine with `enable_vortex_sparsity=False` (full flashinfer attention).

3. **Benchmark request.** `/vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention/html_request/request_005_20260316_221014/request.json` — a JSON dict with `model`, `messages` (a `system` + a `user` message, both pure text, ~28.6 K chars total → roughly 7–8.5 K tokens after the chat template), and `parameters`. It will be **copied into the benchmark folder** for self-containment.

4. **Harness pattern.** `vortex_torch/profile_decode.py` is a near-complete template: it uses `from sglang.bench_one_batch import decode, extend, load_model`, builds `Req` objects by hand, prefills with `extend(reqs, model_runner)`, then loops `decode(next_token_ids, batch, model_runner)`. `bench_one_batch` is the canonical fixed-batch-size decode tool. `profile_decode.py` *disables* vortex sparsity — this plan *enables* it, which is just a server-arg flag; integration is verified in the smoke test (Task 5).

5. **Fairness — KV-cache reuse disabled.** Per the user's explicit requirement, the engine is booted with `disable_radix_cache=True`. `bench_one_batch` already bypasses the radix tree by constructing `Req`s with `prefix_indices=[]`, but the flag makes it explicit and guaranteed for both modes.

6. **GPU / memory.** Host `dgx006`, 1× dedicated B200 (183 GB). Qwen3-8B bf16 KV cache costs ~288 KB/token; at ~8 K context, batch 64 ≈ 147 GB of KV plus prefill activations — it may OOM. Per the user's decision, KV stays **bf16**; a batch size that OOMs is recorded as `status=oom` and the ascending sweep stops.

7. **Reference env.** Conda env `vortex` at `/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/vortex` (Python 3.12, torch 2.7.1+cu126, sgl-kernel 0.2.4, flashinfer-python 0.2.7.post1, flashinfer-cubin 0.6.8.post1, triton 3.3.1, numpy 2.3.5) is the *known-good* version set. Its torch prints `sm_100 not compatible` on the B200 but runs via PTX JIT. The uv venv mirrors these versions. Its `nvcc` (CUDA 12.6) at `/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/vortex/bin/nvcc` is borrowed only to *build* the `vortex_torch_C` CUDA extension.

---

## File Structure

All paths relative to `/vast/projects/liuv/pennnetworks/xutingl/vortex_torch/` (the `vortex_torch` git repo root).

```
quest_batch_benchmark/
  IMPLEMENTATION_PLAN.md      # this file
  README.md                   # description + reproduce instructions (Task 9)
  request.json                # copy of the benchmark request (Task 3)
  prompt_io.py                # request.json -> token ids (Task 3)
  benchmark_quest_tpot.py      # the GPU harness: one attention mode -> raw CSV rows (Task 4)
  aggregate_results.py         # raw_results.csv -> tpot_vs_batchsize.csv (Task 6)
  run_benchmark.sh             # driver: runs harness for dense + quest, then aggregates (Task 7)
  setup_env.sh                 # creates the uv venv + installs the stack (Task 1)
  reference_freeze.txt         # pip freeze of the `vortex` conda env, for provenance (Task 1)
  pytest.ini                   # pytest config (Task 3)
  tests/
    test_prompt_io.py          # Task 3
    test_summarize.py          # Task 4
    test_aggregate.py          # Task 6
  results/
    raw_results.csv            # per-(mode,batch,step) latencies (produced by Task 8)
    tpot_vs_batchsize.csv       # processed curve (produced by Task 8)
  logs/                        # per-run stdout/stderr (produced by Task 8)
  .venv/                       # uv venv (gitignored)
  .vortex_cache/               # vortex JIT compilation cache (gitignored)
  .gitignore
```

**Responsibilities:** `prompt_io.py` is pure (no GPU) and unit-tested. `benchmark_quest_tpot.py` mixes pure helpers (`summarize_step_latencies` — unit-tested) with the GPU run loop (verified by the Task 5 smoke run, not unit tests). `aggregate_results.py` is pure and unit-tested. `run_benchmark.sh` is orchestration only.

---

## Risks & Contingencies

Read this section before Task 1. These are the three things most likely to go wrong.

- **R1 — uv venv build on the B200 (highest risk).** The stack mixes prebuilt wheels (torch, sgl-kernel, flashinfer) with two source installs (the sglang fork, `vortex_torch`). torch 2.7.1+cu126 is built for ≤ sm_90 and runs on the B200 (sm_100) only via PTX JIT. Mitigations baked into Task 1: mirror the *exact* `vortex` conda-env versions; build `vortex_torch_C` with `TORCH_CUDA_ARCH_LIST="9.0+PTX"` (so nvcc 12.6 emits PTX that JITs to sm_100); a hard verification gate (Step 8 of Task 1) that runs a real CUDA matmul on the B200. If the gate fails, stop and report — do not proceed to GPU tasks.

- **R2 — vortex sparsity through `bench_one_batch`.** `profile_decode.py` only ever exercised `bench_one_batch` with vortex *disabled*. The vortex backend attaches at `model_runner.attn_backend`, which `bench_one_batch`'s `extend`/`decode` do use, so it *should* work. Task 5 is a hard gate: it asserts the resolved decode attention backend class is the vortex one and that a quest decode step runs. **Contingency if Task 5 fails:** switch to the blessed `sgl.Engine` path (proven by `examples/run_lcb.py`) using the *delta method* — for each `(mode, batch_size)` call `llm.generate()` on `batch_size` identical prompts (with `ignore_eos=True`) twice, once with `max_new_tokens=S` and once with `=L`; `TPOT = (mean_e2e_latency(L) − mean_e2e_latency(S)) / (L − S)`, which cancels prefill/queue. This keeps the same CSV schema (`step_idx` becomes the trial index). Do not pre-build this path; only pivot if Task 5's gate fails.

- **R3 — batch-64 OOM.** Expected and handled: the harness catches `torch.cuda.OutOfMemoryError` / out-of-memory `RuntimeError`, writes a `status=oom` row, and breaks the ascending sweep. Not a blocker.

---

## Task 1: Create the uv venv and install the stack

**Files:**
- Create: `quest_batch_benchmark/setup_env.sh`
- Create: `quest_batch_benchmark/reference_freeze.txt` (generated)
- Create: `quest_batch_benchmark/.gitignore`

This task has inherent iteration risk (R1). Run the steps interactively; the verification gate (Step 8) is the success criterion.

- [ ] **Step 1: Write `.gitignore`**

Create `quest_batch_benchmark/.gitignore`:

```
.venv/
.vortex_cache/
results/
logs/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 2: Fetch the sglang fork submodule**

The `vortex_torch` repo pins the sglang fork as a submodule but it is not checked out.

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
git submodule update --init third_party/sglang
ls third_party/sglang/python/sglang/srt/models/qwen3.py
```
Expected: the `qwen3.py` path prints (file exists). If the submodule fails to fetch, stop and report.

- [ ] **Step 3: Capture the known-good version set**

Run:
```bash
VORTEX_ENV=/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/vortex
"$VORTEX_ENV/bin/python" -m pip freeze \
  | grep -v '^-e ' \
  > /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark/reference_freeze.txt
wc -l quest_batch_benchmark/reference_freeze.txt
```
Expected: a non-empty file. This is provenance only — Task 1 pins the critical packages explicitly below; `reference_freeze.txt` is the reference if a version needs checking.

- [ ] **Step 4: Write `setup_env.sh`**

Create `quest_batch_benchmark/setup_env.sh`:

```bash
#!/usr/bin/env bash
# Create the uv venv for the Quest batch benchmark and install the full stack.
# Mirrors the known-good `vortex` conda env. Re-runnable.
set -euo pipefail

REPO=/vast/projects/liuv/pennnetworks/xutingl/vortex_torch
BENCH="$REPO/quest_batch_benchmark"
VENV="$BENCH/.venv"
# nvcc (CUDA 12.6) is borrowed from the reference conda env only to BUILD
# the vortex_torch_C CUDA extension.
BUILD_CUDA_HOME=/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/vortex
TORCH_INDEX=https://download.pytorch.org/whl/cu126

cd "$REPO"

echo "[1/6] create uv venv (python 3.12)"
uv venv --python 3.12 "$VENV"

echo "[2/6] install torch 2.7.1 (cu126 wheels)"
uv pip install --python "$VENV" \
  --index-url "$TORCH_INDEX" \
  torch==2.7.1 torchvision==0.22.1

echo "[3/6] install the bundled-stack wheels (pinned to the vortex conda env)"
uv pip install --python "$VENV" \
  sgl-kernel==0.2.4 \
  flashinfer-python==0.2.7.post1 \
  flashinfer-cubin==0.6.8.post1 \
  triton==3.3.1 \
  numpy==2.3.5 \
  pandas \
  orjson \
  uvloop \
  xgrammar==0.2.0 \
  "huggingface-hub[cli]" \
  pytest

echo "[4/6] install the patched sglang fork (editable, deps resolved)"
uv pip install --python "$VENV" -e "$REPO/third_party/sglang/python"

echo "[5/6] build + install vortex_torch (editable, CUDA extension)"
# nvcc 12.6 cannot emit sm_100 SASS; 9.0+PTX JITs to the B200 at runtime.
CUDA_HOME="$BUILD_CUDA_HOME" \
PATH="$BUILD_CUDA_HOME/bin:$PATH" \
TORCH_CUDA_ARCH_LIST="9.0+PTX" \
  uv pip install --python "$VENV" -e "$REPO" --no-build-isolation

echo "[6/6] done — venv at $VENV"
```

Make it executable: `chmod +x quest_batch_benchmark/setup_env.sh`.

- [ ] **Step 5: Run the install**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
bash quest_batch_benchmark/setup_env.sh
```
Expected: all 6 phases finish without error. If phase 4 reports a dependency conflict, note the conflicting package and re-pin it from `reference_freeze.txt`, then re-run. If phase 5 fails to compile, confirm `nvcc` resolves: `/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/vortex/bin/nvcc --version`.

- [ ] **Step 6: Verify the imports**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
quest_batch_benchmark/.venv/bin/python - <<'PY'
import torch, sglang, vortex_torch
import vortex_torch_C
from sglang.bench_one_batch import decode, extend, load_model
print("torch", torch.__version__)
print("sglang", sglang.__version__)
print("imports OK")
PY
```
Expected: prints `torch 2.7.1+cu126`, an sglang version, and `imports OK`. A `vortex_torch_C` `ModuleNotFoundError` means the CUDA extension build (phase 5) silently failed — re-run phase 5 and read its compiler output.

- [ ] **Step 7: Verify the sglang fork lacks `qwen3_vl` (sanity check on Context finding 1)**

Run:
```bash
ls third_party/sglang/python/sglang/srt/models/ | grep -E 'qwen3'
```
Expected: `qwen3.py` and `qwen3_moe.py` only — confirms the model substitution decision.

- [ ] **Step 8: VERIFICATION GATE — real CUDA op on the B200**

Run:
```bash
quest_batch_benchmark/.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
c = (a @ b)
torch.cuda.synchronize()
print("matmul mean:", float(c.float().mean()))
print("CUDA GATE PASSED")
PY
```
Expected: prints `device: NVIDIA B200`, `capability: (10, 0)`, a finite matmul mean, and `CUDA GATE PASSED`. **If this errors (e.g. `no kernel image is available for execution on the device`), STOP — the PTX-JIT path failed; report it and do not continue to GPU tasks.**

- [ ] **Step 9: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
git add quest_batch_benchmark/setup_env.sh quest_batch_benchmark/.gitignore quest_batch_benchmark/reference_freeze.txt
git add .gitmodules third_party/sglang
git commit -m "quest-batch-benchmark: uv venv + stack setup"
```

---

## Task 2: Download the Qwen3-8B checkpoint

**Files:**
- Downloads to: `/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B/`

- [ ] **Step 1: Download Qwen3-8B**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
quest_batch_benchmark/.venv/bin/hf download Qwen/Qwen3-8B \
  --local-dir /vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B
```
Expected: ~16 GB of safetensors download. If `hf` is not found, use `quest_batch_benchmark/.venv/bin/huggingface-cli download` with the same arguments.

- [ ] **Step 2: Verify the checkpoint**

Run:
```bash
ls /vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B/
quest_batch_benchmark/.venv/bin/python - <<'PY'
import json
cfg = json.load(open("/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B/config.json"))
print("architectures:", cfg["architectures"])
print("num_hidden_layers:", cfg["num_hidden_layers"])
print("num_key_value_heads:", cfg["num_key_value_heads"])
PY
```
Expected: `config.json`, `tokenizer.json`, and 4 `model-0000*.safetensors` present; `architectures: ['Qwen3ForCausalLM']`, `num_hidden_layers: 36`, `num_key_value_heads: 8`. (This is a no-commit task — the model lives outside the repo.)

---

## Task 3: Scaffold the folder, copy the request, write `prompt_io.py`

**Files:**
- Create: `quest_batch_benchmark/request.json` (copied)
- Create: `quest_batch_benchmark/prompt_io.py`
- Create: `quest_batch_benchmark/pytest.ini`
- Test: `quest_batch_benchmark/tests/test_prompt_io.py`

- [ ] **Step 1: Copy the benchmark request into the folder**

Run:
```bash
cp /vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention/html_request/request_005_20260316_221014/request.json \
   /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark/request.json
```

- [ ] **Step 2: Write `pytest.ini`**

Create `quest_batch_benchmark/pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 3: Write the failing test for `prompt_io`**

Create `quest_batch_benchmark/tests/test_prompt_io.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_io import build_input_ids, load_request_messages

MODEL_DIR = "/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B"
REQUEST = os.path.join(os.path.dirname(__file__), "..", "request.json")


def test_load_request_messages_has_system_and_user():
    messages = load_request_messages(REQUEST)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all(isinstance(m["content"], str) and m["content"] for m in messages)


def test_build_input_ids_returns_long_token_list():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = build_input_ids(REQUEST, tok)
    assert isinstance(ids, list)
    assert all(isinstance(t, int) for t in ids)
    # request_005 is ~28.6K chars of system+user text -> several thousand tokens
    assert 3000 < len(ids) < 20000
```

- [ ] **Step 4: Run the test to verify it fails**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
.venv/bin/python -m pytest tests/test_prompt_io.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'prompt_io'`.

- [ ] **Step 5: Write `prompt_io.py`**

Create `quest_batch_benchmark/prompt_io.py`:

```python
"""Load the benchmark request and turn it into model input token ids."""
from __future__ import annotations

import json
from typing import List


def load_request_messages(request_path: str) -> List[dict]:
    """Return the chat `messages` list from a request.json file."""
    with open(request_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["messages"]


def build_input_ids(request_path: str, tokenizer) -> List[int]:
    """Apply the model chat template to the request and return token ids.

    `add_generation_prompt=True` appends the assistant turn marker so the
    model is positioned to decode the first output token.
    """
    messages = load_request_messages(request_path)
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    return [int(t) for t in ids]
```

- [ ] **Step 6: Run the test to verify it passes**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
.venv/bin/python -m pytest tests/test_prompt_io.py -v
```
Expected: both tests PASS. Note the printed token count of `request_005` for the README later.

- [ ] **Step 7: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
git add quest_batch_benchmark/request.json quest_batch_benchmark/prompt_io.py \
        quest_batch_benchmark/pytest.ini quest_batch_benchmark/tests/test_prompt_io.py
git commit -m "quest-batch-benchmark: request.json + prompt_io with tests"
```

---

## Task 4: Write the benchmark harness

**Files:**
- Create: `quest_batch_benchmark/benchmark_quest_tpot.py`
- Test: `quest_batch_benchmark/tests/test_summarize.py`

The harness's pure helper (`summarize_step_latencies`) is unit-tested here; the GPU run loop is verified by the Task 5 smoke run.

- [ ] **Step 1: Write the failing test for `summarize_step_latencies`**

Create `quest_batch_benchmark/tests/test_summarize.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
.venv/bin/python -m pytest tests/test_summarize.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark_quest_tpot'`.

- [ ] **Step 3: Write `benchmark_quest_tpot.py`**

Create `quest_batch_benchmark/benchmark_quest_tpot.py`:

```python
#!/usr/bin/env python3
"""Quest decode-speed (TPOT) benchmark at fixed decode batch sizes.

Boots ONE sglang ModelRunner (via sglang.bench_one_batch) for a single
attention mode (`quest` or `dense`). Then, for each batch size in
ascending order, it:

  * replicates the benchmark request `batch_size` times into Req objects,
  * prefills the batch with extend(),
  * runs `warmup` then `measured` decode steps,
  * times every measured decode step with CUDA events.

A decode step emits exactly one output token per request, so the step
latency *is* the time-per-output-token (TPOT). One CSV row is written per
measured step. On CUDA OOM the batch size is recorded with status=oom and
the (ascending) sweep stops — larger batches would also OOM.

KV-cache reuse is disabled (disable_radix_cache=True) so quest and dense
are compared on equal footing.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch

import vortex_torch  # noqa: F401  -- registers the vortex attention backend
from vortex_torch.engine.sgl import DEFAULT_SCHEDULE_POLICY

from sglang.bench_one_batch import decode, extend, load_model
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger

from transformers import AutoTokenizer

from prompt_io import build_input_ids

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
QUEST_MODULE = "gqa_quest_sparse_attention"

RAW_CSV_FIELDS = [
    "run_timestamp", "attention", "batch_size", "model", "topk_val",
    "input_tokens", "warmup_steps", "measured_steps", "step_idx",
    "step_latency_ms", "status",
]


def summarize_step_latencies(step_ms: List[float]) -> dict:
    """Aggregate per-step decode latencies (ms) into TPOT statistics."""
    ordered = sorted(step_ms)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        idx = (n - 1) * p
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    return {
        "tpot_ms_mean": statistics.fmean(step_ms),
        "tpot_ms_p50": pct(0.50),
        "tpot_ms_p90": pct(0.90),
        "tpot_ms_std": statistics.pstdev(step_ms) if n > 1 else 0.0,
        "tpot_ms_min": min(step_ms),
        "tpot_ms_max": max(step_ms),
    }


def build_server_args(args, n_input_tokens: int) -> ServerArgs:
    """Construct ServerArgs for one attention mode.

    Mirrors profile_decode.py's dataclass-backfill so it is robust to
    ServerArgs fields that this sglang build does not expose via CLI.
    """
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns, _ = parser.parse_known_args([])
    for field in dataclasses.fields(ServerArgs):
        if hasattr(ns, field.name):
            continue
        if field.default is not dataclasses.MISSING:
            setattr(ns, field.name, field.default)
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
            setattr(ns, field.name, field.default_factory())  # type: ignore[misc]
        else:
            setattr(ns, field.name, None)
    ns.model_path = args.model_path
    server_args = ServerArgs.from_cli_args(ns)

    is_quest = args.attention == "quest"
    # headroom for input + warmup + measured decode tokens
    max_seq = max(args.max_seq_lens, n_input_tokens + args.warmup_steps + args.measured_steps + 64)

    # --- backend + scheduling -------------------------------------------
    server_args.model_path = args.model_path
    server_args.attention_backend = "flashinfer"
    server_args.disable_overlap_schedule = True
    server_args.disable_cuda_graph = False
    server_args.disable_radix_cache = True            # fair: no KV-cache reuse
    server_args.tp_size = 1
    server_args.mem_fraction_static = args.mem_fraction_static
    server_args.context_length = max_seq
    server_args.kv_cache_dtype = "auto"               # bf16 KV
    try:
        server_args.cuda_graph_max_bs = max(args.batch_sizes)
    except Exception:
        pass

    # --- vortex / quest --------------------------------------------------
    server_args.enable_vortex_sparsity = is_quest
    server_args.vortex_module_name = QUEST_MODULE
    server_args.vortex_block_size = 16
    server_args.page_size = 16
    server_args.vortex_topk_val = args.topk_val
    server_args.vortex_topk_ratio = 0.0               # pure static budget
    server_args.vortex_block_reserved_bos = 1
    server_args.vortex_block_reserved_eos = 2
    server_args.vortex_workload_chunk_size = 32
    server_args.vortex_layers_skip = [0]
    server_args.vortex_schedule_policy = DEFAULT_SCHEDULE_POLICY
    server_args.vortex_dtype = "bfloat16"
    server_args.vortex_max_seq_lens = max_seq
    server_args.vortex_compilation_cache_dir = args.vortex_cache_dir
    return server_args


def make_reqs(input_ids: List[int], batch_size: int, max_new_tokens: int) -> List[Req]:
    """Replicate the request `batch_size` times as prefix-free Req objects."""
    sampling_params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)
    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []                       # no radix-cache prefix sharing
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)
    return reqs


def time_decode(next_token_ids, batch, model_runner, warmup: int, measured: int) -> List[float]:
    """Run warmup + measured decode steps; return per-step latency in ms."""
    for _ in range(warmup):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
    torch.cuda.synchronize()

    events = [torch.cuda.Event(enable_timing=True) for _ in range(measured + 1)]
    events[0].record()
    for i in range(measured):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        events[i + 1].record()
    torch.cuda.synchronize()
    return [events[i].elapsed_time(events[i + 1]) for i in range(measured)]


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def run(args) -> None:
    raw_path = Path(args.raw_csv)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not raw_path.exists() or raw_path.stat().st_size == 0

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    input_ids = build_input_ids(args.request, tokenizer)
    n_input = len(input_ids)
    print(f"[setup] attention={args.attention}  input_tokens={n_input}")

    server_args = build_server_args(args, n_input)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    _set_envs_and_config(server_args)
    configure_logger(server_args, prefix=" TP0")
    port_args = PortArgs.init_new(server_args)
    model_runner, _ = load_model(server_args, port_args, tp_rank=0)

    backend = getattr(model_runner, "attn_backend", None)
    backend_cls = f"{type(backend).__module__}.{type(backend).__name__}" if backend else "<none>"
    print(f"[setup] resolved attention backend: {backend_cls}")
    print(f"[setup] enable_vortex_sparsity={server_args.enable_vortex_sparsity}")

    max_new_tokens = args.warmup_steps + args.measured_steps + 8
    run_ts = datetime.now(timezone.utc).isoformat()
    model_tag = Path(args.model_path).name

    with raw_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
        if write_header:
            writer.writeheader()

        def emit(batch_size, step_idx, latency_ms, status):
            writer.writerow({
                "run_timestamp": run_ts,
                "attention": args.attention,
                "batch_size": batch_size,
                "model": model_tag,
                "topk_val": args.topk_val if args.attention == "quest" else "",
                "input_tokens": n_input,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.measured_steps,
                "step_idx": step_idx,
                "step_latency_ms": "" if latency_ms is None else f"{latency_ms:.6f}",
                "status": status,
            })

        for batch_size in sorted(args.batch_sizes):
            print(f"[run] batch_size={batch_size} ...", flush=True)
            try:
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                reqs = make_reqs(input_ids, batch_size, max_new_tokens)
                with torch.no_grad():
                    next_token_ids, _, batch = extend(reqs, model_runner)
                    torch.cuda.synchronize()
                    step_ms = time_decode(
                        next_token_ids, batch, model_runner,
                        args.warmup_steps, args.measured_steps,
                    )
                for step_idx, latency in enumerate(step_ms):
                    emit(batch_size, step_idx, latency, "ok")
                f.flush()
                stats = summarize_step_latencies(step_ms)
                print(f"[run] batch_size={batch_size}  TPOT mean={stats['tpot_ms_mean']:.3f} ms "
                      f"p50={stats['tpot_ms_p50']:.3f}  p90={stats['tpot_ms_p90']:.3f}")
            except Exception as exc:  # noqa: BLE001
                if _is_oom(exc):
                    print(f"[run] batch_size={batch_size} OOM -- recording N/A, stopping sweep")
                    emit(batch_size, -1, None, "oom")
                    f.flush()
                    torch.cuda.empty_cache()
                    break
                emit(batch_size, -1, None, "error")
                f.flush()
                raise

    print(f"[done] raw rows written to {raw_path}")


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Quest decode-speed (TPOT) batch benchmark")
    p.add_argument("--attention", choices=["quest", "dense"], required=True)
    p.add_argument("--model-path",
                   default="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B")
    p.add_argument("--request", default=str(here / "request.json"))
    p.add_argument("--raw-csv", default=str(here / "results" / "raw_results.csv"))
    p.add_argument("--batch-sizes", type=lambda s: [int(x) for x in s.split(",")],
                   default=DEFAULT_BATCH_SIZES)
    p.add_argument("--topk-val", type=int, default=64,
                   help="Quest static block budget (blocks of 16 tokens kept).")
    p.add_argument("--warmup-steps", type=int, default=16)
    p.add_argument("--measured-steps", type=int, default=128)
    p.add_argument("--max-seq-lens", type=int, default=16384)
    p.add_argument("--mem-fraction-static", type=float, default=0.9)
    p.add_argument("--vortex-cache-dir", default=str(here / ".vortex_cache"))
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
.venv/bin/python -m pytest tests/test_summarize.py -v
```
Expected: both tests PASS. (The GPU run loop is intentionally not unit-tested — Task 5 verifies it.)

- [ ] **Step 5: Verify the harness `--help` works (no GPU)**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
.venv/bin/python benchmark_quest_tpot.py --help
```
Expected: argparse prints usage including `--attention {quest,dense}`. An ImportError here means the stack from Task 1 is incomplete.

- [ ] **Step 6: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
git add quest_batch_benchmark/benchmark_quest_tpot.py quest_batch_benchmark/tests/test_summarize.py
git commit -m "quest-batch-benchmark: decode-speed harness with TPOT stats"
```

---

## Task 5: Smoke test — VERIFICATION GATE

**Files:**
- Produces: `quest_batch_benchmark/results/smoke_raw.csv` (temporary)

This gate decides whether the `bench_one_batch` path carries vortex sparsity (risk R2).

- [ ] **Step 1: Smoke-run the dense baseline at batch size 1**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark_quest_tpot.py \
  --attention dense --batch-sizes 1 --warmup-steps 4 --measured-steps 8 \
  --raw-csv results/smoke_raw.csv
```
Expected: prints `input_tokens=<N>`, a resolved attention backend class, `enable_vortex_sparsity=False`, and a `[run] batch_size=1 TPOT mean=...` line. Note `<N>` (the request token count).

- [ ] **Step 2: Smoke-run Quest at batch size 1**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark_quest_tpot.py \
  --attention quest --batch-sizes 1 --warmup-steps 4 --measured-steps 8 \
  --raw-csv results/smoke_raw.csv
```
Expected: prints `enable_vortex_sparsity=True`, a resolved attention backend class that **differs from the dense run** (a vortex/sparse backend class name — e.g. contains `vortex` or `sparse`), and a `TPOT mean=...` line. The first quest run JIT-compiles kernels into `.vortex_cache/` (slower; fine).

- [ ] **Step 3: GATE — confirm both modes ran and rows landed**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
.venv/bin/python - <<'PY'
import csv
rows = list(csv.DictReader(open("results/smoke_raw.csv")))
ok = [r for r in rows if r["status"] == "ok"]
modes = {r["attention"] for r in ok}
print("ok rows:", len(ok), "modes:", sorted(modes))
assert modes == {"dense", "quest"}, modes
assert all(float(r["step_latency_ms"]) > 0 for r in ok)
print("SMOKE GATE PASSED")
PY
rm -f results/smoke_raw.csv
```
Expected: `SMOKE GATE PASSED`.

**If Step 2 errors, or the quest backend class is identical to the dense one, the `bench_one_batch` path does not carry vortex sparsity — STOP and switch to the `sgl.Engine` delta-method contingency described in Risks R2.** Do not continue to Task 7 until this gate passes.

- [ ] **Step 4: Commit (gate result note)**

No code changed; record the gate outcome in the commit message of the next task. Proceed.

---

## Task 6: Write the results aggregator

**Files:**
- Create: `quest_batch_benchmark/aggregate_results.py`
- Test: `quest_batch_benchmark/tests/test_aggregate.py`

- [ ] **Step 1: Write the failing test for the aggregator**

Create `quest_batch_benchmark/tests/test_aggregate.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
.venv/bin/python -m pytest tests/test_aggregate.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'aggregate_results'`.

- [ ] **Step 3: Write `aggregate_results.py`**

Create `quest_batch_benchmark/aggregate_results.py`:

```python
#!/usr/bin/env python3
"""Collapse the per-step raw CSV into a per-config TPOT-vs-batch-size CSV."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from benchmark_quest_tpot import summarize_step_latencies

OUT_FIELDS = [
    "attention", "batch_size", "model", "topk_val", "input_tokens",
    "measured_steps", "status", "tpot_ms_mean", "tpot_ms_p50", "tpot_ms_p90",
    "tpot_ms_std", "tpot_ms_min", "tpot_ms_max",
]


def aggregate(raw_csv: str, out_csv: str) -> None:
    """Read the raw per-step CSV; write one aggregated row per (attention, batch_size)."""
    with open(raw_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["attention"], int(r["batch_size"]))].append(r)

    out_rows = []
    for (attention, batch_size), grp in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sample = grp[0]
        statuses = {r["status"] for r in grp}
        base = {
            "attention": attention,
            "batch_size": batch_size,
            "model": sample["model"],
            "topk_val": sample["topk_val"],
            "input_tokens": sample["input_tokens"],
            "measured_steps": sample["measured_steps"],
        }
        ok_latencies = [
            float(r["step_latency_ms"])
            for r in grp
            if r["status"] == "ok" and r["step_latency_ms"] != ""
        ]
        if ok_latencies:
            stats = summarize_step_latencies(ok_latencies)
            out_rows.append({**base, "status": "ok", **{k: f"{v:.6f}" for k, v in stats.items()}})
        else:
            # oom / error — no latencies; leave stat columns blank
            bad = "oom" if "oom" in statuses else "error"
            blanks = {k: "" for k in OUT_FIELDS if k.startswith("tpot_ms_")}
            out_rows.append({**base, "status": bad, **blanks})

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[aggregate] {len(out_rows)} config rows -> {out_csv}")


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Aggregate raw decode latencies into TPOT-vs-batch-size")
    p.add_argument("--raw-csv", default=str(here / "results" / "raw_results.csv"))
    p.add_argument("--out-csv", default=str(here / "results" / "tpot_vs_batchsize.csv"))
    args = p.parse_args()
    aggregate(args.raw_csv, args.out_csv)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
.venv/bin/python -m pytest tests/test_aggregate.py -v
```
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
git add quest_batch_benchmark/aggregate_results.py quest_batch_benchmark/tests/test_aggregate.py
git commit -m "quest-batch-benchmark: raw->tpot_vs_batchsize aggregator"
```

---

## Task 7: Write the driver script

**Files:**
- Create: `quest_batch_benchmark/run_benchmark.sh`

- [ ] **Step 1: Write `run_benchmark.sh`**

Create `quest_batch_benchmark/run_benchmark.sh`:

```bash
#!/usr/bin/env bash
# Drive the full Quest batch benchmark: dense baseline, then quest, then aggregate.
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH"

GPU="${GPU:-0}"
PY="$BENCH/.venv/bin/python"
RAW="$BENCH/results/raw_results.csv"
OUT="$BENCH/results/tpot_vs_batchsize.csv"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p results logs
rm -f "$RAW"   # fresh raw CSV; harness appends per mode

for mode in dense quest; do
  echo ">>> running $mode  (GPU $GPU)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
    --attention "$mode" \
    --raw-csv "$RAW" \
    2>&1 | tee "logs/${mode}_${TS}.log"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "!!! $mode run failed (exit $status) -- see logs/${mode}_${TS}.log" >&2
    exit "$status"
  fi
done

echo ">>> aggregating"
"$PY" aggregate_results.py --raw-csv "$RAW" --out-csv "$OUT"

echo ">>> done"
echo "    raw      : $RAW"
echo "    processed: $OUT"
```

Make it executable: `chmod +x quest_batch_benchmark/run_benchmark.sh`.

- [ ] **Step 2: Sanity-check the script parses**

Run:
```bash
bash -n /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark/run_benchmark.sh
```
Expected: no output (no syntax errors).

- [ ] **Step 3: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
git add quest_batch_benchmark/run_benchmark.sh
git commit -m "quest-batch-benchmark: full-run driver script"
```

---

## Task 8: Run the full benchmark

**Files:**
- Produces: `quest_batch_benchmark/results/raw_results.csv`, `quest_batch_benchmark/results/tpot_vs_batchsize.csv`

- [ ] **Step 1: Run the driver**

Run (this takes a while — 2 model loads + 14 batch-size sweeps; batch 64 may OOM by design):
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
GPU=0 bash run_benchmark.sh
```
Expected: a `dense` then a `quest` block of `[run] batch_size=N TPOT mean=...` lines, then `[aggregate] ... config rows`. A `batch_size=64 OOM` line is acceptable (recorded as N/A).

- [ ] **Step 2: Inspect the processed CSV**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
column -t -s, results/tpot_vs_batchsize.csv
```
Expected: 14 rows (7 batch sizes × 2 modes; OOM rows have blank `tpot_ms_*`). Verify the trends: TPOT rises with batch size for both modes, and `quest` TPOT ≤ `dense` TPOT at the larger batch sizes (Quest reads ~67 KV blocks vs the full ~500 at ~8 K context).

- [ ] **Step 3: Verify raw CSV completeness**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
.venv/bin/python - <<'PY'
import csv
rows = list(csv.DictReader(open("results/raw_results.csv")))
ok = [r for r in rows if r["status"] == "ok"]
print("total raw rows:", len(rows), " ok step rows:", len(ok))
for status in ("oom", "error"):
    n = sum(1 for r in rows if r["status"] == status)
    if n:
        print(f"  {status} rows: {n}")
assert ok, "no successful measurements recorded"
print("RAW CSV OK")
PY
```
Expected: `RAW CSV OK`, with the ok-row count = (successful configs) × 128.

- [ ] **Step 4: Commit the results**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
git add -f quest_batch_benchmark/results/raw_results.csv quest_batch_benchmark/results/tpot_vs_batchsize.csv
git commit -m "quest-batch-benchmark: decode TPOT results (quest vs dense, bs 1-64)"
```
(The `-f` overrides the `results/` `.gitignore` entry — the committed CSVs are the deliverable; transient logs stay ignored.)

---

## Task 9: Write the README

**Files:**
- Modify: `quest_batch_benchmark/README.md` (currently a one-line stub)

- [ ] **Step 1: Write the README**

Replace `quest_batch_benchmark/README.md` with the following. Fill the bracketed `<...>` values from the actual run: `<N_INPUT_TOKENS>` from Task 5 Step 1, and a short results summary from Task 8 Step 2.

````markdown
# Quest Batch Benchmark

Decode-speed (TPOT) benchmark of **Quest** sparse attention vs a dense-attention
baseline, across decode batch sizes **1, 2, 4, 8, 16, 32, 64**, on the
`vortex_torch` framework.

## What this measures

**Quest** ([arXiv:2406.10774](https://arxiv.org/abs/2406.10774)) is a
query-aware sparse-attention method: at each decode step it scores KV blocks
with a cheap query–envelope product and attends to only the top-`k` blocks.
Here it is the `gqa_quest_sparse_attention` flow registered in
`vortex_torch/flow/algorithms.py`.

**TPOT** (time-per-output-token) is the decode-step latency. The harness uses
`sglang.bench_one_batch` to run a *fixed* decode batch size: it replicates one
request `batch_size` times, prefills, then times each decode step with CUDA
events. A decode step emits exactly one token per request, so the step latency
*is* TPOT. Reported TPOT is the mean over 128 measured steps (after 16 warmup
steps).

## Setup substitutions (read this)

- **Model.** The intended `Qwen/Qwen3-VL-8B-Instruct` cannot load: the patched
  sglang 0.4.9 fork that `vortex_torch` depends on has no `qwen3_vl` model.
  The benchmark therefore uses **`Qwen/Qwen3-8B`** — the text twin of
  Qwen3-VL-8B's backbone (identical 36 layers / 32 heads / 8 KV heads /
  head_dim 128). The benchmark request is text-only and decode TPOT is governed
  entirely by the text transformer, so the numbers transfer faithfully.
- **Input.** `request.json` is a copy of `request_005_20260316_221014` — a
  system + user chat message pair (~`<N_INPUT_TOKENS>` tokens after the chat
  template).
- **Fairness.** The engine is booted with `disable_radix_cache=True` so no KV
  prefix is reused; quest and dense are measured on equal footing.
- **KV cache** stays bf16. Batch 64 at ~8 K context (~147 GB KV) may exceed B200
  memory; if a batch size OOMs it is recorded as `status=oom` and the sweep
  stops.

## Reproduce

Requires 1× NVIDIA B200 (or an sm_90+ GPU) on host `dgx006`.

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch

# 1. one-time: create the uv venv + install the stack (~15-40 min)
bash quest_batch_benchmark/setup_env.sh

# 2. one-time: download the model (~16 GB)
quest_batch_benchmark/.venv/bin/hf download Qwen/Qwen3-8B \
  --local-dir /vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B

# 3. run the full benchmark (dense + quest, then aggregate)
cd quest_batch_benchmark
GPU=0 bash run_benchmark.sh
```

Run unit tests with `.venv/bin/python -m pytest`.

## Files

| File | Purpose |
|------|---------|
| `setup_env.sh` | Creates `.venv` (uv) and installs torch / sglang fork / vortex_torch. |
| `benchmark_quest_tpot.py` | The GPU harness — one attention mode, all batch sizes. |
| `aggregate_results.py` | Collapses the raw per-step CSV into the processed curve. |
| `run_benchmark.sh` | Driver: runs the harness for `dense` then `quest`, then aggregates. |
| `prompt_io.py` | Turns `request.json` into model input token ids. |
| `request.json` | The benchmark input request. |
| `results/raw_results.csv` | Per-(mode, batch size, decode step) latency. |
| `results/tpot_vs_batchsize.csv` | Processed TPOT vs batch size. |

## Tunable knobs

`benchmark_quest_tpot.py` flags: `--topk-val` (Quest block budget, default 64 =
1024 tokens kept), `--batch-sizes`, `--warmup-steps` (16), `--measured-steps`
(128), `--mem-fraction-static` (0.9), `--max-seq-lens` (16384).

## Output CSV schemas

**`raw_results.csv`** — one row per measured decode step:

`run_timestamp, attention, batch_size, model, topk_val, input_tokens,
warmup_steps, measured_steps, step_idx, step_latency_ms, status`

`status` ∈ {`ok`, `oom`, `error`}. An `oom`/`error` config has a single row
with `step_idx=-1` and an empty `step_latency_ms`.

**`tpot_vs_batchsize.csv`** — one row per (attention, batch size):

`attention, batch_size, model, topk_val, input_tokens, measured_steps, status,
tpot_ms_mean, tpot_ms_p50, tpot_ms_p90, tpot_ms_std, tpot_ms_min, tpot_ms_max`

## Results summary

<Fill from results/tpot_vs_batchsize.csv after Task 8 — e.g. a small table of
attention × batch_size → tpot_ms_mean, and a one-line note on Quest's speedup
at the larger batch sizes.>
````

- [ ] **Step 2: Verify the README has no unfilled placeholders**

Run:
```bash
grep -n '<.*>' /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark/README.md || echo "no placeholders left"
```
Expected: `no placeholders left` (every `<...>` filled with real values).

- [ ] **Step 3: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch
git add quest_batch_benchmark/README.md
git commit -m "quest-batch-benchmark: detailed README"
```

---

## Self-Review Notes

- **Spec coverage.** Quest decode TPOT at batch 1–64 → Task 4/8. CSV raw + processed → Task 4 (`raw_results.csv`) + Task 6 (`tpot_vs_batchsize.csv`). New uv venv → Task 1. Detailed README → Task 9. Dense baseline (user choice) → harness `--attention dense`, driver runs both. bf16 + OOM-skip (user choice) → `kv_cache_dtype="auto"` + `_is_oom` handling. Disable KV-cache reuse (user requirement) → `disable_radix_cache=True` in `build_server_args`. Qwen3-8B substitution (user choice) → Task 2 + harness default `--model-path`.
- **Naming consistency.** `RAW_CSV_FIELDS` (harness) and `RAW_FIELDS` (aggregator test) list identical columns; `aggregate_results.py` reads them via `csv.DictReader` (order-independent). `summarize_step_latencies` is defined once in `benchmark_quest_tpot.py` and imported by `aggregate_results.py` — single source of truth.
- **Known open risk.** Task 5 is the gate for whether `bench_one_batch` carries vortex sparsity; if it fails, pivot to the `sgl.Engine` delta method (Risks R2) before Task 7.
