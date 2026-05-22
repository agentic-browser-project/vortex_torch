# Quest Batch Benchmark on vortex_torch v0.5 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Quest sparse-attention decode-TPOT batch benchmark on vortex_torch **v0.5**, on the new branch `quest-batch-benchmark-v0.5`, and run it on **Qwen3-VL-8B-Instruct** (the sgl baseline's exact model) by patching v0.5's vortex attention backend so it accepts that multimodal architecture.

**Architecture:** The benchmark keeps the same shape as the existing `quest-batch-benchmark` branch — one in-process `sgl.Engine` per attention mode (`dense`/`quest`), streaming `engine.generate`, `TPOT = (end − first_token) / (tokens − 1)`. v0.5 changes three things the harness must adapt to: (1) the engine kwargs use v0.5's vortex API (`vortex_attention_backend`, `kv_cache_dtype`, `vortex_compilation_cache_dir`, the built-in registered flow `gqa_quest_sparse_attention`); (2) v0.5 is pure-Python + Triton JIT (no `vortex_torch_C` CUDA extension to build) and vendors sglang **v0.5.9**; (3) v0.5's vortex attention backend hard-asserts `not is_multimodal`, which must be patched to run Qwen3-VL-8B.

**Tech Stack:** Python 3.12, uv venv, vortex_torch v0.5 (Triton-JIT sparse attention), sglang v0.5.9 (`third_party/sglang/v0.5.9/sglang`), torch 2.7.1+cu128, flashinfer 0.6.8.post1, NVIDIA B200 (sm_100). Model: `Qwen3-VL-8B-Instruct`.

---

## Starting State (already done — do NOT redo)

- The git worktree **already exists** at
  `/vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5`
  on branch `quest-batch-benchmark-v0.5`, tracking `origin/v0.5` (HEAD `9b5c3ae`).
- `.worktrees/` is git-ignored via `.git/info/exclude`.
- **All paths in this plan are relative to that worktree root** unless absolute.
- The previous (v0.3-based) benchmark lives on branch `quest-batch-benchmark`; this
  plan ports files from it via `git checkout quest-batch-benchmark -- <path>`.

## Background & Key Findings (from v0.5 exploration — the executor has zero context)

1. **sglang version.** v0.5 vendors two sglang trees; `roadmap.md` / `solution.md` state
   v0.5 uses **`third_party/sglang/v0.5.9/sglang`**. Only the v0.5.9 tree has
   `python/sglang/srt/models/qwen3_vl.py` (`EntryClass = Qwen3VLForConditionalGeneration`).
   The README's "support 0.4.9" line is stale (it documents the `v1` branch).

2. **Engine API.** v0.5 still drives sglang **in-process** via `sgl.Engine(**kwargs)`
   (see `examples/run_ruler.py`, `examples/run_lcb.py`). The old harness's
   one-engine-per-mode + streaming-generate design ports directly.

3. **Vortex engine kwargs (v0.5).** From `examples/run_ruler.py` and
   `vortex_torch/engine/sgl/api.py:get_engine`, the vortex-relevant `sgl.Engine`
   kwargs are: `enable_vortex_sparsity`, `vortex_module_name`,
   `vortex_attention_backend` (`"flashinfer"` | `"trtllm"`, **new in v0.5**),
   `vortex_topk_val`, `vortex_topk_ratio`, `vortex_block_size`,
   `vortex_block_reserved_bos`/`_eos`, `vortex_workload_chunk_size`,
   `vortex_layers_skip`, `vortex_schedule_policy`, `vortex_dtype`,
   `vortex_max_seq_lens`, `vortex_compilation_cache_dir`, `kv_cache_dtype`,
   plus `page_size` (must be a multiple of `vortex_block_size`).
   `from vortex_torch.engine.sgl import DEFAULT_SCHEDULE_POLICY` **still works**
   (`vortex_torch/engine/sgl/__init__.py` re-exports it).

4. **Quest flow is built-in.** `vortex_torch/flow/algorithms.py` registers
   `@register("gqa_quest_sparse_attention")`. `run_ruler.py` passes a built-in
   `vortex_module_name` with **no** `vortex_module_path` — so registered flows
   resolve by name alone. (`check_engine_config` requires a `vortex_module_path`,
   but the harness uses `sgl.Engine` directly, not `check_engine_config`.)

5. **`disable_overlap_schedule=True` is still required** for the vortex backend
   in v0.5 (`get_engine` hardcodes it; `roadmap.md` documents the open
   overlap-schedule bug). Same rationale as the v0.3 branch.

6. **The multimodal guard.** `vortex_torch/engine/sgl/attention_backend/flashinfer.py`
   line 87 has `assert not self.is_multimodal` (and `trtllm.py` has the identical
   line). Qwen3-VL-8B-Instruct (`architectures: ["Qwen3VLForConditionalGeneration"]`)
   is multimodal → the assertion blocks it. **Task 4 patches this.** sglang's
   `get_hf_text_config()` already unwraps Qwen3-VL's nested `text_config`, so
   `model_config.num_attention_heads` / `head_dim` / `get_num_kv_heads()` (all the
   shape attrs the backend reads) are already correct for Qwen3-VL's text decoder.
   The `is_encoder_decoder` assertion is left in place; Qwen3-VL is a plain
   decoder, not encoder-decoder.

7. **Build.** v0.5 deleted `csrc/` and `setup.py`; it is pure-Python + Triton JIT
   (`pyproject.toml`, `version = "0.5.0"`). `pip install -e .` for vortex_torch
   needs no nvcc build. sglang v0.5.9 installs via its own `install.sh`
   (torch 2.7.1+cu128, flashinfer 0.6.8.post1).

8. **No conda here.** The v0.5 `.claude/CLAUDE.md` / `roadmap.md` mention conda
   envs (`vortex_v04`, `vortex_v1`) — those are not present on this host
   (`conda` is not installed). The benchmark uses a **fresh uv venv**, exactly
   as the v0.3 branch did.

9. **Models on disk.** `/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct/`
   (headline model) and `/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B/`
   (text-twin fallback) both exist. GPU: one NVIDIA B200, 183 GB, currently idle.

## File Structure

| Path (worktree-relative) | Responsibility | Action |
|---|---|---|
| `quest_batch_benchmark/setup_env.sh` | Build the uv venv (v0.5.9 sglang + vortex v0.5 stack) | Rewrite |
| `quest_batch_benchmark/benchmark_quest_tpot.py` | The `sgl.Engine` TPOT harness | Port + adapt `build_engine_kwargs` |
| `quest_batch_benchmark/prompt_io.py` | Render `request.json` → chat-templated prompt | Port unchanged |
| `quest_batch_benchmark/aggregate_results.py` | Collapse raw CSV → per-config TPOT CSV | Port unchanged |
| `quest_batch_benchmark/request.json` | The 9.6k-token benchmark request | Port unchanged |
| `quest_batch_benchmark/run_benchmark.sh` | Drive dense+quest+aggregate | Port + adapt paths |
| `quest_batch_benchmark/pytest.ini`, `.gitignore` | Test config / ignores | Port unchanged |
| `quest_batch_benchmark/tests/test_prompt_io.py` | prompt_io tests | Port unchanged |
| `quest_batch_benchmark/tests/test_aggregate.py` | aggregator tests | Port unchanged |
| `quest_batch_benchmark/tests/test_harness.py` | harness-function tests | Port + adapt for v0.5 kwargs |
| `quest_batch_benchmark/README.md`, `PROGRESS.md` | Methodology + handoff docs | Rewrite for v0.5 |
| `quest_batch_benchmark/results/*.csv` | Benchmark output | Regenerate |
| `vortex_torch/engine/sgl/attention_backend/flashinfer.py` | Vortex flashinfer backend | Patch (line 87) |
| `vortex_torch/engine/sgl/attention_backend/trtllm.py` | Vortex trtllm backend | Patch (identical line) |

---

## Task 1: Build the v0.5 benchmark environment

**Files:**
- Create: `quest_batch_benchmark/setup_env.sh`

- [ ] **Step 1: Create the benchmark directory and write `setup_env.sh`**

```bash
mkdir -p /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5/quest_batch_benchmark
```

Write `quest_batch_benchmark/setup_env.sh`:

```bash
#!/usr/bin/env bash
# Create the uv venv for the Quest batch benchmark on vortex_torch v0.5.
# v0.5 is pure-Python + Triton JIT (no vortex_torch_C CUDA extension to build).
# Serving backend is the vendored sglang v0.5.9. Target GPU: NVIDIA B200 (sm_100).
# Re-runnable (clears and recreates the venv if it already exists).
set -euo pipefail

REPO=/vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
BENCH="$REPO/quest_batch_benchmark"
VENV="$BENCH/.venv"
SGLANG="$REPO/third_party/sglang/v0.5.9/sglang"
TORCH_INDEX=https://download.pytorch.org/whl/cu128

command -v uv >/dev/null || { echo "ERROR: 'uv' is not on PATH." >&2; exit 1; }
[ -d "$SGLANG/python" ] || { echo "ERROR: sglang v0.5.9 not found at $SGLANG" >&2; exit 1; }

echo "[1/5] create uv venv (python 3.12)"
uv venv --python 3.12 "$VENV" 2>/dev/null || uv venv --python 3.12 --clear "$VENV"

echo "[2/5] install torch 2.7.1+cu128 (ships sm_100 cubins for B200)"
uv pip install --python "$VENV" --index-url "$TORCH_INDEX" \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

echo "[3/5] install sglang v0.5.9 (editable) + flashinfer"
uv pip install --python "$VENV" flashinfer-python==0.6.8.post1
uv pip install --python "$VENV" -e "$SGLANG/python[all]"
uv pip install --python "$VENV" pytest PyYAML==6.0.3 uvloop==0.21.0 uvicorn==0.35.0

echo "[4/5] install vortex_torch v0.5 (editable, pure python) + harness deps"
uv pip install --python "$VENV" --no-deps -e "$REPO"
uv pip install --python "$VENV" transformers==4.57.1 accelerate einops ninja pandas

echo "[5/5] verify the stack imports and sees the B200"
"$VENV/bin/python" - <<'PY'
import torch, sglang, vortex_torch
print("torch  ", torch.__version__, "cuda_ok", torch.cuda.is_available())
print("device ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("sglang ", getattr(sglang, "__version__", "?"))
print("vortex_torch import OK")
PY
echo "[done] venv at $VENV"
```

- [ ] **Step 2: Run `setup_env.sh`**

Run: `bash quest_batch_benchmark/setup_env.sh`
Expected: ends with `device  NVIDIA B200`, `sglang  0.5.9` (or `0.5.9.x`),
`vortex_torch import OK`, `[done] venv at ...`.

**Iteration note (env setup is inherently iterative):** if a transitive pin
downgrades `torch` off `2.7.1+cu128`, re-pin torch and add `--no-deps` to the
offending install line. If `transformers` version conflicts between sglang's
`[all]` extra and vortex_torch's `4.57.1` pin, install `transformers==4.57.1`
last and confirm `import sglang` + `import vortex_torch` still succeed. Do not
proceed to Step 3 until the Step 5 verification block prints all three OK lines.

- [ ] **Step 3: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add quest_batch_benchmark/setup_env.sh
git commit -m "quest-batch-benchmark-v0.5: uv venv setup for the v0.5 stack

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Port the mode-agnostic harness files

These files are engine-API-agnostic and port **unchanged** from the
`quest-batch-benchmark` branch. `git checkout <branch> -- <path>` copies the
file into the worktree and stages it.

**Files:**
- Create (port unchanged): `quest_batch_benchmark/prompt_io.py`,
  `aggregate_results.py`, `request.json`, `pytest.ini`, `.gitignore`,
  `tests/test_prompt_io.py`, `tests/test_aggregate.py`
- Create (port, modified in Tasks 3): `quest_batch_benchmark/benchmark_quest_tpot.py`,
  `tests/test_harness.py`, `run_benchmark.sh`

- [ ] **Step 1: Port every benchmark file from the v0.3 branch**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
for f in prompt_io.py aggregate_results.py request.json pytest.ini .gitignore \
         benchmark_quest_tpot.py run_benchmark.sh \
         tests/test_prompt_io.py tests/test_aggregate.py tests/test_harness.py; do
  git checkout quest-batch-benchmark -- "quest_batch_benchmark/$f"
done
ls -la quest_batch_benchmark quest_batch_benchmark/tests
```

Expected: all 10 files present under `quest_batch_benchmark/`.

- [ ] **Step 2: Run the ported unit tests that need no changes**

Run: `quest_batch_benchmark/.venv/bin/python -m pytest quest_batch_benchmark/tests/test_prompt_io.py quest_batch_benchmark/tests/test_aggregate.py -q`
Expected: PASS (these modules are unchanged from v0.3).

`test_harness.py` will FAIL here — that is expected; Task 3 updates it.

- [ ] **Step 3: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add quest_batch_benchmark/
git commit -m "quest-batch-benchmark-v0.5: port the benchmark harness from the v0.3 branch

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Adapt `build_engine_kwargs` for the v0.5 vortex API (TDD)

The v0.3 `build_engine_kwargs` produced engine kwargs where dense had **no**
vortex keys and quest used the old API. v0.5 needs: `page_size` and
`kv_cache_dtype` for both modes, an explicit `enable_vortex_sparsity=False` for
dense, and `vortex_attention_backend` for quest.

**Files:**
- Modify: `quest_batch_benchmark/tests/test_harness.py` (the `build_engine_kwargs` tests)
- Modify: `quest_batch_benchmark/benchmark_quest_tpot.py`
  (`build_engine_kwargs`, lines 71-130; and the `--model-path` default, line 347)

- [ ] **Step 1: Rewrite the `build_engine_kwargs` tests for v0.5**

In `quest_batch_benchmark/tests/test_harness.py`, replace the four tests in the
`--- build_engine_kwargs ---` section (`test_dense_kwargs_match_baseline`,
`test_quest_kwargs_add_vortex`, `test_enable_cuda_graph_flag`,
`test_mem_fraction_static_passed_when_set`) with:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `quest_batch_benchmark/.venv/bin/python -m pytest quest_batch_benchmark/tests/test_harness.py -q`
Expected: FAIL — `test_dense_kwargs_match_baseline` / `test_quest_kwargs_add_vortex`
fail on the new `page_size` / `kv_cache_dtype` / `enable_vortex_sparsity` /
`vortex_attention_backend` assertions (the ported v0.3 code does not emit them).

- [ ] **Step 3: Rewrite `build_engine_kwargs` in `benchmark_quest_tpot.py`**

Replace the entire `build_engine_kwargs` function (lines 71-130) with:

```python
def build_engine_kwargs(args, n_input_tokens: int) -> dict:
    """sgl.Engine kwargs for one attention mode on vortex_torch v0.5.

    Mirrors the `tpot-no-share` sgl baseline (CUDA graph off, radix cache off,
    flashinfer backend, debug logging) and disables chunked prefill (offline
    batch inference prefills each request whole). `quest` additionally enables
    v0.5's vortex sparsity with the built-in `gqa_quest_sparse_attention` flow.

    v0.5 notes: `page_size` must be a multiple of `vortex_block_size` (the
    vortex backend asserts this), so it is set to 16 for both modes; the
    vortex backend is present in-process either way once `vortex_torch` is
    imported. `enable_vortex_sparsity=False` makes `dense` full attention.
    """
    # vortex buffers are sized for the input + the decode tokens + headroom
    max_seq = max(args.max_seq_lens, n_input_tokens + args.max_tokens + 64)

    # Disable chunked prefill -- offline batch inference prefills each request
    # whole, never split or co-packed. sglang's -1 "disable" sentinel is
    # rejected by the vortex page-size assertion (chunked_prefill_size %
    # page_size == 0), so set the budget to one whole request rounded up to
    # the 16-token page plus a one-page margin (>= one request, < two).
    chunked_prefill_size = ((n_input_tokens + 15) // 16) * 16 + 16

    kwargs = {
        "model_path": args.model_path,
        "tp_size": 1,
        "trust_remote_code": True,
        "attention_backend": "flashinfer",
        "page_size": 16,
        "kv_cache_dtype": "auto",
        "disable_cuda_graph": not args.enable_cuda_graph,
        "disable_radix_cache": True,
        # The vortex backend reuses shared metadata buffers across forwards;
        # sglang's overlap scheduler runs forwards in a separate thread and
        # races them. Disabling the overlap schedule serializes scheduling and
        # the forward, closing the race. Required for the vortex backend in
        # v0.5 (vortex_torch/engine/sgl/api.py hardcodes the same).
        "disable_overlap_schedule": True,
        "chunked_prefill_size": chunked_prefill_size,
        "decode_log_interval": 1,
        "show_time_cost": True,
        "log_level": "debug",
    }
    if args.mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = args.mem_fraction_static

    if args.attention == "quest":
        kwargs.update({
            "enable_vortex_sparsity": True,
            "vortex_module_name": QUEST_MODULE,
            "vortex_attention_backend": "flashinfer",
            "vortex_topk_val": args.topk_val,
            "vortex_topk_ratio": 0.0,            # pure static block budget
            "vortex_block_size": 16,
            "vortex_block_reserved_bos": 1,
            "vortex_block_reserved_eos": 2,
            "vortex_workload_chunk_size": 32,
            "vortex_layers_skip": [0],
            "vortex_schedule_policy": DEFAULT_SCHEDULE_POLICY,
            "vortex_dtype": "bfloat16",
            "vortex_max_seq_lens": max_seq,
            "vortex_compilation_cache_dir": args.vortex_cache_dir,
        })
    else:
        # dense -- full attention; sparsity explicitly off
        kwargs["enable_vortex_sparsity"] = False
    return kwargs
```

- [ ] **Step 4: Point the `--model-path` default at Qwen3-VL-8B-Instruct**

In `build_parser()`, change the `--model-path` argument's default. Replace:

```python
    p.add_argument("--model-path",
                   default="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B")
```

with:

```python
    p.add_argument("--model-path",
                   default="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct",
                   help="Headline model is Qwen3-VL-8B-Instruct (the sgl "
                        "baseline's model). Fallback: .../Qwen/Qwen3-8B if "
                        "Qwen3-VL cannot run -- see README.md.")
```

- [ ] **Step 5: Run the harness tests to verify they pass**

Run: `quest_batch_benchmark/.venv/bin/python -m pytest quest_batch_benchmark/tests/ -q`
Expected: PASS — all tests in `test_harness.py`, `test_aggregate.py`,
`test_prompt_io.py` green.

- [ ] **Step 6: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add quest_batch_benchmark/benchmark_quest_tpot.py quest_batch_benchmark/tests/test_harness.py
git commit -m "quest-batch-benchmark-v0.5: adapt engine kwargs to the v0.5 vortex API

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Patch the vortex attention backend's multimodal guard

v0.5's vortex flashinfer/trtllm backends hard-assert `not is_multimodal`, which
blocks Qwen3-VL-8B-Instruct. Qwen3-VL is a plain decoder with a vision encoder
feeding pre-computed embeddings — not encoder-decoder — and sglang's
`get_hf_text_config()` already unwraps its `text_config` so every shape the
backend reads is correct. Remove the `is_multimodal` assertion (keep the
`is_encoder_decoder` assertion).

**Files:**
- Modify: `vortex_torch/engine/sgl/attention_backend/flashinfer.py:87`
- Modify: `vortex_torch/engine/sgl/attention_backend/trtllm.py` (identical line)

- [ ] **Step 1: Confirm the guard text in both backends**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
grep -n "is_multimodal\|is_encoder_decoder\|skip_prefill" \
  vortex_torch/engine/sgl/attention_backend/flashinfer.py \
  vortex_torch/engine/sgl/attention_backend/trtllm.py
```
Expected: each file shows `self.is_multimodal = model_runner.model_config.is_multimodal`
and a later `assert not self.is_multimodal` line. Confirm both files contain
the same assertion block before editing.

- [ ] **Step 2: Patch `flashinfer.py`**

In `vortex_torch/engine/sgl/attention_backend/flashinfer.py`, replace:

```python
        assert model_runner.sliding_window_size is None
        assert not model_runner.model_config.is_encoder_decoder 
        assert not self.skip_prefill
        assert not self.is_multimodal
```

with:

```python
        assert model_runner.sliding_window_size is None
        assert not model_runner.model_config.is_encoder_decoder
        assert not self.skip_prefill
        # `is_multimodal` is NOT asserted off: Qwen3-VL-8B is a multimodal
        # model but a plain decoder (no cross-attention -- is_encoder_decoder
        # is still asserted above). sglang's get_hf_text_config() unwraps its
        # nested text_config, so num_attention_heads / head_dim / kv-heads are
        # already correct for the text decoder. The vortex sparse path only
        # touches decoder attention, so a text-only prompt runs unchanged.
```

- [ ] **Step 3: Patch `trtllm.py` identically**

Apply the exact same replacement (drop `assert not self.is_multimodal`, add the
same explanatory comment) in
`vortex_torch/engine/sgl/attention_backend/trtllm.py`. The benchmark uses the
flashinfer backend, but keeping the two backends consistent avoids a surprise
if `vortex_attention_backend="trtllm"` is ever used.

- [ ] **Step 4: Confirm no other multimodal guard remains on the decode path**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
grep -rn "is_multimodal\|assert.*multimodal" vortex_torch/engine/
```
Expected: only the two `self.is_multimodal = ...` assignment lines remain (the
attribute may still be read elsewhere); **no surviving `assert not ...multimodal`**.
If any other multimodal `assert` exists on the engine/attention path, note it —
Task 5's smoke run is the empirical gate that catches anything missed.

- [ ] **Step 5: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add vortex_torch/engine/sgl/attention_backend/flashinfer.py \
        vortex_torch/engine/sgl/attention_backend/trtllm.py
git commit -m "vortex backend: allow multimodal decoder models (Qwen3-VL)

Drop the conservative \`assert not is_multimodal\` guard in the flashinfer
and trtllm attention backends. Qwen3-VL is a plain decoder; the vortex
sparse path only touches decoder attention. is_encoder_decoder is still
asserted off.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Smoke test — boot Qwen3-VL-8B through the harness (dense + quest)

This is the **gate** that proves Task 4's patch works end-to-end. It runs the
real harness at tiny scale on the B200.

**Files:** none modified — this task only runs commands.

- [ ] **Step 1: Smoke-run dense**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5/quest_batch_benchmark
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark_quest_tpot.py \
  --attention dense --batch-sizes 1,2 --max-tokens 16 --repeat 1 \
  --raw-csv /tmp/smoke_dense.csv 2>&1 | tail -30
```
Expected: no `is_multimodal` AssertionError; lines `[run] repeat=0 status=ok
tpot=... ms`; ends `[done] raw rows written to /tmp/smoke_dense.csv`.

- [ ] **Step 2: Smoke-run quest**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5/quest_batch_benchmark
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark_quest_tpot.py \
  --attention quest --batch-sizes 1,2 --max-tokens 16 --repeat 1 \
  --raw-csv /tmp/smoke_quest.csv 2>&1 | tail -40
```
Expected: the vortex quest flow JIT-compiles on first use (Triton compilation
log lines), then `status=ok` rows; ends `[done]`.

**Decision gate.** If both smoke runs reach `[done]` with `status=ok` rows →
proceed to Task 6 on Qwen3-VL-8B. If quest still fails on a multimodal/vortex
assertion that cannot be resolved by a small follow-up patch, invoke the
**documented fallback**: re-run both smoke commands with
`--model-path /vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B`; if
those pass, the benchmark proceeds on Qwen3-8B and the README records that
Qwen3-VL-8B could not run. Do not silently continue past a failing gate —
report which model the benchmark will use.

- [ ] **Step 3: Verify generated text is coherent (not garbage)**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5/quest_batch_benchmark
CUDA_VISIBLE_DEVICES=0 .venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
import vortex_torch  # noqa: F401
import sglang as sgl
from benchmark_quest_tpot import build_engine_kwargs
from types import SimpleNamespace
a = SimpleNamespace(attention="quest", max_tokens=32, topk_val=64,
                    max_seq_lens=16384, enable_cuda_graph=False,
                    mem_fraction_static=0.8, vortex_cache_dir="/tmp/vcache",
                    model_path="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct")
e = sgl.Engine(**build_engine_kwargs(a, 64))
out = e.generate(["The capital of France is"],
                 {"max_new_tokens": 24, "temperature": 0.0})
print("OUTPUT:", out[0]["text"] if isinstance(out, list) else out)
e.shutdown()
PY
```
Expected: the printed `OUTPUT:` text is coherent English mentioning Paris — this
confirms the patched quest path produces correct decode output, not just
non-crashing output. (If fallback to Qwen3-8B was taken in Step 2, use that
`model_path` here instead.)

- [ ] **Step 4: Clean up smoke artifacts (no commit — nothing changed)**

```bash
rm -f /tmp/smoke_dense.csv /tmp/smoke_quest.csv
```

---

## Task 6: Run the full benchmark and aggregate results

Run dense and quest sweeps over batch sizes 1/2/4/8/16/32/64, 256 output tokens,
`repeat=3`, then aggregate. The runs are GPU-bound and long (≈30–60 min each,
plus first-use JIT compilation) — run each mode as its own command so neither
exceeds a background-task lifetime limit.

**Files:**
- Modify: `quest_batch_benchmark/run_benchmark.sh` (path + GPU adjustments)
- Create: `quest_batch_benchmark/results/raw_results.csv`,
  `quest_batch_benchmark/results/tpot_vs_batchsize.csv`

- [ ] **Step 1: Adjust `run_benchmark.sh` for the worktree**

Confirm `run_benchmark.sh` (ported in Task 2) computes `BENCH` from its own
location and uses `$BENCH/.venv/bin/python` — it does, so it needs no path
edits. Confirm the default `GPU` is `0` (the only B200). No commit needed if
the file is already correct; if any path was branch-specific, fix it and note
it in the Task 7 commit.

- [ ] **Step 2: Run the dense sweep**

Run (foreground or background; if background, it re-invokes on completion):
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5/quest_batch_benchmark
mkdir -p results logs
rm -f results/raw_results.csv
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark_quest_tpot.py \
  --attention dense --raw-csv results/raw_results.csv \
  2>&1 | tee logs/dense_$(date +%Y%m%d_%H%M%S).log
```
Expected: 7 batch sizes × 3 repeats = 21 `status=ok` rows appended; ends `[done]`.

- [ ] **Step 3: Run the quest sweep**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5/quest_batch_benchmark
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark_quest_tpot.py \
  --attention quest --raw-csv results/raw_results.csv \
  2>&1 | tee logs/quest_$(date +%Y%m%d_%H%M%S).log
```
Expected: 21 more rows appended (the harness appends per mode); ends `[done]`.

- [ ] **Step 4: Aggregate**

Run:
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5/quest_batch_benchmark
.venv/bin/python aggregate_results.py \
  --raw-csv results/raw_results.csv --out-csv results/tpot_vs_batchsize.csv
cat results/tpot_vs_batchsize.csv
```
Expected: 14 config rows (7 dense + 7 quest), each `status=ok` (or `capped` if a
batch's KV footprint exceeds the pool). Print the dense-vs-quest TPOT table.

- [ ] **Step 5: Commit results**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add -f quest_batch_benchmark/results/raw_results.csv \
           quest_batch_benchmark/results/tpot_vs_batchsize.csv
git commit -m "quest-batch-benchmark-v0.5: engine-based TPOT results on Qwen3-VL-8B

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

(`results/` is git-ignored by `quest_batch_benchmark/.gitignore`; `-f` force-adds
the two CSVs deliberately, matching the v0.3 branch.)

---

## Task 7: Documentation

**Files:**
- Rewrite: `quest_batch_benchmark/README.md`, `quest_batch_benchmark/PROGRESS.md`
- Create: `quest_batch_benchmark/reference_freeze.txt`

- [ ] **Step 1: Freeze the environment**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5/quest_batch_benchmark
.venv/bin/python -m pip freeze > reference_freeze.txt
```

- [ ] **Step 2: Rewrite `README.md`**

`quest_batch_benchmark/README.md` must document, for an engineer with zero
context: the methodology (one `sgl.Engine` per mode, streaming generate,
`TPOT = (end − first_token)/(tokens − 1)`); the v0.5 stack (sglang v0.5.9,
vortex_torch v0.5, torch 2.7.1+cu128, B200); the **deviations from the sgl
baseline** (model = Qwen3-VL-8B-Instruct — note this now *matches* the baseline,
unlike the v0.3 branch which used Qwen3-8B; chunked prefill disabled; overlap
schedule disabled); the **`is_multimodal` guard patch** (Task 4 — what was
changed, why it is safe, which files); the results table; and how to re-run
(`setup_env.sh`, then `run_benchmark.sh`). If Task 5 fell back to Qwen3-8B,
README must say so plainly and explain why Qwen3-VL could not run.

- [ ] **Step 3: Rewrite `PROGRESS.md`**

`quest_batch_benchmark/PROGRESS.md` is the handoff doc: status, branch
(`quest-batch-benchmark-v0.5`), the result table, the v0.5-specific notes
(JIT compilation, the multimodal patch, sglang v0.5.9), and how to resume.

- [ ] **Step 4: Run the full test suite once more**

Run: `quest_batch_benchmark/.venv/bin/python -m pytest quest_batch_benchmark/tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
git add quest_batch_benchmark/README.md quest_batch_benchmark/PROGRESS.md \
        quest_batch_benchmark/reference_freeze.txt quest_batch_benchmark/run_benchmark.sh
git commit -m "quest-batch-benchmark-v0.5: document the v0.5 methodology and results

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Whole-implementation review and branch finish

- [ ] **Step 1: Request a code review**

Use the superpowers:requesting-code-review skill (or dispatch the
superpowers:code-reviewer agent) against the full diff
`git diff v0.5..quest-batch-benchmark-v0.5`. Review focus: the
`build_engine_kwargs` v0.5 adaptation, the `is_multimodal` patch (is dropping
the assertion actually safe — any decode-path code that assumed text-only?),
the aggregator/error handling, and whether README's deviations are accurate.

- [ ] **Step 2: Fix any issues the review raises**

Address findings; commit fixes with descriptive messages.

- [ ] **Step 3: Finish the branch**

Use the superpowers:finishing-a-development-branch skill to present merge / PR /
keep-as-is options. The v0.3 branch was kept as-is (not merged); default to the
same unless the user chooses otherwise.

---

## Self-Review (plan author's checklist — completed)

- **Spec coverage:** new branch ✔ (worktree, Task 1-8 on `quest-batch-benchmark-v0.5`);
  "same benchmark on v0.5" ✔ (Tasks 1-3, 6 — ported harness, v0.5 engine kwargs);
  "try to run qwen3 vl 8B" ✔ (Task 4 patch + Task 5 gate + Task 6 runs on it).
- **Placeholder scan:** no TBD/"handle errors"/"similar to" — every code step
  shows complete code or exact commands.
- **Type consistency:** `build_engine_kwargs(args, n_input_tokens)` signature is
  unchanged from the ported file; `QUEST_MODULE` / `DEFAULT_SCHEDULE_POLICY` are
  module-level names already present in the ported `benchmark_quest_tpot.py`.
- **Known iteration points (flagged, not placeholders):** Task 1 env setup may
  need a torch/transformers pin tweak (Step 2 note); Task 5 is an explicit
  decision gate with a documented Qwen3-8B fallback.
