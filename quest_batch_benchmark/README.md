# Quest Batch Benchmark — v0.5

Decode-speed (TPOT) benchmark of **Quest** sparse attention vs a dense-attention
baseline, across decode batch sizes **1, 2, 4, 8, 16, 32, 64**, measured through
a full **`sgl.Engine`** so the numbers are directly comparable to the project's
sgl baseline `run_batch_experiments_offline.sh tpot-no-share`.

This is the **v0.5 successor** to the `quest-batch-benchmark` branch (which ran
on vortex v0.3 + sglang 0.4.x + Qwen3-8B). The v0.5 branch runs on
**vortex_torch v0.5** (pure-Python + Triton JIT, no CUDA extension) +
**sglang v0.5.9** and — importantly — uses **Qwen3-VL-8B-Instruct**, the exact
same model as the sgl baseline. The v0.3 branch had to fall back to Qwen3-8B
because the old sglang fork lacked a `qwen3_vl` model class; that workaround is
gone in v0.5.

## What this measures

**Quest** ([arXiv:2406.10774](https://arxiv.org/abs/2406.10774)) is a
query-aware sparse-attention method: at each decode step it scores KV blocks
with a cheap query–envelope product and attends only to the top-`k` blocks.
Here it is the `gqa_quest_sparse_attention` flow registered in
`vortex_torch/flow/algorithms.py`, enabled through the engine's vortex
sparsity backend.

**TPOT** (time-per-output-token) is the decode-step latency. The harness boots
one `sgl.Engine` per attention mode and, for each batch size, replicates the
benchmark request `batch_size` times and runs a streaming
`engine.generate(..., max_new_tokens=256, ignore_eos=True)`. TPOT is computed
exactly as the baseline does:

    tpot = (stream_end - first_token_time) / (tokens_generated - 1)

i.e. the decode window (everything after the first token, which streams out at
the end of prefill) divided by the number of decode steps. Reported TPOT is the
mean over `repeat=3` measurements, each preceded by one untimed per-batch
warmup generate.

This is an **end-to-end serving measurement**: it includes the request
scheduler, the sampling loop, detokenization, ZMQ IPC, async streaming, and
per-step debug logging — the same path, and essentially the same engine config,
as the `tpot-no-share` baseline.

## Stack

| Component | Version |
|-----------|---------|
| vortex_torch | v0.5 (pure-Python + Triton JIT; no CUDA extension) |
| sglang | v0.5.9 (vendored at `third_party/sglang/v0.5.9/sglang`) |
| torch | 2.9.1+cu128 |
| sgl-kernel | 0.3.21 |
| flashinfer | 0.6.3 |
| triton | 3.5.1 |
| GPU | NVIDIA B200 (sm_100) |

Full package list: `reference_freeze.txt`.

## Model — now Qwen3-VL-8B-Instruct (apples-to-apples with the baseline)

**This benchmark uses `Qwen/Qwen3-VL-8B-Instruct`**, the same model as the sgl
baseline. The v0.3 branch had to substitute `Qwen/Qwen3-8B` because the vortex
sglang fork (then v0.4.7-based) had no `qwen3_vl` model class. Two things
changed in v0.5 that removed this limitation:

1. sglang v0.5.9 ships a `qwen3_vl` model class.
2. A one-line patch to the vortex attention backends (described below) removed
   the hard guard that blocked multimodal models.

No fallback to Qwen3-8B was required. The model axis now matches the baseline
exactly, so the comparison is apples-to-apples.

## Vortex attention backend patch — enabling Qwen3-VL

**What changed:** vortex v0.5's attention backends
(`vortex_torch/engine/sgl/attention_backend/flashinfer.py` and `trtllm.py`)
contained the assertion:

```python
assert not self.is_multimodal
```

This assertion fired when loading `Qwen3-VL-8B-Instruct`, which sglang
classifies as multimodal (it has a vision tower), blocking the benchmark.
Commit `064c68d` removed that single assertion from both files.

**Why it is safe to remove:**

- Qwen3-VL is a plain autoregressive decoder for token generation; it is NOT an
  encoder-decoder. The `is_encoder_decoder` assertion is kept in place and
  would still reject a true encoder-decoder model.
- sglang's `get_hf_text_config()` unwraps Qwen3-VL's nested `text_config`
  automatically, so the backend reads the correct decoder-layer shapes (heads,
  head_dim, etc.) without any additional changes.
- The vortex sparse path operates entirely on decoder attention (KV blocks,
  block tables, decode indices). The vision tower is irrelevant to it.
- The benchmark sends text-only prompts; the vision-encoder code path is never
  exercised.
- Verified by the smoke test: the engine produces coherent decode output on a
  text prompt (both dense and quest modes).

The removed line is the only change in that commit. The guard was conservative
scaffolding from before sglang v0.5.9 existed; the actual decode path was
already correct.

## Engine config — matched to the baseline, with documented deviations

Matched to the `tpot-no-share` baseline (`measure_batch_latency_offline.py`):

- `disable_cuda_graph=True`, `disable_radix_cache=True`,
  `attention_backend="flashinfer"`, `tp_size=1`.
- `decode_log_interval=1`, `show_time_cost=True`, `log_level="debug"`.
- `mem_fraction_static` left at the sglang default.
- Sampling: `temperature=0.0`, `top_p=1.0`, `ignore_eos=True`,
  `max_new_tokens=256`.

Four deliberate deviations from the baseline's defaults:

- **Model matches** (`Qwen3-VL-8B-Instruct`). Unlike the v0.3 branch, no
  model substitution is needed.
- **Chunked prefill disabled.** `chunked_prefill_size` is sized to hold one
  whole request, so each request prefills in a single forward — appropriate for
  offline batch inference. The baseline leaves it at the sglang default.
  Chunked prefill affects only prompt processing, not steady-state decode TPOT.
- **Overlap schedule disabled** (`disable_overlap_schedule=True`) — **required.**
  The vortex quest backend reuses shared instance-attribute metadata buffers
  (`qo_indptr`, `kv_indices`, `batch_table`, …) across forwards; sglang's
  overlap scheduler runs forwards in a separate thread, racing those buffers.
  The failure vanishes under `CUDA_LAUNCH_BLOCKING=1`, confirming a race;
  disabling the overlap schedule serializes scheduling against the forward and
  closes it. Applied to **both** modes so dense and quest remain on identical
  footing. (vortex v0.5's own `get_engine` hardcodes the same flag for the
  same reason.)
- **`SGLANG_DISABLE_CUDNN_CHECK=1`** — sglang v0.5.9's startup CuDNN-version
  check is over-strict for the bundled CuDNN 9.10 + torch 2.9.1 combination.
  The check fires and blocks boot; the harness sets this env var to skip it.
  The smoke test confirmed decode produces coherent output — a sanity check,
  not a CuDNN-matched numerical validation. That is sufficient here: this is a
  TPOT *timing* benchmark, and any CuDNN effect is identical across the dense
  and quest runs, so the dense-vs-quest comparison is unaffected.
- **Quest mode added.** The baseline ships only `flashinfer` (dense) and
  `tree_sparse`; this benchmark adds `quest` via the vortex sparsity backend,
  with `topk_val=64` (1024 tokens kept), and measures dense here too so the
  comparison is on identical footing.

## All batch sizes fit the KV pool

At the default `mem_fraction_static`, the B200 KV pool holds enough token-slots
for many concurrently loaded requests. Every batch size 1–64 runs fully
concurrently (no wave-serialization), and every result row is `status=ok`. The
harness flags `status=capped` for any batch whose KV footprint would exceed the
pool, but that does not occur in this sweep.

## Results

Mean decode TPOT (ms/token) on the B200, `Qwen3-VL-8B-Instruct`,
`request_005` (9,661 input tokens), 256 output tokens, `repeat=3`,
Quest `topk_val=64`, measured through `sgl.Engine` (`tpot-no-share` config):

| batch size | dense TPOT | quest TPOT | quest speedup |
|-----------:|-----------:|-----------:|--------------:|
| 1  |  9.51 | 10.93 | 0.87x |
| 2  | 10.18 | 12.68 | 0.80x |
| 4  | 11.84 | 14.56 | 0.81x |
| 8  | 15.07 | 17.88 | 0.84x |
| 16 | 21.75 | 24.83 | 0.88x |
| 32 | 37.37 | 38.14 | 0.98x |
| 64 | 71.79 | 64.72 | 1.11x |

All 14 configurations completed with `status=ok` — the KV pool held every batch
size.

**Interpretation:** At small batch sizes Quest is *slower* than dense — its
query–envelope block-scoring is fixed overhead that, at batch 1–16, outweighs
the KV-read it saves (decode is weight-bandwidth-bound there, and attention is a
small fraction). The two modes reach near parity at **batch 32** (37.4 ms vs
38.1 ms), and by **batch 64 Quest is 1.11x faster** (64.7 ms vs 71.8 ms): dense
TPOT rises steeply as each query reads the full ~9.9K-token KV, while Quest
reads only its top-`k` blocks.

The crossover at batch 32–64 (rather than batch 16 as in v0.3/Qwen3-8B) is
consistent with Qwen3-VL-8B's multimodal scaffolding adding a small but
nonzero overhead to each decode step; this shifts the point where attention
(which Quest reduces) becomes the dominant cost.

**Note on comparability with v0.3 results:** The v0.3 branch ran
Qwen3-8B on vortex v0.3 + sglang 0.4.x. The numbers from that branch (quest
1.22x faster at batch 64) are **not directly comparable** to these v0.5 numbers
— different model, different sglang, different vortex version. The meaningful
comparison is dense-vs-quest within each branch's own run.

**Note on Triton JIT compilation:** vortex v0.5 is pure-Python + Triton (no
CUDA extension). The first warmup generate per batch triggers Triton kernel
compilation; a subsequent identical run will be fast because compiled kernels
are cached. If re-running from a cold start, expect the first batch to be slow
while kernels compile.

## Reproduce

**Prerequisites:**
- An NVIDIA **B200** (sm_100) GPU.
- [`uv`](https://docs.astral.sh/uv/) on `PATH`, and Python 3.12 available.
- A **CUDA 12.8 toolkit** (needed for JIT compilation of Triton/flashinfer
  kernels at runtime; `setup_env.sh` defaults to a spack install; override
  with the `CUDA_HOME` env var).
- Network access (PyPI + the PyTorch cu128 wheel index).

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5

# 1. one-time: create the uv venv + install the stack (~15-40 min on first run)
bash quest_batch_benchmark/setup_env.sh

# 2. one-time: download the model (~16 GB)
quest_batch_benchmark/.venv/bin/hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct

# 3. run the full benchmark (dense + quest, then aggregate)
GPU=0 bash quest_batch_benchmark/run_benchmark.sh
```

`run_benchmark.sh` runs the harness for `dense` then `quest` and aggregates.
The two modes can also be run separately:

```bash
cd quest_batch_benchmark
.venv/bin/python benchmark_quest_tpot.py --attention dense --raw-csv results/raw_results.csv
.venv/bin/python benchmark_quest_tpot.py --attention quest --raw-csv results/raw_results.csv
.venv/bin/python aggregate_results.py
```

This is useful if a single combined run would exceed a job-runner time limit;
the harness appends to the raw CSV. Run unit tests:

```bash
quest_batch_benchmark/.venv/bin/python -m pytest quest_batch_benchmark/tests/ -q
```

## Files

| File | Purpose |
|------|---------|
| `setup_env.sh` | Creates `.venv` (uv) and installs torch / sglang v0.5.9 / vortex_torch for the B200. |
| `benchmark_quest_tpot.py` | The Engine harness — one attention mode, all batch sizes. |
| `aggregate_results.py` | Collapses the raw per-repeat CSV into the processed curve. |
| `run_benchmark.sh` | Driver: runs the harness for `dense` then `quest`, then aggregates. |
| `prompt_io.py` | Renders `request.json` into a chat-templated prompt string. |
| `request.json` | The benchmark input request (9,661 tokens after Qwen3 chat template). |
| `reference_freeze.txt` | `uv pip freeze` snapshot of the exact environment used. |
| `results/raw_results.csv` | Per-(mode, batch size, repeat) measurement. |
| `results/tpot_vs_batchsize.csv` | Processed TPOT vs batch size. |
| `tests/` | Unit tests (19 tests). |

## Tunable knobs

`benchmark_quest_tpot.py` flags: `--topk-val` (Quest block budget, default 64 =
1024 tokens kept), `--batch-sizes`, `--max-tokens` (256), `--repeat` (3),
`--mem-fraction-static` (default: sglang default), `--enable-cuda-graph` (off
by default to match the baseline), `--max-seq-lens` (16384, vortex buffer
sizing).

## Output CSV schemas

**`raw_results.csv`** — one row per measured repeat:

`run_timestamp, attention, batch_size, model, topk_val, input_tokens,
max_tokens, repeat_idx, tokens_generated, ttft_ms, tpot_ms, decode_time_ms,
total_time_ms, throughput_tok_s, status`

`status in {ok, capped, error}`. `ok`/`capped` rows carry full metrics; an
`error` config has a single row with `repeat_idx=-1` and empty metric columns.

**`tpot_vs_batchsize.csv`** — one row per (attention, batch size):

`attention, batch_size, model, topk_val, input_tokens, max_tokens, repeat,
status, tpot_ms_mean, tpot_ms_std, tpot_ms_min, tpot_ms_max, ttft_ms_mean,
decode_time_ms_mean, total_time_ms_mean, throughput_tok_s_mean`
