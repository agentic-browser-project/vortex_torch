# Sparse-Attention Batch Benchmark — v0.5 (dense vs Quest vs TreeSparse)

Decode-speed (TPOT) benchmark comparing **dense**, **Quest**, and
**TreeSparseAttention** sparse attention across decode batch sizes
**1, 2, 4, 8, 16, 32, 64**, measured on the same 9,661-token text prompt and
256 output tokens. Dense and Quest run through a full **`sgl.Engine`** so those
numbers are directly comparable to the project's sgl baseline
`run_batch_experiments_offline.sh tpot-no-share`. TreeSparse runs through its
own HuggingFace + FlashInfer harness and is merged into the same table for
comparison.

This is the **v0.5 successor** to the `quest-batch-benchmark` branch (which ran
on vortex v0.3 + sglang 0.4.x + Qwen3-8B). The v0.5 branch runs on
**vortex_torch v0.5** (pure-Python + Triton JIT, no CUDA extension) +
**sglang v0.5.9** and — importantly — uses **Qwen3-VL-8B-Instruct**, the exact
same model as the sgl baseline. The v0.3 branch had to fall back to Qwen3-8B
because the old sglang fork lacked a `qwen3_vl` model class; that workaround is
gone in v0.5.

**Second model — Qwen3-8B.** This benchmark is also run on **Qwen3-8B**
(`/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B`), the v0.3
branch's text-only model, as a second benchmarked model. Its results live
under `results/qwen3_8b/` and are reported in their own section
([Qwen3-8B (second model)](#results--qwen3-8b-second-model)) below. The
Qwen3-8B run covers **dense + Quest (`topk_val=64` and `topk_val=29`)** in
both the no-graph and CUDA-graph categories. **TreeSparse is omitted for
Qwen3-8B** — its harness hardcodes the Qwen3-VL model class and cannot load a
text-only Qwen3-8B checkpoint correctly; see
[Why TreeSparse is omitted for Qwen3-8B](#why-treesparse-is-omitted-for-qwen3-8b).

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

## Model — Qwen3-VL-8B-Instruct

### Vortex attention backend patch — enabling Qwen3-VL

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

The engine is constructed via Quest's official wrapper
`vortex_torch.engine.sgl.get_engine`, which folds in vortex defaults and
ends in `sgl.Engine(**kwargs)`. `--engine-api direct` is an opt-in alternative
that builds the kwargs locally and calls `sgl.Engine(...)` itself with
identical fairness flags; the two paths are independently sanity-checked to
produce equivalent TPOT (see "Engine-API comparison" below).

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

## CUDA graph (second category)

The benchmark is run in **two categories**: a *no-graph* category (the
default, matching the `tpot-no-share` baseline's `disable_cuda_graph=True`),
and a *CUDA-graph* category (`--enable-cuda-graph`, which sets
`disable_cuda_graph=False`). Both share every other engine flag — sampling,
chunked-prefill sizing, overlap-schedule disabling, debug logging — so the
two-category comparison isolates the effect of the CUDA-graph capture alone.

### Coverage

| method | no-graph | CUDA graph |
|---|:---:|:---:|
| dense (sgl.Engine + flashinfer) | yes | yes |
| quest, topk_val=64 (sgl.Engine + vortex sparsity) | yes | yes |
| quest, topk_val=29 (sgl.Engine + vortex sparsity) | yes | yes |
| TreeSparseAttention (own HF+FlashInfer harness) | yes | **no** |

TreeSparseAttention is excluded from the CUDA-graph category because its
decode harness is structurally graph-incompatible (per-step
`torch.cuda.synchronize()` for timing, per-step `sampled.tolist()` host
copy in the sampling path, a per-step Python loop over layers in
`decode_step`, and a hardcoded `is_cuda_graph_enabled=False` argument in
its FlashInfer plan call). See [`cuda_graph_status.md`](cuda_graph_status.md)
for code citations and the full discussion.

## Results — CUDA graph off (no-graph baseline, 4 methods)

The headline category, matching the `tpot-no-share` sgl baseline
(`disable_cuda_graph=True`). All four methods run on the same
`Qwen3-VL-8B-Instruct` model, the same `request.json` (9,661-token
text prompt), 256 output tokens, `repeat=3`, on the same B200. Numbers
are the mean of three warm repeats per configuration. The `quest (topk=64)`
column is the headline Quest configuration (1024 tokens kept per
query, the value used in the original v0.5 benchmark); `quest (topk=29)`
is Quest at the **vortex_torch `get_engine` default** (`vortex_topk_val=29`,
~464 tokens kept) — what an out-of-the-box user sees if they call the
wrapper without overriding the topk.

| batch size | dense TPOT (ms) | quest (topk=64) TPOT (ms) | quest (topk=29) TPOT (ms) | TreeSparse TPOT (ms) |
|-----------:|----------------:|--------------------------:|--------------------------:|---------------------:|
|  1 |  9.14 | 11.23 | 11.27 | 10.90 |
|  2 | 10.53 | 13.09 | 13.06 | 13.98 |
|  4 | 12.11 | 14.88 | 14.87 | 14.36 |
|  8 | 15.54 | 18.30 | 18.31 | 15.52 |
| 16 | 21.91 | 25.27 | 25.38 | 16.64 |
| 32 | 37.42 | 38.62 | 38.63 | 18.56 |
| 64 | 71.88 | 65.08 | 65.01 | 24.86 |

All 28 configurations (4 methods × 7 batch sizes) completed `status=ok`.

### Fairness contract

**Same input:** all four methods run on `request.json` (the established
9,661-token text-only prompt). `run_treesparse.sh` overrides TreeSparse's
hardcoded WebVoyager request with this file. TreeSparse's tokenizer
produced an identical 9,661-token prefill, so the input axis matches
exactly.

**Same output and sweep:** 256 decode tokens. Batch sizes 1–64, `repeat=3`.
Identical on both harnesses.

**TPOT — fair representative value:** quest/dense report a *warm* mean-of-3
(an untimed warmup runs before each batch). TreeSparse's harness
(`benchmark_batch.py`) runs **no** separate warmup, so its first repetition
absorbs one-time FlashInfer JIT / cold-cache cost and inflates its
`tpot_mean_ms`. The merge therefore uses TreeSparse's
**`tpot_median_ms`** (median of 3 — discards the single cold rep), which
is the apples-to-apples match for quest's warm mean.

**Documented unavoidable differences** (different methods require
different engines — cannot be unified): quest/dense run through Quest's
`vortex_torch.engine.sgl.get_engine` wrapper around `sgl.Engine` (torch
2.9.1, streaming wall-clock TPOT); TreeSparse runs through its own
HuggingFace + FlashInfer harness (torch 2.11.0, per-decode-step
`cuda.synchronize()` timing). Both measure mean decode-step latency
excluding the first token. Sparsity operating points also differ by
design: **quest is measured at two operating points** — `topk_val=64`
(1024 tokens kept, the original headline value) and `topk_val=29` (~464
tokens kept, the vortex_torch `get_engine` default) — and TreeSparse at
`top-k=128` chunks (its own `run_batch_experiments.sh tpot-no-share`
default). These are each method's intended setting — recorded
transparently, not forced equal.

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
> sglang at all — it is a different engine end-to-end. The four-way table
> is produced by merging two independent measurements, not by one harness.


**Note on Triton JIT compilation:** vortex v0.5 is pure-Python + Triton
(no CUDA extension). The first warmup generate per batch triggers Triton
kernel compilation; a subsequent identical run will be fast because
compiled kernels are cached. If re-running from a cold start, expect the
first batch to be slow while kernels compile.

## Results — CUDA graph on

Same fairness contract as the no-graph category above (same model, same
9,661-token input, 256 output tokens, `repeat=3`, same `get_engine`
wrapper), but with `disable_cuda_graph=False`. TreeSparseAttention is
**not** included in this category because its decode harness is
structurally graph-incompatible (per-step `torch.cuda.synchronize()` for
timing, per-step `sampled.tolist()` host copy in the sampling path, a
per-step Python loop over layers in `decode_step`, and a hardcoded
`is_cuda_graph_enabled=False` argument in its FlashInfer plan call). See
[`cuda_graph_status.md`](cuda_graph_status.md) for code citations.

| batch size | dense TPOT (ms) | quest (topk=64) TPOT (ms) | quest (topk=29) TPOT (ms) |
|-----------:|----------------:|--------------------------:|--------------------------:|
|  1 |  5.60 |  5.92 |  5.87 |
|  2 |  6.61 |  6.93 |  6.82 |
|  4 |  8.71 |  8.96 |  8.85 |
|  8 | 12.85 | 12.63 | 12.54 |
| 16 | 20.71 | 19.92 | 19.71 |
| 32 | 36.71 | 34.50 | 34.28 |
| 64 | 71.15 | 64.03 | 62.91 |

All 21 configurations completed with `status=ok`.


## Results — Qwen3-8B (second model)

The same benchmark run on **Qwen3-8B** (`/vast/.../Qwen/Qwen3-8B`), same
`request.json` (9,661-token text prompt), 256 output tokens, `repeat=3`, same
B200, same `get_engine` wrapper and the same fairness contract as the headline
Qwen3-VL run. The only difference is the model. This section covers **dense +
Quest** at two operating points — `topk_val=64` (1024 tokens kept, the headline
Quest configuration) and `topk_val=29` (~464 tokens kept, the vortex_torch
`get_engine` default) — in both the no-graph and CUDA-graph categories.
TreeSparse is **not** part of the Qwen3-8B comparison (see
[Why TreeSparse is omitted for Qwen3-8B](#why-treesparse-is-omitted-for-qwen3-8b)).

### Coverage

| method | no-graph | CUDA graph |
|---|:---:|:---:|
| dense (sgl.Engine + flashinfer) | yes | yes |
| quest, topk_val=64 (sgl.Engine + vortex sparsity) | yes | yes |
| quest, topk_val=29 (sgl.Engine + vortex sparsity) | yes | yes |
| TreeSparseAttention | **no** | **no** |

### CUDA graph off (no-graph)

| batch size | dense TPOT (ms) | quest (topk=64) TPOT (ms) | quest (topk=64) speedup | quest (topk=29) TPOT (ms) | quest (topk=29) speedup |
|-----------:|----------------:|--------------------------:|------------------------:|--------------------------:|------------------------:|
|  1 |  8.19 | 10.63 | 0.77x | 10.78 | 0.76x |
|  2 |  9.23 | 12.65 | 0.73x | 12.96 | 0.71x |
|  4 | 10.61 | 14.38 | 0.74x | 14.60 | 0.73x |
|  8 | 12.98 | 17.81 | 0.73x | 17.70 | 0.73x |
| 16 | 18.83 | 24.62 | 0.76x | 24.50 | 0.77x |
| 32 | 31.80 | 37.65 | 0.84x | 37.39 | 0.85x |
| 64 | 62.00 | 63.75 | 0.97x | 63.71 | 0.97x |

All 21 configurations (3 methods × 7 batch sizes) completed `status=ok`. As on
Qwen3-VL, Quest's fixed per-step block-scoring overhead dominates at small batch
(dense is fastest there — quest@64 is 0.77× dense at bs=1), and the gap closes
as the batch grows until quest nearly reaches dense at bs=64 (0.97×, 63.75 ms vs
62.00 ms). The two Quest operating points are within noise of each other at
every batch size — at this prompt length the tighter `topk_val=29` budget does
not yet visibly out-run `topk_val=64`.

### CUDA graph on

Same fairness contract, with `disable_cuda_graph=False`.

| batch size | dense TPOT (ms) | quest (topk=64) TPOT (ms) | quest (topk=64) speedup | quest (topk=29) TPOT (ms) | quest (topk=29) speedup |
|-----------:|----------------:|--------------------------:|------------------------:|--------------------------:|------------------------:|
|  1 |  5.52 |  5.74 | 0.96x |  5.67 | 0.97x |
|  2 |  6.37 |  6.89 | 0.92x |  6.68 | 0.95x |
|  4 |  8.13 |  8.82 | 0.92x |  8.69 | 0.93x |
|  8 | 11.54 | 12.54 | 0.92x | 12.42 | 0.93x |
| 16 | 17.91 | 19.62 | 0.91x | 19.40 | 0.92x |
| 32 | 31.18 | 33.83 | 0.92x | 33.41 | 0.93x |
| 64 | 61.21 | 62.81 | 0.97x | 61.87 | 0.99x |

All 21 configurations completed with `status=ok`. CUDA-graph capture compresses
the per-step overhead, so the dense/quest gap is much tighter than no-graph
(quest@64 stays within 0.91–0.97× dense across the whole sweep), and quest@29
reaches 0.99× dense at bs=64.

### CUDA-graph vs no-graph speedup (same method) — Qwen3-8B

Side-by-side per-method speedup from CUDA-graph capture (speedup > 1 means CUDA
graph is faster; `abs diff = CUDA-graph − no-graph`, so a negative number also
means CUDA graph is faster). CUDA-graph capture helps Quest more than dense at
small batch (quest@29 bs=1: 1.90× vs dense 1.48×), because Quest's larger
per-step Python/scheduling overhead is exactly what graph capture removes.

| attention | batch size | no-graph TPOT (ms) | CUDA-graph TPOT (ms) | abs diff (ms) | speedup (no-graph / CUDA-graph) |
|---|---:|---:|---:|---:|---:|
| dense | 1 | 8.186 | 5.519 | -2.667 | 1.483247 |
| dense | 2 | 9.233 | 6.371 | -2.862 | 1.449203 |
| dense | 4 | 10.614 | 8.129 | -2.486 | 1.305789 |
| dense | 8 | 12.977 | 11.537 | -1.440 | 1.124829 |
| dense | 16 | 18.834 | 17.909 | -0.925 | 1.051627 |
| dense | 32 | 31.800 | 31.176 | -0.624 | 1.020008 |
| dense | 64 | 62.001 | 61.206 | -0.795 | 1.012992 |
| quest | 1 | 10.633 | 5.743 | -4.890 | 1.851577 |
| quest | 2 | 12.650 | 6.890 | -5.760 | 1.835984 |
| quest | 4 | 14.378 | 8.824 | -5.554 | 1.629359 |
| quest | 8 | 17.812 | 12.539 | -5.273 | 1.420558 |
| quest | 16 | 24.622 | 19.617 | -5.005 | 1.255125 |
| quest | 32 | 37.653 | 33.826 | -3.827 | 1.113137 |
| quest | 64 | 63.748 | 62.810 | -0.938 | 1.014933 |
| quest_topk29 | 1 | 10.783 | 5.674 | -5.108 | 1.900273 |
| quest_topk29 | 2 | 12.960 | 6.681 | -6.279 | 1.939721 |
| quest_topk29 | 4 | 14.600 | 8.694 | -5.907 | 1.679415 |
| quest_topk29 | 8 | 17.704 | 12.421 | -5.283 | 1.425298 |
| quest_topk29 | 16 | 24.502 | 19.402 | -5.100 | 1.262853 |
| quest_topk29 | 32 | 37.387 | 33.415 | -3.972 | 1.118881 |
| quest_topk29 | 64 | 63.706 | 61.869 | -1.838 | 1.029703 |

Display values for the result tables above are 2-decimal-place rounded; the
appendix `abs diff` / `speedup` columns were computed from the underlying
full-precision aggregates.

### Why TreeSparse is omitted for Qwen3-8B

The headline Qwen3-VL comparison includes TreeSparseAttention, but the Qwen3-8B
comparison does **not**. TreeSparse's harness (`benchmark_batch.py`) loads every
model through a `Qwen3VLInference` class that tries `Qwen3VLForConditionalGeneration`
first. That loader does not *error* on a text-only Qwen3-8B checkpoint — it
silently builds the **Qwen3-VL** architecture and treats the real checkpoint as
incompatible: every Qwen3-8B weight loads as "unexpected" (discarded) and every
Qwen3-VL weight is "missing" and **newly (randomly) initialized**. The resulting
in-memory model is a *random-weight Qwen3-VL text tower* (32 layers, 32 KV heads,
rope_theta 5e5), not the real Qwen3-8B (36 layers, 8 KV heads / GQA, rope_theta
1e6). Any TPOT it produced would be timing the wrong architecture with random
weights, so it is not a meaningful comparison point and is excluded. (The
headline Qwen3-VL run is unaffected: there the real Qwen3-VL checkpoint matches
the class the loader picks.) Making TreeSparse benchmark a real text-only
Qwen3-8B would require a loader change in the TreeSparse project itself (branch
on `config.model_type` to load `Qwen3ForCausalLM`), which is out of scope here.
The `run_benchmark_qwen3_8b.sh` wrapper therefore sets `RUN_TREESPARSE=0`,
skipping the TreeSparse stage and the four-way merge.


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

# 3. run the full benchmark (dense + quest + treesparse, then aggregate)
GPU=0 bash quest_batch_benchmark/run_benchmark.sh
```

To run the **Qwen3-8B** model instead (writes into `results/qwen3_8b/`; the
model is already on disk from the v0.3 work, otherwise
`hf download Qwen/Qwen3-8B --local-dir .../Qwen/Qwen3-8B`):

```bash
# no-graph sweep (dense + quest@64 + quest@29) on Qwen3-8B
GPU=0 bash quest_batch_benchmark/run_benchmark_qwen3_8b.sh

# then the CUDA-graph sweep (dense + quest@64 + quest@29) on Qwen3-8B
GPU=0 bash quest_batch_benchmark/run_benchmark_cudagraph_qwen3_8b.sh
```

Both Qwen3-8B drivers are thin wrappers over the headline drivers — they set
`MODEL_PATH` + `RESULTS_DIR=results/qwen3_8b` and reuse every fairness flag
unchanged. The no-graph wrapper also sets `RUN_TREESPARSE=0`, skipping the
TreeSparse stage (see
[Why TreeSparse is omitted for Qwen3-8B](#why-treesparse-is-omitted-for-qwen3-8b)).

`run_benchmark.sh` now runs three stages: `dense`, `quest`, and `treesparse`.
The `treesparse` stage is driven by `run_treesparse.sh`, which runs TreeSparse
in its own pre-built environment (see `treesparse_env_notes.md`). A separate
environment build is **not** required — TreeSparse's environment is already
built. After all three stages complete, `build_comparison.py` merges the
results into `results/tpot_three_way.csv` and `results/comparison_table.md`.

The CUDA-graph (second category) sweep is driven separately:

```bash
# 4. (optional) the CUDA-graph category for dense + quest, plus the side-by-side
GPU=0 bash quest_batch_benchmark/run_benchmark_cudagraph.sh
```

This requires the no-graph sweep (step 3) to have completed first, because
the comparison merges against the no-graph aggregate. Outputs land in
`results/raw_results_cudagraph.csv`, `tpot_vs_batchsize_cudagraph.csv`, and
`cuda_graph_comparison.{csv,md}`.

The dense and quest modes can also be run separately:

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

## Engine-API comparison (sanity check)

The benchmark routes through Quest's official wrapper
`vortex_torch.engine.sgl.get_engine`. The wrapper accepts `**kwargs` and
applies them after its own defaults, so every fairness-relevant flag the
baseline sets (`disable_cuda_graph=True`, `disable_radix_cache=True`,
`chunked_prefill_size=...`, debug logging) flows through unchanged. The
opt-in alternative `--engine-api direct` builds the same kwargs locally and
calls `sgl.Engine(**kwargs)` itself — useful as an independent sanity check
that the wrapper isn't perturbing measurements.

The driver `run_engine_api_comparison.sh` runs the full dense+quest sweep
through both paths with matched fairness flags and writes a side-by-side
table at `results/engine_api_comparison.md`. Across all 14 configurations
the two paths agree within 1.5% TPOT (within 0.5% on the Quest path Quest's
wrapper is designed for, modulo a single bs=4 noise spike at 2.6%); the
dense arm under `get_engine` runs ~1-1.5% slower than under `direct` at
small batch sizes (shrinking to ~0.1% at bs=64), plausibly because the
wrapper passes its full set of `vortex_*` kwargs through to `sgl.Engine`
even when sparsity is off. The bias does not affect any dense-vs-quest
comparison conclusion — both numbers are produced by the same constructor.

```bash
GPU=0 bash quest_batch_benchmark/run_engine_api_comparison.sh
```

## CUDA-graph vs no-graph speedup (same method) — appendix

Side-by-side per-method speedup from CUDA-graph capture, derived from
`results/cuda_graph_comparison.md` (speedup > 1 means CUDA graph is
faster than no-graph at that point; `abs diff = CUDA-graph − no-graph`,
so a negative number also means CUDA graph is faster):

| attention | batch size | no-graph TPOT (ms) | CUDA-graph TPOT (ms) | abs diff (ms) | speedup (no-graph / CUDA-graph) |
|---|---:|---:|---:|---:|---:|
| dense | 1 | 9.140 | 5.596 | -3.544 | 1.633244 |
| dense | 2 | 10.527 | 6.614 | -3.913 | 1.591615 |
| dense | 4 | 12.111 | 8.711 | -3.400 | 1.390327 |
| dense | 8 | 15.537 | 12.850 | -2.687 | 1.209096 |
| dense | 16 | 21.906 | 20.713 | -1.192 | 1.057556 |
| dense | 32 | 37.417 | 36.706 | -0.711 | 1.019359 |
| dense | 64 | 71.884 | 71.152 | -0.731 | 1.010278 |
| quest | 1 | 11.227 | 5.918 | -5.309 | 1.897063 |
| quest | 2 | 13.086 | 6.928 | -6.157 | 1.888705 |
| quest | 4 | 14.875 | 8.964 | -5.912 | 1.659488 |
| quest | 8 | 18.304 | 12.632 | -5.672 | 1.448999 |
| quest | 16 | 25.274 | 19.916 | -5.358 | 1.269026 |
| quest | 32 | 38.622 | 34.503 | -4.119 | 1.119389 |
| quest | 64 | 65.083 | 64.034 | -1.049 | 1.016378 |
| quest_topk29 | 1 | 11.267 | 5.867 | -5.400 | 1.920445 |
| quest_topk29 | 2 | 13.055 | 6.821 | -6.235 | 1.914114 |
| quest_topk29 | 4 | 14.872 | 8.852 | -6.020 | 1.680163 |
| quest_topk29 | 8 | 18.313 | 12.536 | -5.777 | 1.460863 |
| quest_topk29 | 16 | 25.378 | 19.712 | -5.666 | 1.287424 |
| quest_topk29 | 32 | 38.628 | 34.279 | -4.349 | 1.126883 |
| quest_topk29 | 64 | 65.007 | 62.905 | -2.102 | 1.033420 |

Display values are 3-decimal-place rounded; `abs diff` and `speedup`
were computed from the underlying full-precision aggregates (re-deriving
them from the rounded display columns can give slightly different last-digit
values).

## Files

| File | Purpose |
|------|---------|
| `setup_env.sh` | Creates `.venv` (uv) and installs torch / sglang v0.5.9 / vortex_torch for the B200. |
| `benchmark_quest_tpot.py` | The Engine harness — one attention mode, all batch sizes. |
| `aggregate_results.py` | Collapses the raw per-repeat CSV into the processed curve. |
| `run_benchmark.sh` | Driver: runs dense, quest, and treesparse stages, then builds the three-way comparison. |
| `run_benchmark_qwen3_8b.sh` | Thin wrapper: runs the no-graph dense+quest sweep on **Qwen3-8B** into `results/qwen3_8b/` (sets `MODEL_PATH`/`RESULTS_DIR`/`RUN_TREESPARSE=0` and execs `run_benchmark.sh`; TreeSparse is skipped — see the Qwen3-8B section). |
| `run_benchmark_cudagraph_qwen3_8b.sh` | Thin wrapper: runs the CUDA-graph sweep on **Qwen3-8B** into `results/qwen3_8b/`. |
| `results/qwen3_8b/` | All Qwen3-8B result files (same filenames as `results/` for the headline model). |
| `run_treesparse.sh` | Orchestrator: drives TreeSparse's own harness on `request.json` and writes `results/treesparse_raw.json`. |
| `treesparse_results.py` | Converts `treesparse_raw.json` into a comparison row (uses `tpot_median_ms`). |
| `build_comparison.py` | Merges dense/quest aggregated CSV with TreeSparse row into `results/tpot_three_way.csv` and `results/comparison_table.md`. |
| `treesparse_env_notes.md` | Notes on TreeSparse's pre-built Python 3.13 environment and CUDA kernels. |
| `prompt_io.py` | Renders `request.json` into a chat-templated prompt string. |
| `request.json` | The benchmark input request (9,661 tokens after Qwen3 chat template). |
| `reference_freeze.txt` | `uv pip freeze` snapshot of the exact environment used. |
| `results/raw_results.csv` | Per-(mode, batch size, repeat) measurement (dense + quest). |
| `results/tpot_vs_batchsize.csv` | Processed TPOT vs batch size (dense + quest). |
| `results/treesparse_raw.json` | Raw per-repeat output from TreeSparse's `benchmark_batch.py`. |
| `results/tpot_three_way.csv` | Merged three-way TPOT table (dense, quest, treesparse). |
| `results/comparison_table.md` | Rendered markdown of the three-way comparison table. |
| `compare_engine_apis.py` | Joins two per-API aggregated CSVs into the engine-API comparison table. |
| `run_engine_api_comparison.sh` | Driver: runs the dense+quest sweep via both `--engine-api` paths, aggregates each, then builds the comparison. |
| `results/raw_results_{direct,get_engine}.csv` | Per-repeat measurements from the engine-API comparison sweep. |
| `results/tpot_vs_batchsize_{direct,get_engine}.csv` | Per-API aggregated TPOT vs batch size. |
| `results/engine_api_comparison.{csv,md}` | Side-by-side engine-API TPOT comparison table. |
| `tests/` | Unit tests (45 tests). |
| `cuda_graph_status.md` | Per-method CUDA-graph capability note (dense yes, quest yes-by-smoke, TreeSparse no with code citations). |
| `compare_cuda_graph.py` | Joins the no-graph and CUDA-graph aggregated CSVs into the graph-vs-no-graph comparison table. |
| `run_benchmark_cudagraph.sh` | Driver: runs dense+quest with `--enable-cuda-graph`, aggregates, then builds the graph-vs-no-graph comparison. |
| `results/raw_results_cudagraph.csv` | Per-(mode, batch size, repeat) measurements from the CUDA-graph sweep. |
| `results/tpot_vs_batchsize_cudagraph.csv` | Aggregated TPOT vs batch size from the CUDA-graph sweep. |
| `results/cuda_graph_comparison.{csv,md}` | Side-by-side CUDA-graph vs no-graph TPOT comparison table. |

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
