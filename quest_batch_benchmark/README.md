# Quest Batch Benchmark

Decode-speed (TPOT) benchmark of **Quest** sparse attention vs a dense-attention
baseline, across decode batch sizes **1, 2, 4, 8, 16, 32, 64**, on the
`vortex_torch` framework.

## What this measures

**Quest** ([arXiv:2406.10774](https://arxiv.org/abs/2406.10774)) is a
query-aware sparse-attention method: at each decode step it scores KV blocks
with a cheap query–envelope product and attends only to the top-`k` blocks.
Here it is the `gqa_quest_sparse_attention` flow registered in
`vortex_torch/flow/algorithms.py`.

**TPOT** (time-per-output-token) is the decode-step latency. The harness uses
`sglang.bench_one_batch` to run a *fixed* decode batch size: it replicates one
request `batch_size` times, prefills, then times each decode step with CUDA
events. A decode step emits exactly one token per request, so the step latency
*is* TPOT. Reported TPOT is the mean over 256 measured steps (after 16 warmup
steps). The CUDA-event interval includes host-side launch overhead, but that
overhead is identical for the quest and dense modes, so the relative
comparison — the benchmark's deliverable — is unaffected.

This is a **tight decode-loop microbenchmark**: `bench_one_batch` calls the
model's `decode()` directly, so the measured TPOT is decode-kernel time plus
CUDA launch overhead only. It does *not* include the full serving stack
(request scheduler, tokenizer/detokenizer, streaming/IPC) that an end-to-end
sglang *server* TPOT measurement would — a server-path TPOT for the same model
is typically several times larger. The output length (number of decode steps)
has no material effect on TPOT: over 256 steps the context grows only
9,661 → 9,917 tokens (~2.6%).

## Setup substitutions and deviations (read this)

- **Model.** The intended `Qwen/Qwen3-VL-8B-Instruct` cannot load: the patched
  sglang 0.4.9 fork that `vortex_torch` depends on has no `qwen3_vl` model.
  The benchmark therefore uses **`Qwen/Qwen3-8B`** — the text twin of
  Qwen3-VL-8B's backbone (identical 36 layers / 32 heads / 8 KV heads /
  head_dim 128). The request is text-only and decode TPOT is governed entirely
  by the text transformer, so the numbers transfer faithfully.
- **Input.** `request.json` is a copy of `request_005_20260316_221014` — a
  system + user chat message pair that tokenizes to **9,661 tokens** after the
  Qwen3 chat template.
- **GPU stack (B200).** The benchmark runs on an NVIDIA **B200** (sm_100). The
  framework's original "known-good" stack (torch 2.7.1+cu126, sgl-kernel 0.2.4)
  is **Hopper-only** — sgl-kernel 0.2.x ships no sm_100 kernels. The stack was
  bumped to **torch 2.8.0+cu128** and **sgl-kernel 0.3.17.post1** (the first
  series with a Blackwell build), and `vortex_torch_C` was rebuilt for sm_100.
  `setup_env.sh` reproduces this.
- **Fairness.** The engine is booted with `disable_radix_cache=True` so no KV
  prefix is reused; quest and dense are measured on equal footing.
- **KV cache** stays bf16. If a batch size OOMs it is recorded as `status=oom`
  and the ascending sweep stops.

## Reproduce

Requires 1× NVIDIA B200 (sm_100) — e.g. any `dgx-b200` node on this cluster.

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
| `setup_env.sh` | Creates `.venv` (uv) and installs torch / sglang fork / vortex_torch for the B200. |
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
(256), `--max-seq-lens` (16384), `--mem-fraction-static` (0.6 — see note below),
`--disable-cuda-graph` (run decode eager instead of with a CUDA graph).

`--mem-fraction-static` defaults to **0.6**, not the sglang default 0.9: at 0.9
the static KV pool is allocated ~144 GB (far more than this 9.6K-token workload
needs), leaving too little memory for large-batch prefill activations, and a
batch ≥ 16 prefill OOMs. 0.6 keeps enough KV pool for batch 64 while freeing
room for the prefill.

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

Mean decode TPOT (ms/token) on the B200, `request_005` (9,661 input tokens),
Quest `topk_val=64`:

| batch size | dense TPOT | quest TPOT | quest speedup |
|-----------:|-----------:|-----------:|--------------:|
| 1  | 5.64  | 5.53 | 1.02× |
| 2  | 6.01  | 5.72 | 1.05× |
| 4  | 6.88  | 6.08 | 1.13× |
| 8  | 8.76  | 6.44 | 1.36× |
| 16 | 12.03 | 6.93 | 1.74× |
| 32 | 19.27 | 8.53 | 2.26× |
| 64 | OOM   | OOM  | —     |

**Quest decode TPOT scales far better with batch size than dense attention.**
At batch 1 the two are within ~2% (the query–envelope scoring overhead roughly
cancels the savings at tiny batch). As batch size grows, dense TPOT rises
steeply — each query attends over the full ~600 KV blocks of the 9.6K-token
context — while Quest stays nearly flat because each query reads only the
top-64 blocks. By batch 32, Quest is **2.26× faster** (8.53 ms vs 19.27 ms).
Both modes OOM at batch 64 (the prefill of 64 × 9,661 tokens exceeds B200
memory); this is recorded as `status=oom` and the sweep stops.
