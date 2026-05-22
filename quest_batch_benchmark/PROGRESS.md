# Quest Batch Benchmark — Progress / Handoff

**Status:** COMPLETE — harness rebuilt for vortex v0.5 + sglang v0.5.9,
multimodal guard patched, full benchmark run, all 14 configs status=ok.
**Last updated:** 2026-05-21.
**Branch:** `quest-batch-benchmark-v0.5`
**Worktree:** `/vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5`

The benchmark measures decode TPOT through a full `sgl.Engine` in streaming
mode, with the engine config matched to the sgl baseline
`run_batch_experiments_offline.sh tpot-no-share`
(`measure_batch_latency_offline.py`), so the numbers are directly comparable.
This is the v0.5 successor to the `quest-batch-benchmark` branch, which ran on
vortex v0.3 + sglang 0.4.x + Qwen3-8B (the old fork lacked a `qwen3_vl` model).
v0.5 runs on `Qwen3-VL-8B-Instruct` — the exact model the sgl baseline uses.

## Result

`results/tpot_vs_batchsize.csv` — mean decode TPOT (ms/token),
`Qwen3-VL-8B-Instruct`, `request_005` (9,661 input tokens), 256 output tokens,
`repeat=3`, Quest `topk_val=64`, `sgl.Engine` (`tpot-no-share` config):

| batch | dense (ms/tok) | quest (ms/tok) | speedup |
|------:|---------------:|---------------:|--------:|
| 1  |  9.51 | 10.93 | 0.87x |
| 2  | 10.18 | 12.68 | 0.80x |
| 4  | 11.84 | 14.56 | 0.81x |
| 8  | 15.07 | 17.88 | 0.84x |
| 16 | 21.75 | 24.83 | 0.88x |
| 32 | 37.37 | 38.14 | 0.98x |
| 64 | 71.79 | 64.72 | 1.11x |

Quest is slower than dense at small batch (query-envelope scoring overhead),
reaches near-parity at batch 32, and is 1.11x faster at batch 64. All 14
configs completed `status=ok` (the KV pool holds every batch — no capping).

Note: these numbers are not directly comparable to the v0.3 branch's
Qwen3-8B results. The meaningful comparison is dense-vs-quest within each run.

## Branch commits

| Commit | Description |
|--------|-------------|
| `2035d41` | `setup_env.sh` — build venv for vortex v0.5 / sglang v0.5.9 / B200 |
| `82c0551` | Harness port — `benchmark_quest_tpot.py`, `aggregate_results.py`, tests |
| `0105d00` | v0.5 engine kwargs (`disable_overlap_schedule`, `chunked_prefill_size`) |
| `064c68d` | Patch `is_multimodal` guard in flashinfer.py and trtllm.py |
| `1dcb4a0` | CuDNN-check workaround (`SGLANG_DISABLE_CUDNN_CHECK=1`) |
| `a60c599` | Benchmark results committed |

## Methodology

- One `sgl.Engine` per attention mode; streaming `engine.generate`,
  `max_new_tokens=256`, `repeat=3`, `ignore_eos`, one untimed warmup per batch.
- TPOT = `(stream_end - first_token_time) / (tokens_generated - 1)` — the
  baseline's exact formula.
- Engine config matched to `tpot-no-share`: `disable_cuda_graph=True`,
  `disable_radix_cache=True`, `attention_backend=flashinfer`,
  `decode_log_interval=1`, `show_time_cost=True`, `log_level=debug`, default
  `mem_fraction_static`.

## Deviations from the sgl baseline

- **Model: now MATCHES the baseline** (`Qwen3-VL-8B-Instruct`). The v0.3
  branch had to use `Qwen3-8B`; that workaround is eliminated in v0.5.
- **GPU stack:** B200 (sm_100); torch 2.9.1+cu128, sgl-kernel 0.3.21,
  flashinfer 0.6.3, triton 3.5.1.
- **Chunked prefill disabled:** `chunked_prefill_size` sized to one request;
  offline batch inference prefills each request whole.
- **Overlap schedule disabled** (`disable_overlap_schedule=True`): required for
  the vortex backend — see below.
- **`SGLANG_DISABLE_CUDNN_CHECK=1`**: sglang v0.5.9's CuDNN version check is
  over-strict for the bundled CuDNN 9.10 + torch 2.9.1 — see below.
- **Quest mode added** on top of the baseline's dense set; `topk_val=64`,
  `gqa_quest_sparse_attention` flow, flashinfer vortex backend.

## V0.5-specific notes

### 1. Multimodal guard patch (commit `064c68d`)

vortex v0.5's attention backends
(`vortex_torch/engine/sgl/attention_backend/flashinfer.py`, `trtllm.py`)
hard-asserted `assert not self.is_multimodal`. This blocked Qwen3-VL-8B-Instruct
(which sglang classifies as multimodal). The assertion was removed from both
files in commit `064c68d`.

Safe because: Qwen3-VL is a plain decoder (not encoder-decoder; the
`is_encoder_decoder` assertion is kept); sglang's `get_hf_text_config()`
unwraps Qwen3-VL's nested `text_config` so the backend reads correct shapes;
the vortex sparse path only touches decoder attention; the benchmark sends
text-only prompts. Verified by smoke test (coherent decode output).

### 2. Overlap schedule (`disable_overlap_schedule=True`)

The vortex quest backend reuses shared instance-attribute metadata buffers
across forwards; sglang's overlap scheduler runs forwards in a separate thread,
racing those buffers. Applied to both modes. vortex v0.5's own `get_engine`
hardcodes this flag for the same reason.

### 3. CuDNN-check workaround (`SGLANG_DISABLE_CUDNN_CHECK=1`)

sglang v0.5.9's startup check is over-strict for the bundled CuDNN 9.10 +
torch 2.9.1 stack on the B200. The harness sets `SGLANG_DISABLE_CUDNN_CHECK=1`
before importing sglang. The smoke test confirmed coherent decode output -- a
sanity check, not a CuDNN-matched numerical validation; sufficient for a TPOT
timing benchmark (any CuDNN effect is identical across the dense and quest runs).

### 4. Triton JIT compilation cache

vortex v0.5 is pure-Python + Triton (no CUDA extension). On first run from a
cold start, Triton compiles kernels during the first warmup generate for each
batch shape. A second run reuses the compiled cache and is faster. If running
benchmarks under a time-limited job, factor in ~5-10 min for initial
compilation before timed measurements begin.

### 5. Environment build deviated from the plan

The plan's Task 1 specified torch 2.7.1+cu128 / flashinfer 0.6.8 via sglang's
`[all]` extra. `setup_env.sh` instead installs sglang v0.5.9's **base**
dependencies only, which pin a different but internally-consistent stack
(torch 2.9.1, sgl-kernel 0.3.21, flashinfer 0.6.3). The `[all]` extra pulls
unrelated diffusion/tracing packages and desynced torch on the first attempt;
base-deps-only is the correct, reproducible build. `reference_freeze.txt`
records the exact resolved versions.

## How to re-run

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5

# Full benchmark (dense + quest, then aggregate)
GPU=0 bash quest_batch_benchmark/run_benchmark.sh

# Run modes separately (useful under time limits; harness appends to raw CSV)
quest_batch_benchmark/.venv/bin/python quest_batch_benchmark/benchmark_quest_tpot.py \
  --attention dense --raw-csv quest_batch_benchmark/results/raw_results.csv
quest_batch_benchmark/.venv/bin/python quest_batch_benchmark/benchmark_quest_tpot.py \
  --attention quest --raw-csv quest_batch_benchmark/results/raw_results.csv
quest_batch_benchmark/.venv/bin/python quest_batch_benchmark/aggregate_results.py

# Unit tests (19 tests)
quest_batch_benchmark/.venv/bin/python -m pytest quest_batch_benchmark/tests/ -q
```

If re-building the environment from scratch:

```bash
bash quest_batch_benchmark/setup_env.sh   # ~15-40 min on first run
```

## File index

| File | Purpose |
|------|---------|
| `setup_env.sh` | uv venv build for v0.5 stack |
| `benchmark_quest_tpot.py` | Engine harness, one attention mode, all batch sizes |
| `aggregate_results.py` | Aggregates raw CSV into processed curve |
| `run_benchmark.sh` | Driver for dense + quest + aggregate |
| `prompt_io.py` | Chat-template rendering of `request.json` |
| `request.json` | Benchmark input (9,661 tokens) |
| `reference_freeze.txt` | `uv pip freeze` of the exact environment |
| `results/raw_results.csv` | Per-(mode, batch, repeat) measurements |
| `results/tpot_vs_batchsize.csv` | Aggregated TPOT curve |
| `tests/` | Unit tests (19 tests) |
