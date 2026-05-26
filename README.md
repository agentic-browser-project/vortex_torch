# bench1: Block Fetch vs Method 1 vs Method 2

A **simulation analysis** comparing three KV-fetch policies on real selection
traces produced by vortex_torch's `block_sparse_attention` upstream algorithm.

## The three policies

| Policy | What it does | At what granularity |
|---|---|---|
| **Block Fetch** | Load only the selected blocks (vortex_torch's actual current behavior) | block_size (= 16) |
| **Method 1** | If any selected block falls inside a page, load the whole page | page_size > block_size |
| **Method 2** | Load a page only when its hit ratio ≥ threshold; otherwise drop entire page | page_size > block_size |

For Method 1 and Method 2 to differ from Block Fetch, `page_size` must be
strictly larger than `block_size`. We sweep `page_size ∈ {32, 64, 128, 256}`.

This is **simulation only**: vortex_torch actually runs Block Fetch (page = block
= 16); the analyzer computes what Method 1 / Method 2 would have loaded
instead, using the real selection traces. No alternative fetch is actually
executed.

## Headline result

After running, see `logs/trace_policy_analysis.csv`. Block Fetch wins:
`coverage = 1.0`, `waste = 0.0`, smallest `loaded MB`. Method 1 over-fetches
(waste grows with page_size); Method 2 either matches Method 1 (low threshold)
or catastrophically drops content (high threshold), because the upstream
algorithm's selection is already block-aligned in small clusters.

---

## Setup on B200

1. **Install vortex_torch + sglang** per the upstream install instructions
   (~25 min compute + 15min–2h model download).
2. **Use a 70B-class model** (e.g. `Qwen/Qwen2.5-72B-Instruct` or
   `meta-llama/Llama-3.1-70B-Instruct`). Set it via
   `"model_path": "..."` in `submissions/block_size_sweep/batch_0_id0.json`,
   or by editing `MODEL_PATH` in
   [`vortex_torch/engine/sgl/api.py:22`](vortex_torch/engine/sgl/api.py#L22).
3. **Revert the four sm_120 (RTX 5060 Ti / Blackwell) workarounds** from this
   branch — B200 has full sm_100 sgl-kernel binaries and doesn't need them:

   | File | What to revert |
   |---|---|
   | `third_party/sglang/v0.4.9/sglang/python/sglang/srt/layers/layernorm.py` | The `# PATCH (block_size_sweep, sm_120)` block inside `RMSNorm.forward_cuda` — restore the original sgl-kernel `fused_add_rmsnorm` / `rmsnorm` body |
   | `third_party/sglang/v0.4.9/sglang/python/sglang/srt/layers/rotary_embedding.py` | Same pattern in `forward_cuda` — restore the sgl-kernel `apply_rope_with_cos_sin_cache_inplace` path |
   | `third_party/sglang/v0.4.9/sglang/python/sglang/srt/layers/activation.py` | Same pattern in `SiluAndMul.forward_cuda` — restore the sgl-kernel `silu_and_mul` call |
   | `submissions/block_size_sweep/batch_0_id0.json` | Delete `"disable_cuda_graph": true` and `"sampling_backend": "pytorch"` |

---

## Run

```bash
bash run_all_experiments.sh
```

Expected time on B200 + 70B + warm caches: **~5–10 min** (2-sample RULER run
+ analyzer).

To use your own Python interpreter:
```bash
PYTHON=/path/to/env/bin/python bash run_all_experiments.sh
```

The script does two steps:

1. **Dump traces** — runs RULER (2 prompts) with vortex_torch's
   `block_sparse_attention`, captures every layer's `sparse_kv_indices`
   into `logs/traces/`.
2. **Analyze traces** — computes Block Fetch / Method 1 / Method 2 metrics
   for every (request, kv_head) selection across all decode steps; aggregates
   into `logs/trace_policy_analysis.csv`.

---

## Output: `logs/trace_policy_analysis.csv`

| column | meaning |
|---|---|
| **method** | `Block Fetch` / `Method 1` / `Method 2` |
| **P** | For Method 1/2: the page_size (the "load whole or drop" decision unit; > block_size). For Block Fetch: equal to block_size (= 16). |
| **threshold** | Method 2 only: a page is loaded only when `(selected_blocks_in_page / blocks_per_page) ≥ threshold`. Otherwise dropped (its selected blocks contribute nothing to attention). |
| **count** | Number of (request, kv_head) selections aggregated |
| **cov_mean** | Mean **coverage**: fraction of needed tokens that were actually loaded. 1.0 = no loss. Block Fetch and Method 1 are always 1.0 by construction; Method 2 may be < 1.0. |
| **cov_p50 / cov_p10** | 50th / 10th percentile of coverage across selections (shows the variability) |
| **waste_mean** | Mean **waste**: fraction of loaded bytes that were not needed. 0.0 = perfect. Block Fetch is 0; Method 1 = `1 − block_size/P`. |
| **waste_p50** | Median waste |
| **N_mean** | Mean number of needed tokens per (request, kv_head). With default vortex_torch config this is ~511. |
| **loadMB_mean** | Mean megabytes actually loaded from HBM per (request, kv_head). This is the bandwidth cost. |

### How to read the table

- **Block Fetch row**: the baseline. `cov_mean = 1.0`, `waste_mean = 0.0`,
  smallest `loadMB_mean`. This is what vortex_torch already does.
- **Method 1 rows** (one per page_size): `cov_mean = 1.0` always; `waste_mean` and
  `loadMB_mean` grow with `P`. Tells you "if you grouped blocks into pages of
  size P, how much would you over-fetch".
- **Method 2 rows** (one per page_size × threshold): `cov_mean` falls off a
  cliff once `threshold × (P/block_size)` exceeds the average cluster size in
  the trace. Below the cliff it equals Method 1; above the cliff it's zero.

The expected pattern (and what we observed on the RTX 5060 Ti development run):
Block Fetch is strictly the best on this workload because the upstream
algorithm's selection is already block-aligned in small (~2–4 block) clusters,
which leaves no room for either coarser fetch or selective dropping to win.

---

## Reproducing pieces individually

Just dump traces:
```bash
VORTEX_DUMP_TRACE_DIR=logs/traces python algorithm_scientist/run_ruler_trace.py \
    --config submissions/block_size_sweep/batch_0_id0.json
```

Just analyze existing traces (try different page sizes / thresholds):
```bash
python algorithm_scientist/trace_policy_analyzer.py \
    --trace-dir logs/traces \
    --page-sizes 32 64 128 256 \
    --thresholds 0.10 0.25 0.50 0.75 \
    --out-csv logs/trace_policy_analysis.csv
```

---

## Other files in this branch (not used by the main experiment)

These were exploratory experiments from earlier iterations. They are not
called by `run_all_experiments.sh` and can be ignored or deleted:

- `bench_decode_bandwidth_notc.py`, `run_bandwidth_sweep*.sh` — hardware
  bandwidth profiling (establishes that vortex_torch's default workload is
  launch-overhead-bound)
- `algorithm_scientist/page_policy_sweep.py` — synthetic-distribution
  comparison of the three policies (random vs clustered token patterns;
  superseded by the real-trace analysis above)
- `algorithm_scientist/tpot_microbench.py` — per-decode-step latency measurement
- `submissions/block_size_sweep/batch_0_id{1,2,3}.*` — additional submission
  variants at block_size ∈ {8, 4, 1}; only `id0` (block_size=16) is used
- `run_aime24_sequential.sh`, `run_remaining_ruler.sh`, `run_env.sh` — earlier
  orchestration scripts
