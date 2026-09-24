# Web-agent sweep settings: quest & gqa_block_sparse (Qwen3-32B baselines)

Authoritative record of the exact vortex_torch settings behind the web-agent
sparsity baselines (GAIA-web 3-run baselines 2026-09-23; the same launcher
pattern is used for WebVoyager / WebArena / Online-Mind2Web suites). The
launchers themselves live in the BrowserSparseAttention repo:
`experiments/bsa_val/claude_gaia_geom.sbatch` (sparse cells) and
`claude_gaia_dense.sbatch` / `claude_om2w_dense.sbatch` (dense; no vortex
config). Results land under
`BrowserSparseAttention/results/webvoyager_e2e/results/gaia_baselines_3run/`
(and `om2w_baselines/` for Online-Mind2Web).

## Flow modules — both baselines are the GQA variants

| sweep name | vortex module (`module_name` / `module_path`) |
|---|---|
| `quest` | `gqa_quest_sparse_attention_sub` — `submissions/_flow_algorithms_test/gqa_quest_sparse_attention.py` |
| `gqa_block_sparse` | `gqa_block_sparse_attention_sub` — `submissions/_flow_algorithms_test/gqa_block_sparse_attention.py` |
| `block_sparse` (legacy) | `block_sparse_attention_sub` — `submissions/_flow_algorithms_test/block_sparse_attention.py` |

Naming convention (2026-09-23): **"block sparse" means `gqa_block_sparse`**
(max over the GQA query group); only "mean_pool_block_sparse" refers to the
mean-pool+centroid `block_sparse_attention` module. `quest` has always been
the GQA variant (`gqa_quest_sparse_attention`) in these sweeps. Historical
caveat: the big 643-task WebVoyager table (2026-08) used the mean-pool module
under the name "block_sparse" — do not mix those rows with the GQA rows.

## Vortex engine config (exact JSON passed as `--vortex-config`)

```json
{"impl_backend": "triton", "attention_backend": "flashinfer",
 "block_size": <32 or 64>,
 "topk_val": 0, "topk_ratio": <0.03125 | 0.0625 | 0.125>,
 "block_reserved_bos": 1, "block_reserved_eos": 1,
 "workload_chunk_size": 32, "layers_skip": [],
 "module_name": "<see table>", "module_path": "<see table>",
 "dtype": "bfloat16", "compilation_cache_dir": "~/.vortex_compilation_cache"}
```

- **Zero-reservation convention**: `topk_val = 0` so the sparsity label is the
  ratio alone (static floor = `bos + eos` = 2 blocks). `bos = 1` stays — the
  BOS sink is load-bearing (removing it gave 0% pass). `eos` must stay >= 1
  (planner enforces it; `kv_last_page_len` is applied to the final slot).
  Budget: `max(topk_val + bos + eos, int(num_blocks * topk_ratio))`, min 1.
- **Geometry axes**: `page_size = 64` fixed (sglang `--page-size 64`);
  `block_size ∈ {32, 64}`; `topk_ratio ∈ {0.03125, 0.0625, 0.125}`
  (3.125% / 6.25% / 12.5%). Finding: at 3.125% the b64 cells collapse
  (quest 0.4%, gqa_block 4.8% on GAIA) while b32 holds — replicated on
  WebVoyager; prefer b32 for low ratios.
- Labels are near-honest for quest/gqa_block (per-head selection, no packing);
  BSA packed labels are NOT comparable without the real-ratio audit.

## Serving (vortex sglang, one B200)

```
<vortex-bench python> -c 'import vortex_torch; sglang.launch_server ...'
  --model-path Qwen/Qwen3-32B --attention-backend flashinfer
  --vortex-config "$(cat cfg.json)" --page-size 64 --context-length 32768
  --mem-fraction-static 0.9 --enable-metrics
```

Env: `SGLANG_DISABLE_CUDNN_CHECK=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`LD_LIBRARY_PATH=<env312>/lib` prepended. Dense baseline = same server with NO
`--vortex-config`. `VORTEX_DISABLE_SPLIT_KV=1` exists for TPOT-methodology
runs (forces non-split tensor-core decode); leave unset for accuracy sweeps.

## Agent harness (browser-simulator, headed)

- `python -m simulator capture --no-vision --task-num <N> --batch-size 16
  --source <gaia|webvoyager|online_mind2web> --shuffle --max-steps 15
  --task-timeout 5400 --llm-timeout 1200`
- Env: `SIM_HEADLESS=0` under containerized Xvfb (headless triggers
  bot-blocking), `SIM_MAX_TOKENS=1024`, `SIM_STEP_TIMEOUT=1200`,
  in-run judge `SIM_JUDGE=1 SIM_JUDGE_MODEL=gemini-3.6-flash
  SIM_JUDGE_CONCURRENCY=8 SIM_JUDGE_VISION=1`.
- **3 runs per cell** (`GAIA_RUN=1..3`), report avg and best-of-3; per-run sd
  on GAIA-90 is ~3.5-4 pts, so single-run <10 pt differences are noise.
- Each job also records a batch-1 TTFT/TPOT probe at 4k/16k/25k prompt tokens
  (unique nonce per request so the radix cache cannot fake TTFT) into
  `timing.json`, plus sglang Prometheus counters over the real workload
  (`metrics.txt`).

## Reference results (GAIA-90, flash judge, 3-run averages, 2026-09-23)

| cell | 3.125% | 6.25% | 12.5% |
|---|---|---|---|
| dense (no ratio) | 18.1 avg / 36.7 bo3 | " | " |
| quest b32 | 11.9 | 15.9 | 18.6 |
| gqa_block b32 | 14.4 | 17.8 | 18.1 |
| quest b64 | 0.4 (collapse) | — | 13.8 |
| gqa_block b64 | 4.8 (collapse) | — | 20.4 |

Entries are averages; per-run numbers and best-of-3 live in the run dirs
(`gaia_baselines_3run/<mode>_r<ratio*1000>_b<block>/run*/`).
Baselines saturate at dense by 12.5%.
