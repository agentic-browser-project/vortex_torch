# Paper sweep settings: WebVoyager and GAIA

Settings, launchers, and tabulation scripts for the task-success experiments in the
paper: WebVoyager (Table 1, 12.5% selection ratio, page size 32 and 64) and GAIA
(task success at 6.25% and 12.5%, page size 32).

The files here are snapshots of the scripts that ran the sweeps on the Penn Betty
cluster. They use absolute cluster paths: `BSA` =
`/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention`, `E2E` =
`BSA/results/webvoyager_e2e`. Run them from their original locations (noted below)
or adjust the paths. The Vortex engine config is also described in
[`../webagent_sweep_settings.md`](../webagent_sweep_settings.md).

```
paper_sweeps/
  browsa/       gen_dpb_sweep.sh (WebVoyager sbatch generator), run_e2e_sglang_longwait.sh (runner),
                claude_pkBdp_r0p{0625,125}_r{1,2,3}.sbatch (GAIA)
  baselines/    run_{sweep,rest}_geom_noresv.sbatch (WebVoyager Quest / Block Sparse),
                run_dense_full.sbatch, claude_gaia_{geom,dense}.sbatch
  judge/        rejudge_existing.py, scan_none_verdicts.py, claude_default_judge_gaia.py,
                claude_gaia_default_judge.sbatch
  tabulate/     tab_per_site.py, check_coverage_all.py, tab_baselines.py, tab_distinctpage.py,
                tab_gaia_paper.py
  task_lists/   subset_vortex200.txt + subset_rest443.txt (BrowSA), subset_{200,rest443,full643}.json (Vortex/Dense)
```

---

## 1. Shared settings

| item | value |
|---|---|
| model | Qwen3-32B BF16 (`/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-32B`), TP=1, 1x B200 |
| agent | browser-use simulator (`jiaheng/browser-simulator/browser-use`, conda env `webvoyager`), text-only (`--no-vision`), `--max-steps 15`, `--task-timeout 5400`, `--llm-timeout 1200`, `--shuffle`, `SIM_MAX_TOKENS=1024` |
| judge | gemini-3.6-flash (details per benchmark below) |
| metric | Pass@3 (best-of-3): a task is solved if any of 3 independent runs solves it; the denominator is all tasks (643 or 90) |
| selection ratio | target fraction of the KV cache attended per decode step: `BSA2_TOKEN_SELECT_RATIO` for BrowSA, `topk_ratio` for Vortex |
| validity | every run has every task completed and judged; `verdict=None` means unjudged (re-judge it), never a fail |

**Page size.** For Quest and Block Sparse, the Vortex `block_size` is also the
FlashInfer page that the kernel reads (`vortex_torch/engine/sgl/api.py`,
`page_size=vortex_block_size`), even though the sglang allocator page is 64.
"Page Size 32/64" therefore means `block_size` 32/64 for the baselines and
`SGLANG_PAGE_SIZE` 32/64 for BrowSA.

---

## 2. BrowSA (packed, distinct-page budget)

- **Server:** the vendored sglang in `BSA`, run with `SGLANG_DECODE_BACKEND=bsa_v2_tuned_packing`.
- **Runner:** `E2E/run_e2e_sglang_longwait.sh <port_hint> <out_root> <task_num> 16`.
- **Concurrency:** 16, equal to the CUDA-graph max batch size.

### 2.1 The two configurations

| | Page Size 64 (Setting A) | Page Size 32 (Setting B) |
|---|---|---|
| `SGLANG_PAGE_SIZE` | 64 | 32 |
| chunk targets `BSA2_MIN_CHUNK` / `BSA2_MAX_CHUNK` | 6 / 32 | 6 / 16 |
| `BSA2_WN_REF` | 32 | 16 |
| `SGLANG_MEM_FRACTION_STATIC` | 0.60 | 0.55 (WebVoyager), 0.60 (GAIA) |
| used for | WebVoyager page 64 | WebVoyager page 32, GAIA |

### 2.2 Knobs shared by A and B

```bash
export SGLANG_DECODE_BACKEND=bsa_v2_tuned_packing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BSA2_E2E_BOUNDARY=1
export BSA_GRID_ROOT=/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention
export BSA_TUNED_ATTENTION=tc_no_split
export SGLANG_CUDA_GRAPH=1
export SGLANG_CUDA_GRAPH_MAX_BS=16
export SGLANG_RADIX_CACHE=0
export BSA2_KERNELS_DIR=$BSA_GRID_ROOT/build
# semantic chunking + scoring
export BSA2_CHUNK_MODE=semantic
export BSA2_CHUNK_EXTRACT=semantic_v2          # with the soft-min merge pass (BSA commit 3f66dfd)
export BSA2_MIN_CHUNK=6
export BSA2_SCORE_METHOD=mixmax_wn
export BSA2_WN_ALPHA=0.25
export BSA2_HEAD_SELECT=per_kv
export BSA2_QUERY_COMBINE=mean_q
# budget: page-count budget, so the label equals the measured attended fraction
export BSA_SELECTION_POLICY=distinct_page_budget
export BSA2_BUDGET_CONVENTION=count_generation_reservation
export BSA2_GEN_CHUNKS=0                       # generated tokens always attended, charged to the budget
export BSA2_RESERVED_BOS_PAGES=0
export BSA2_RESERVED_RECENT_PAGES=0
# packing
export BSA2_STATIC_PACK=d_worst
export BSA2_PACK_PRESERVE_RESERVED=1
export BSA2_ALWAYS_INCLUDE_START_PAGES=1
export BSA2_ALWAYS_INCLUDE_RECENT_PAGES=2
export BSA2_GRAPH_MAX_CHUNKS=4096
export BSA2_TOKEN_SELECT_RATIO=<0.125 | 0.0625>
export BSA2_REQUIRE_SPARSE=1                   # abort on any dense fallback
export BSA2_LOG_EVERY=20
export BSA2_PAGE_TELEMETRY=$OUT/<packed_selection|selected_pages>.jsonl
export BSA2_PAGE_TELEMETRY_EVERY=1             # per-step measured selection ratio
export SIM_STEP_TIMEOUT=1200
```

### 2.3 WebVoyager vs GAIA differences

| knob | WebVoyager | GAIA |
|---|---|---|
| `BSA2_PACK_CLUSTERING` | `gpu` | `cpu` |
| explicit zeros (the defaults) | — | `BSA2_PAGE_BUDGET=0 BSA2_SYS_FLOOR=0 BSA2_SYS_CAP=0 BSA2_HIST_FLOOR=0 BSA2_ALWAYS_INCLUDE_FIRST_CHUNK=0 BSA2_PACK_VERIFY_GPU=0` |
| tasks | `TASK_IDS_FILE=subset_vortex200.txt` or `subset_rest443.txt` | `SIM_SOURCE=gaia` (all 90) |
| runner call | `run_e2e_sglang_longwait.sh 10011 "$OUT" {200\|443} 16` | `run_e2e_sglang_longwait.sh 17251 "$OUT" 90 16` |

---

## 3. WebVoyager (Table 1): 643 tasks, 12.5% selection ratio

The 643 tasks are split into two halves: `subset_vortex200.txt` (200 tasks, "vtx")
and `subset_rest443.txt` (443 tasks, "rest"). Each run is two jobs, one per half.
Run *i* is the union of vtx[i] and rest[i].

### 3.1 BrowSA

```bash
cd $E2E/mixmax_sweep_sbatch            # gen_dpb_sweep.sh lives here
for half in vtx rest; do for run in 1 2 3; do
  sbatch $(bash gen_dpb_sweep.sh A 0.125 $half $run)   # page 64
  sbatch $(bash gen_dpb_sweep.sh B 0.125 $half $run)   # page 32
done; done
```

**Results** are written to
`E2E/results/mixmax_wn_sweep/sglang_packed_distinctpage/{setA_r0p125_ps64,setB_r0p125_ps32}_{vtx,rest}/run{1,2,3}/<session>/webvoyager__<task>/judge.json`.

**Sessions.** A run dir can contain several session dirs. For each task, the latest judged session wins. A session with ≥50 judged tasks and <5% pass is a serving failure and is dropped (`tab_per_site.py: sess()`).

### 3.2 Quest and Block Sparse (Vortex)

- **Stack:** `vortex_torch` + sglang v0.5.9 (conda env `vortex-bench`).
- **Server flags:** `--attention-backend flashinfer --page-size 64 --context-length 32768 --mem-fraction-static 0.9`.
- **Agent batch:** 32.
- **Launchers:** in `BSA/experiments/bsa_val/`, exact submit lines:

```bash
# <launcher> <mode> <ratio> <block_size> <bos> <eos> [topk_val=0]
# run_sweep_geom_noresv.sbatch = vtx half, run_rest_geom_noresv.sbatch = rest half
sbatch run_sweep_geom_noresv.sbatch quest            0.125 {32|64} 0 1 0
sbatch run_rest_geom_noresv.sbatch  quest            0.125 {32|64} 0 1 0
sbatch run_sweep_geom_noresv.sbatch gqa_block_sparse 0.125 {32|64} 1 1
sbatch run_rest_geom_noresv.sbatch  gqa_block_sparse 0.125 {32|64} 1 1
```

| method | Vortex module | reserved blocks |
|---|---|---|
| Quest | `gqa_quest_sparse_attention_sub` (`submissions/_flow_algorithms_test/gqa_quest_sparse_attention.py`) | bos 0, eos 1, topk_val 0 |
| Block Sparse | `gqa_block_sparse_attention_sub` (`submissions/_flow_algorithms_test/gqa_block_sparse_attention.py`, GQA max-group) | bos 1, eos 1, topk_val 0 |

**Vortex config.** It is written to `<OUT>/vortex_config.json`:
`impl_backend=triton, attention_backend=flashinfer, workload_chunk_size=32, layers_skip=[], dtype=bfloat16`.
eos must be ≥1, because FlashInfer applies `kv_last_page_len` to the final slot.
"Block Sparse" in the paper is the `gqa_block_sparse` module, not the mean-pool
`block_sparse` module.

**Run directories** (`E2E/results/sweep_sparsity/`; task dirs are flat, `<jobdir>/webvoyager__<task>/judge.json`):

| cell | vtx runs | rest runs |
|---|---|---|
| Quest p32 | `quest_r125_b32_bos0eos1_{8312894,8314481,8314500}` | `quest_r125_b32_bos0eos1_rest_{8348507,8360970,8360978}` |
| Quest p64 | `quest_r125_b64_bos0eos1_{8312904,8314491,8314509}` | `quest_r125_b64_bos0eos1_rest_{8428398,8428400,8428402}` |
| Block Sparse p32 | `gqa_block_sparse_r125_b32_bos1eos1_{8456429,8456431,8456433}` | `gqa_block_sparse_r125_b32_bos1eos1_rest_{8456430,8456432,8456450}` |
| Block Sparse p64 | `gqa_block_sparse_r125_b64_bos1eos1_{8456614,8456616,8456618}` | `gqa_block_sparse_r125_b64_bos1eos1_rest_{8456615,8456617,8456619}` |

### 3.3 Dense

`sbatch run_dense_full.sbatch dense 1.0 32 1 1 0` runs the same sglang FlashInfer
server with no `--vortex-config`. It uses page 64, all 643 tasks in one job, and
batch 32. The ratio and block arguments are ignored.

Runs are in `E2E/results/sweep_sparsity/`: `dense_full`, `dense_full643_8363108`, and `dense_full643_8363109`.

### 3.4 Judge and tabulation

The judge runs in the run itself (`SIM_JUDGE`, on by default; gemini-3.6-flash, vision on, last 10 screenshots). It writes `<task>/judge.json` with field `verdict`.

```bash
cd $E2E
python3 scan_none_verdicts.py        # find unjudged (None) verdicts
python3 rejudge_existing.py <dir>    # re-judge them; login node, <=6 dirs in parallel (API rate limits)
python3 check_coverage_all.py        # every cell should report 643 per run
python3 tab_per_site.py              # per-website + overall best-of-3, all 7 columns
python3 tab_baselines.py             # per-run and best-of-3 for the Vortex baselines
python3 tab_distinctpage.py          # per-run and best-of-3 for BrowSA
```

---

## 4. GAIA: 90 tasks, page size 32, selection ratio 6.25% and 12.5%

The task set is the GAIA web subset used by WebVoyager: 26 L1 and 64 L2 tasks. Every method runs 3 times at batch 16.

### 4.1 BrowSA (Setting B)

```bash
cd $E2E/mixmax_sweep_sbatch
sbatch claude_pkBdp_r0p0625_r{1,2,3}.sbatch   # 6.25%
sbatch claude_pkBdp_r0p125_r{1,2,3}.sbatch    # 12.5%
```

**Config:** Setting B (§2) with the GAIA column of §2.3. The two ratios differ only in `BSA2_TOKEN_SELECT_RATIO` and the output dir.

**Results:** `E2E/results/mixmax_wn_sweep/sglang_packed/dp_gaia_setB_r0p{0625,125}/run{1,2,3}/<session>/gaia__<task>/`.

### 4.2 Quest and Block Sparse (Vortex, block/page 32)

`claude_gaia_geom.sbatch` (in `BSA/experiments/bsa_val/`) is the Vortex launcher from §3.2, with three changes:
- `--source gaia --task-num ${GAIA_TASKS:-90}`
- batch `${GAIA_BATCH:-16}`
- output to `E2E/results/gaia_baselines_3run/<mode>_r<pct>[_b<block>]/run${GAIA_RUN}_<jobid>`

```bash
for r in 1 2 3; do
  sbatch --job-name=claude-gaia-quest-r$r --export=ALL,GAIA_RUN=$r claude_gaia_geom.sbatch quest            {0.0625|0.125} 32 0 1 0
  sbatch --job-name=claude-gaia-gqa-r$r   --export=ALL,GAIA_RUN=$r claude_gaia_geom.sbatch gqa_block_sparse {0.0625|0.125} 32 1 1 0
done
```

Result dirs are `gaia_baselines_3run/quest_r62`, `quest_r125_b32`, `gqa_block_sparse_r62`, and `gqa_block_sparse_r125_b32`. The `r62` dirs were created before the `_b32` suffix was added. They are block 32 too: their log headers show `block=32`.

### 4.3 Dense

```bash
for r in 1 2 3; do
  sbatch --job-name=claude-gaia-dense-r$r --export=ALL,GAIA_RUN=$r claude_gaia_dense.sbatch dense 0 32 0 1 0   # positional args required (set -u)
done
```

Runs are in `gaia_baselines_3run/dense/run{1,2,3}_<jobid>`.

### 4.4 Judge and tabulation

- **Baselines and Dense:** in-run judge (gemini-3.6-flash + vision + GAIA reference answers), written to `gaia__*/judge.json`.
- **BrowSA:** standard success protocol (official WebVoyager prompt + reference answer + last 2 screenshots, gemini-3.6-flash). Run `claude_default_judge_gaia.py <session_dir>`, or `claude_gaia_default_judge.sbatch` for all 6 runs. Output: `webvoyager_eval.json` with field `success`.
- **Judge gaps:** a task whose in-run judge errored (503 / timeout) is judged with the standard script. Its verdict goes into `webvoyager_eval.json` in the same task dir.

```bash
cd $E2E && python3 tab_gaia_paper.py   # best-of-3 for all 7 bars
```
