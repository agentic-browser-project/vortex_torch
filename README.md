# bench1: block-level KV fetch for vortex_torch

Experimental branch comparing block-granularity (`block_size=4` token)
KV gather against alternative fetch policies on top of vortex_torch's
sparse-attention algorithm.

## Setup on a fresh sm_120 / RTX 5060 Ti box

`vortex_v04` is the conda env name **on the original dev machine**. On a
fresh box you have to create it yourself. Roughly 15–30 min total
including downloads.

```bash
# 1. miniforge (≈ 2 min)
curl -L -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/mf.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh

# 2. env (≈ 1 min)
conda create -y -n vortex_v04 python=3.12
conda activate vortex_v04

# 3. clone (≈ 1 min)
git clone https://github.com/agentic-browser-project/vortex_torch
cd vortex_torch
git checkout bench1

# 4. torch + sglang + flashinfer (≈ 10 min, mostly torch download)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install flashinfer-python==0.6.8.post1
pip install PyYAML==6.0.3
pip install -e third_party/sglang/v0.4.9/sglang/python[all]
pip install uvloop==0.21.0 uvicorn==0.35.0

# 5. vortex_torch itself + transformers pin (≈ 2 min)
pip install -e .
# pyproject pins transformers==4.57.1 — pip will downgrade if you already had a newer one

# 6. sanity check
python -c "import torch, sglang, vortex_torch; print('ok', torch.__version__, torch.version.cuda)"
# expect: ok 2.7.1+cu128 12.8
```

### sm_120 (Blackwell consumer GPUs) — patches are ALREADY APPLIED

The three sgl-kernel-bypassing patches (`forward_cuda` → `forward_native`)
are **already in the bench1 branch**. No manual patching needed. The
JSON configs in this branch also already set `disable_cuda_graph: true`
and `sampling_backend: pytorch`.

You do NOT need to source-build `sgl-kernel`; the patches sidestep the
three modules that lack sm_120 prebuilt binaries. If a different
sgl-kernel module errors out at runtime, apply the same pattern
(replace `forward_cuda` body with `return self.forward_native(...)`).

### B200 reverts

On B200 (sm_100, has prebuilt sgl-kernel binaries), revert the
workarounds for full speed:
- `third_party/sglang/.../layernorm.py`        : restore `forward_cuda` body
- `third_party/sglang/.../rotary_embedding.py` : same
- `third_party/sglang/.../activation.py`       : same
- In the JSON: remove `disable_cuda_graph: true` and `sampling_backend: pytorch`
- Raise `mem_fraction_static` to ~0.85; switch `model_path` to a 70B-class model

### Python interpreters NOT to use

- `/home/wangxian/venv_magicpig/bin/python` (or any pre-existing venv on
  the box) — torch and transformers versions are wrong. vortex_torch's
  `pyproject.toml` pins `transformers==4.57.1`; sglang expects
  `torch==2.7.1+cu128`. A fresh conda env is the cheapest path.

## Methods

This branch evaluates three sub-page fetch implementations, plus two
index-rewrite policies layered on top.

### The three sub-page fetch implementations

All three load only the 4-token blocks the indexer selects. The
difference is *which kernel implements the gather*.

| Method | Backend kernel | Toggle | What it does |
|---|---|---|---|
| **3 B** (baseline) | FlashInfer `BatchDecodeWithPagedKVCacheWrapper(page_size=4)` | `VORTEX_POLICY=block_fetch` | Passthrough — kernel unchanged, indices used as-is |
| **3 C** | FlashInfer `BlockSparseAttentionWrapper(C=4, R=1)` | `VORTEX_USE_BSR=1` | Swap to BSR wrapper, plan per layer |
| **3 A** | Custom Triton / CUDA | `VORTEX_USE_CUSTOM=1` | Our own flash-attention kernel with gather fused in |

Method 3 A has three sub-variants:
- `VORTEX_CUSTOM_KERNEL_VERSION=v1` — Triton v1 (simple, 1 program / `(row, q_head)`)
- `VORTEX_CUSTOM_KERNEL_VERSION=v2` — Triton v2 (tile-fused + `tl.dot`, designed for G≥8)
- `VORTEX_CUSTOM_KERNEL_VERSION=cuda` — hand-written CUDA, JIT-compiled via `cpp_extension`
- `VORTEX_CUSTOM_KERNEL_VERSION=auto` (default) — picks v2 if G≥8 else v1

### Round-up fetch & threshold fetch (methods 1 / 2)

These are policies that REWRITE `sparse_kv_indices` before the kernel runs:

| Method | Toggle | What it does |
|---|---|---|
| **1** — round-up fetch | `VORTEX_POLICY=method1_p32` | If a 32-token page has ≥1 selected block, load all 8 blocks (lossless over-fetch) |
| **2** — threshold fetch | `VORTEX_POLICY=method2_p32_t{25,50,75}` | Load a 32-token page only when ≥TT% of its blocks are selected, else drop (lossy) |

These currently auto-route through the custom Triton kernel because
the BatchDecode wrapper has a stale-plan bug (see TODO below).

## Run

```bash
# single config on Qwen3-0.6B
VORTEX_POLICY=block_fetch python algorithm_scientist/run_ruler_trace.py \
    --config submissions/block_size_sweep/batch_0_id2_page32.json

# the configs available
ls submissions/block_size_sweep/batch_0_id2_page32*.json
#   batch_0_id2_page32.json         — Qwen3-0.6B
#   batch_0_id2_page32_qwen3_4b.json — Qwen3-4B (needs ~10 GB GPU mem)

# Method 1 + Method 2 stats + HBM bytes
VORTEX_HBM_TRACE=logs/hbm.json VORTEX_POLICY_STATS=1 \
    VORTEX_POLICY=method1_p32 python algorithm_scientist/run_ruler_trace.py ...
cat logs/hbm.json   # coverage, waste, KV bytes loaded, page-hit histogram

# All seven configs at once
bash run_method_comparison.sh
```

### Standalone kernel unit test (no sglang import)

```bash
python algorithm_scientist/test_custom_kernel.py
# tests Triton v1, v2, and CUDA against a naive PyTorch reference
```

## Metrics

| Metric | Meaning | How to read |
|---|---|---|
| **accuracy** | RULER substring-match: did model recover the magic UUID? | 0.0 / 0.5 / 1.0 |
| **throughput** (tok/s) | End-to-end tokens generated per second (prefill + decode + sampling) | Higher is better. Noisy on small workloads — only trust deltas > 10% |
| **avg idx/call** | Blocks gathered per layer × decode step | Direct HBM pressure proxy |
| **KV MB/call** | Bytes pulled from HBM per layer × decode step ≈ `idx × BS × D × 2 × 2 B` | Higher = more HBM bandwidth used |
| **coverage** | `(selected ∩ loaded) / selected` — fraction of indexer's choices preserved by the policy | 1.0 for lossless methods (block-level, round-up); <1.0 for threshold |
| **waste** | `(loaded - selected ∩ loaded) / loaded` — fraction of loaded blocks the indexer didn't pick | 0 for block-level fetch; >0 for round-up / threshold when pages aren't full |
| **page-hit histogram** | For each 32-token page, how many of its 8 blocks the indexer selected | Tells you whether selection is page-aligned (mostly 8/8) or scattered (1-7/8) |

`accuracy / throughput` come from the RULER summary JSON. The rest are
in the file pointed to by `VORTEX_HBM_TRACE`.

## Current data we have

On Qwen3-4B (G=4, 36 layers, RULER 2-prompt subset, **with the
recent bug fix** in `apply_policy`):

| method | accuracy | throughput | KV MB/call | coverage | waste |
|---|---|---|---|---|---|
| 3 B (block_fetch baseline) | 1.0 | 8.5 | 4.02 | — (no policy) | — |
| 3 A v1 (Triton) | 1.0 | 8.3 | 4.02 | — | — |
| 3 A cuda | 1.0 | 8.4 | 4.02 | — | — |
| 3 C (BSR) | 1.0 | 7.8 | 4.02 | — | — |
| **1** (round-up, post-fix) | needs re-test on 4B | needs re-test | 4.13 | **1.00** | **0.026** |
| 2 t50 | needs re-test on 4B | needs re-test | similar | TBD | TBD |

**Key empirical finding (page-hit histogram from a method-1 run):**
95% of pages that the indexer hits are **fully selected (8/8 blocks)**.
Only 5% are partial (1–7 blocks). This means round-up fetch's
expansion only adds **2.6% waste** — the indexer's selection is
already nearly page-aligned, so methods 1 and 2 have very little
room to differentiate from block-level fetch on **this** algorithm.

The algorithm in question is `block_size_sweep_id{0,1,2,3}` = same
centroid-based scoring as `submissions/example_block_sparse_attention.py`:

```python
q_mean = Mean(q, dim=1)
score = q_mean · cache["centroids"]      # centroid = Mean(K_block, dim=1)
topK(score) → selected blocks
```

Centroid scoring naturally produces clustered (page-aligned) selection
because adjacent blocks have similar centroids.

## Algorithms available in this branch

| Path | Algorithm | Why it might give different page-hit patterns |
|---|---|---|
| `submissions/block_size_sweep/batch_0_id{0,1,2,3}.py` | Centroid + GeMM + topK (all 4 are the same algorithm; vary only block_size × topk_val) | The current default. Tends to select page-aligned |
| `submissions/example_block_sparse_attention.py` | Same centroid algorithm | (Reference / template) |
| `submissions/gqa_quest_approx.py` | **Quest**: per-block MIN + MAX statistics, `score = max(q·min, q·max)` | Theoretically more discriminative between adjacent blocks → may scatter |
| `submissions/{claude_opus_4_7, claude_sonnet_4_6, gpt_5}/` (each has 20 `innovate_0_id{0..19}` variants) | Other AI agents' algorithm submissions | Worth surveying for diversity |
| `submissions/kimi_v0.py`, `submissions/oai_v0.py` | Centroid-based variants (mean + topK), small renaming | Same family as block_size_sweep, unlikely to scatter |

## Layout

| File | Purpose |
|---|---|
| `vortex_torch/engine/sgl/attention_backend/flashinfer.py` | Backend dispatcher — wires up B / C / A and policy hook |
| `vortex_torch/engine/sgl/policy_transform.py` | Method 1 / 2 index rewriter, coverage/waste/histogram stats |
| `vortex_torch/engine/sgl/hbm_trace.py` | HBM bytes counter + stats aggregator |
| `vortex_torch/engine/sgl/attention_backend/block_sparse_decode_triton.py` | Method 3 A Triton kernels (v1, v2) |
| `vortex_torch/engine/sgl/attention_backend/block_sparse_decode_cuda.py` | Method 3 A CUDA kernel (JIT via `cpp_extension`) |
| `run_method_comparison.sh` | 7-config sweep driver |
| `algorithm_scientist/test_custom_kernel.py` | Standalone unit test for the custom kernels |

## TODO

### High priority — correctness blockers

- [ ] **Fix BatchDecode re-plan for the policy path.** When
  `VORTEX_POLICY=method1_p32` (or method2) modifies `sparse_kv_indptr`
  per layer, the FlashInfer `BatchDecodeWithPagedKVCacheWrapper`
  has a stale plan from `init_forward_metadata`. Calling `plan()`
  again brings accuracy from 0.0 to 0.5 but not 1.0. Suspected
  causes: workspace_buffer too small for expanded indices, or
  cuda-graph state aliasing. **Current workaround**: when a policy
  is active, force the custom Triton path (see comment in
  `flashinfer.py` `forward_decode`). This works because Triton
  kernels don't have plan/run separation. But it means we can't
  measure `policy + BatchDecode` independently of `policy +
  Triton`. Resolving this would let us isolate policy effects.

- [ ] **Re-run the full Qwen3-4B sweep with the bug fix** to get
  real `accuracy / throughput` data for methods 1 / 2 (previous
  sweeps before commit `<TBD>` were measuring passthrough because
  `apply_policy` was returning early due to the `n_rows` bug —
  see history below).

### Medium priority — make Methods 1/2 meaningful

- [ ] **Try Quest indexer** (`submissions/gqa_quest_approx.py`):
  min/max statistics may scatter selection more than centroid,
  giving Methods 1/2 a real workload to operate on. Compare
  page-hit histograms between centroid and Quest.

- [ ] **Sweep `vortex_topk_val`** (e.g. 32, 64) at fixed algorithm.
  Hypothesis: smaller K → less page-saturated → Method 1 expansion
  factor and Method 2 drop rate become non-trivial. Re-test
  coverage / waste at each setting.

- [ ] **Survey other algorithms** in `submissions/{claude_opus_4_7,
  claude_sonnet_4_6, kimi_v0, oai_v0, gpt_5}/` for variants
  (channel sparsity, dual-band centroids, etc.). Run each through
  the page-hit histogram to see which produces scatter.

### Medium priority — kernel performance

- [ ] **Tune Triton v2 for `G < 16`.** Current v2 pads M dim to 16
  for `tl.dot`'s minimum, which wastes compute when `G ∈ {2, 4}`.
  Try `tl.dot` with M=8 (some Triton versions allow this) or fall
  through to v1 logic per layer based on actual G.

- [ ] **Tune CUDA kernel.** Current CUDA is "correctness first" —
  single-threaded softmax, no tensor cores, no warp shuffles.
  Add cooperative QK dot (warp shuffles), use tensor cores via
  cutlass primitives, async copy on Hopper / Blackwell.

- [ ] **B200 + Qwen2.5-72B benchmark.** Local sm_120 sweep is at
  the noise floor (~5%). Real performance differentiation needs
  larger model + longer context. Confirm:
  - BSR (Option C) `prefill-kernel-on-decode` penalty shrinks as
    model size grows
  - Custom kernel actually saves HBM bandwidth in `nsys` profile
  - Triton v2 wins over v1 once G ≥ 8

### Low priority — infrastructure

- [ ] **Resolve the `n_rows` propagation through CUDA graphs.**
  `apply_policy` now correctly takes `n_rows=q.shape[0]`. Verify
  this stays correct when `disable_cuda_graph: false` (currently
  forced to `true` because of sm_120 workarounds).

- [ ] **HBM tracer write granularity.** Currently flushes every 32
  calls (`hbm_trace.py:_FLUSH_EVERY`). Drop to per-call if we want
  layer-resolution debugging.

- [ ] **CUDA kernel: support BS != 4 and D != {64, 128}.** Current
  template specialisations cover only this branch's setup.

## History (the bugs we found)

These are documented because someone reading old commit results
will need context:

1. **`n_rows` bug in `apply_policy` (FIXED).** The CSR `indptr`
   buffer is pre-allocated for `max_bs × num_kv_heads + 1 = 8193`
   entries but only the first ~16 are filled per step. The old
   code read `total_len = indptr[-1] = indptr[8192] = 0` and
   immediately returned without modifying anything. So **all
   sweeps before this fix showed Methods 1 / 2 with `accuracy / KV
   MB / throughput` numerically equal to `block_fetch` not because
   the algorithms are equivalent but because the policy hook was
   never running**. Fixed by passing `n_rows = q.shape[0]` explicitly
   from `flashinfer.py:forward_decode` and using `indptr[n_rows]`
   instead of `indptr[-1]`.

2. **BatchDecode stale-plan bug (WORKAROUND, see TODO above).**
   Once `apply_policy` actually modifies indices, BatchDecode's
   pre-computed split-K plan is wrong → accuracy drops. Currently
   worked around by routing through the Triton kernel.

3. **BSR plan-time indices snapshot (FIXED).** FlashInfer's
   `BlockSparseAttentionWrapper.plan()` snapshots indices values
   at plan time (unlike BatchDecode which can reuse a registered
   buffer pointer). For BSR we plan per-layer inside `forward_decode`
   instead of once-per-step in `init_forward_metadata`.

4. **sm_120 toolchain workarounds (B200 needs reverts).** Three
   sgl-kernel files patched to use `forward_native` fallbacks
   because sgl-kernel doesn't ship sm_120 prebuilt binaries.
   Revert all three on B200.
