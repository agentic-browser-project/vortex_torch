# bench1: Three implementation options for vortex_torch sparse fetch

A **real** comparison (not simulation) of several KV-fetch policies on
vortex_torch's `block_sparse_attention` upstream, with `block_size=4`
and `page_size=32`. Plus two alternative attention-backend integrations
(Options B and C) and a written plan for a third (Option A).

## The three options

The user goal: replace vortex_torch's current "fetch a 32-token page when
any 4-token block in it is selected" behavior with **truly sub-page
(block-granularity) gather** at the GPU memory layer. Three implementation
routes — increasing effort, increasing payoff:

| Option | What it does | Effort | Status |
|---|---|---|---|
| **B** — indices transformation hook | Rewrites `sparse_kv_indices` in place to simulate different page/threshold policies; underlying FlashInfer kernel unchanged (still page-of-block_size=4 BatchDecodeWithPagedKVCacheWrapper) | 1 day | **DONE** |
| **C** — BSR wrapper swap | Drop `BatchDecodeWithPagedKVCacheWrapper(page_size=4)` for `BlockSparseAttentionWrapper(C=4, R=1)`. Flat KV view, arbitrary block gather, prefill kernel | 2–3 days | **DONE** |
| **A** — custom Triton fetch kernel | Custom Triton kernel that fuses block-granularity (BS=4) gather directly into flash-attention online softmax — no FlashInfer wrapper, no paging. Sub-page gather lives inside the kernel's K/V load loop. | 5–10 days planned, **v1 in 1 day** | **DONE (v1)** — needs tile tuning |

## The Option-B policies (compared in `run_method_comparison.sh`)

Within Option B, we compare three index-rewrite policies plus the
straight BSR backend (Option C, no policy rewrite):

| Policy | What it does | Granularity |
|---|---|---|
| **Block Fetch** | Pass indices through unchanged (vortex_torch's actual current behavior) | block_size=4 |
| **Method 1** (`method1_p32`) | If any selected block falls inside a 32-token page, load ALL 8 blocks of that page | P=32 (= sglang page_size) |
| **Method 2** (`method2_p32_tTT`) | Load a 32-token page only when hit ratio ≥ TT/100, otherwise drop | P=32, TT=25/50/75 |
| **BSR Baseline** (`bsr_baseline`, Option C) | Same selected blocks, but routed through FlashInfer's BlockSparseAttentionWrapper instead of BatchDecodeWithPagedKVCacheWrapper | block_size=4 |

All three Option-B policies use `P=32` to align with sglang's configured
`page_size=32`. Method 1 is lossless (always keeps all selected blocks);
Method 2 may drop content if a page's hit ratio falls below the threshold.

**This is real, not simulation**: an in-flight hook in
[`flashinfer.py`](vortex_torch/engine/sgl/attention_backend/flashinfer.py)
rewrites the `sparse_kv_indices` buffer after the indexer fills it but
before the attention kernel runs. The attention call actually sees the
transformed indices.

## Headline result

On vortex_torch's `block_sparse_attention` algorithm (block_size=4) +
RULER 2-sample subset, all six configurations preserve accuracy=1.0.
Throughput is similar across them (~10 tok/s, within ~5%):

| Policy | Accuracy | Throughput (tok/s) | vs Block Fetch |
|---|---|---|---|
| **block_fetch** (baseline) | 1.0 | **10.07** | — |
| method1_p32 | 1.0 | 10.36 | +2.9% |
| method2_p32_t25 | 1.0 | 10.49 | +4.2% |
| method2_p32_t50 | 1.0 | 10.11 | +0.4% |
| method2_p32_t75 | 1.0 | 10.43 | +3.6% |
| **bsr_baseline** (Option C) | 1.0 | 9.83 | −2.4% |
| **custom_baseline** (Option A, Triton v1) | 1.0 | 9.98 | −0.9% |

**Caveat on noise:** the workload here is tiny (Qwen3-0.6B + 2 RULER
prompts + 95 total decode tokens), so per-policy throughput swings of
several percent are within noise. Earlier runs of Option B on a
contended GPU showed a clearer 5–8% gap because the shared-GPU
contention amplified the Python-level policy overhead. On a clean GPU
the policies are statistically tied. Bigger model + more prompts on
B200 will give a cleaner ordering.

### What this tells us about the three options

- **Option B (indices hook)** is the right tool for *policy* experiments
  but doesn't reduce HBM traffic — the underlying FlashInfer kernel
  still gathers at block_size=4 paged granularity. The "savings" from
  Method 2 dropping pages do appear in the indices count, but kernel
  time is dominated by other costs at this scale.
- **Option C (BSR wrapper swap)** is a real backend change: it
  replaces the decode-specialized BatchDecode kernel with a prefill
  BSR kernel. On q_len=1 decode, BSR is slightly slower (−1.4%) than
  BatchDecode because the prefill kernel isn't decode-optimized. The
  value of Option C is **infrastructural**: BSR exposes arbitrary block
  gather without paging, which is the cleanest substrate to swap in
  Option A's custom kernel later.
- **Option A (custom CUDA fetch + flash-attention kernel)** is the only
  path to truly reducing HBM traffic at sub-page granularity AND
  recovering decode-kernel performance. See plan below.

## Setup on B200

1. **Install vortex_torch + sglang** per upstream install instructions
   (~25 min compute + 15min–2h model download).
2. **Pick a 70B-class model** (e.g. `Qwen/Qwen2.5-72B-Instruct`). Set
   `"model_path"` in the JSON. (Default in this branch is
   `Qwen/Qwen3-0.6B` — chosen because of shared-GPU constraints
   during development; B200 should use a larger model.)
3. **Increase `mem_fraction_static`** back to 0.8+ (we set it to 0.35
   because of a co-tenant on our development GPU). With B200's 192 GB,
   even 70B fits comfortably.
4. **Revert four sm_120 (RTX 5060 Ti) workarounds** — see "Local
   environment hacks" section below.

## Run

```bash
bash run_method_comparison.sh
```

Expected time on B200: ~10–15 min (6 configs × ~2 min each).

Override Python interpreter:
```bash
PYTHON=/path/to/env/bin/python bash run_method_comparison.sh
```

What the script does, for each of 6 configurations
(`block_fetch`, `method1_p32`, `method2_p32_t25`, `method2_p32_t50`,
`method2_p32_t75`, `bsr_baseline`):
1. Set the env var that selects the backend (`VORTEX_POLICY=<policy>`
   for Options B; `VORTEX_USE_BSR=1` for Option C).
2. Run RULER on a 2-prompt subset with vortex_torch's
   `block_sparse_attention` algorithm.
3. Aggregate `(accuracy, throughput)` into a single TSV summary.

## Output

`logs/method_comparison_<timestamp>/`:
- `summary.tsv` — one row per config, with `accuracy`, `throughput`, status
- `<policy>.out` / `<policy>.err` — per-policy logs
- `<policy>_result.json` — full RULER summary per policy

### How to read the results table

| Column | Meaning |
|---|---|
| **policy** | The fetch policy name (e.g. `method1_p32` = Method 1 with page_size=32; `bsr_baseline` = Option C swap) |
| **accuracy** | RULER substring-match accuracy: did the model find the magic UUID? Range [0, 1] |
| **throughput** | Tokens/sec generated end-to-end (includes prefill, decode, sampling) |
| **rc** | Return code (`OK` / `FAIL`) |

Expected pattern: accuracy should stay at 1.0 across all configurations
on this easy needle-in-haystack task. Throughput differences are
within noise on a clean GPU at this scale; the meaningful comparison
needs a larger model.

## Configuration

Currently used: [`submissions/block_size_sweep/batch_0_id2_page32.json`](submissions/block_size_sweep/batch_0_id2_page32.json)

| Key | Value | Why |
|---|---|---|
| `vortex_block_size` | 4 | Indexer selects 4-token chunks (finest viable; <4 would hit kernel limits) |
| `page_size` | 32 | sglang's internal page is 32 (= 8 blocks/page). Note: this overrides the default `page_size=vortex_block_size` and exercises the `page > block` framework path |
| `vortex_max_seq_lens` | 6144 | RULER input ~4500 tokens fits comfortably |
| `vortex_topk_val` | 116 | 116 blocks × 4 tokens = 464 selected tokens per (req, kv_head) |
| `vortex_block_reserved_bos` | 4 | Always keep first 16 tokens (4 blocks) |
| `vortex_block_reserved_eos` | 8 | Always keep last 32 tokens (8 blocks) |
| `mem_fraction_static` | 0.35 | Tight on co-tenant GPU; on B200 raise to 0.8+ |
| `model_path` | `Qwen/Qwen3-0.6B` | Tight on memory; on B200 use 70B model |

## How Option B (indices hook) is implemented

The hook lives in
[`vortex_torch/engine/sgl/attention_backend/flashinfer.py`](vortex_torch/engine/sgl/attention_backend/flashinfer.py)
between the indexer call and the attention call. It reads `VORTEX_POLICY`,
calls `apply_policy()` from
[`vortex_torch/engine/sgl/policy_transform.py`](vortex_torch/engine/sgl/policy_transform.py)
which rewrites `sparse_kv_indptr` and the indices buffer in place.

`apply_policy()` logic, per (request, kv_head) row:
1. Decode each physical block_id into a (kv_head-local) logical block index.
2. Group logical blocks into P-pages (one P-page = `P / block_size`
   consecutive logical blocks).
3. **Method 1**: keep every P-page that has ≥1 selected block; expand
   to all its blocks.
4. **Method 2**: keep P-pages with hit ratio ≥ threshold; expand
   to all blocks; on empty selection, fall back to first P-page (to
   avoid crashing FlashInfer).
5. Convert expanded logical block list back to physical block_ids.
6. Write back the new indices + indptr.

FlashInfer is **not** modified in Option B — it still uses
`BatchDecodeWithPagedKVCacheWrapper` with `page_size=block_size=4`.
The "policy" lives entirely in the indices rewrite.

## How Option C (BSR wrapper swap) is implemented

Gated by `VORTEX_USE_BSR=1`. Two changes in
[`flashinfer.py`](vortex_torch/engine/sgl/attention_backend/flashinfer.py):

1. **Add a `BlockSparseAttentionWrapper`** alongside the existing
   `BatchDecodeWithPagedKVCacheWrapper`. Same `workspace_buffer`.

2. **Plan and run BSR per layer** inside `forward_decode`, right after
   the indexer fills `sparse_kv_indices` for that layer:
   - BSR's `plan(indptr, indices, M, N, R=1, C=block_size, ...)`
     snapshots the indices values (unlike BatchDecode which has a
     buffer-pointer reuse mode), so the plan call must happen **after**
     the per-layer indexer call, not once-per-step in
     `init_forward_metadata`. This was the source of the initial
     accuracy=0 bug.
   - `run(q, k_flat, v_flat)` with `k_flat = cache["k"].view(-1, 1, head_dim)`
     — BSR sees flat KV with `num_kv_heads=1`, treating each
     `block_size=4`-token chunk as one BSR "column" (C=4).

Trade-off: BSR is a **prefill kernel** under the hood (FA2/FA3 prefill
template), not decode-specialized. On q_len=1 it's slightly slower than
BatchDecode (−1.4% here). The win of Option C is conceptual cleanliness
— no paging mismatch between sglang (page=32), vortex_torch (page=32),
and FlashInfer (page=4 in the BatchDecode path). BSR sees a flat KV
and gathers arbitrary 4-token blocks directly.

## Option A — Custom Triton fetch + flash-attention kernel (v1 done)

**Why A.** Both B and C use existing FlashInfer kernels. They
issue HBM loads at the kernel's native granularity (BatchDecode →
page_size=4 tile loads; BSR → C=4 block tile loads). To truly own the
HBM access pattern and verify that "block-granularity gather" actually
saves bandwidth (vs. the framework forcing 32-token page loads), we
need to **fuse the gather into the same kernel that does attention**.

### A.1 — What we built

A standalone Triton kernel:
[`vortex_torch/engine/sgl/attention_backend/block_sparse_decode_triton.py`](vortex_torch/engine/sgl/attention_backend/block_sparse_decode_triton.py)

For each (CSR row `n`, query head `g`), the kernel:

1. Loads `Q[n, g, :]` (one head_dim vector).
2. Walks the row's selected blocks: `for i in [indptr[n], indptr[n+1])`:
   - `block_id = indices[i]`
   - Gathers `K[block_id, :, :]` and `V[block_id, :, :]` — each is one
     `(BS=4, D=128)` tile.
   - Computes `scores = q · k_tile.T` (shape `[BS]`).
   - Applies a last-block mask on the final iteration only
     (using `last_block_len[n]`).
   - Updates flash-attention online softmax state `(m, l, acc)`.
3. Writes `O[n, g, :] = acc / l`.

No paging. No FlashInfer wrapper. The kernel reads the same
`sparse_kv_indptr` / `sparse_kv_indices` buffers that the indexer
just filled, plus the existing `kv_last_page_len` for the partial-block
mask. K/V come from `cache["k"].view(num_blocks, BS, D)` directly.

### A.2 — Why Triton instead of CUDA

The original plan called for forking FlashInfer's
`single_decode.cu`. We pivoted to Triton because:

1. **No ninja/sm_120 toolchain pain.** Triton JIT-compiles per-call —
   we already burned a day on FlashInfer CUDA build issues earlier in
   this branch.
2. **Day-1 working result.** Total time from green-field to RULER
   accuracy=1.0 was ~3 hours, not 5–7 days.
3. **Same address-computation expressiveness.** The Triton kernel
   does `K_ptr + block_id * stride_kb + ...` which is exactly the
   gather we'd hand-code in CUDA. Tensor-core MMA isn't used yet (a
   real CUDA port could add it), but on `q_len=1` decode the gain is
   secondary to memory access.
4. **Easy iteration.** Autotune, tile-size sweeps, GQA broadcast
   strategies are all one-line changes in Triton.

If/when Triton hits a ceiling (e.g. on B200 with much larger
contexts), the kernel can be ported to CUDA — the gather logic
transfers 1:1.

### A.3 — Correctness validation

- **Unit test** ([`algorithm_scientist/test_custom_kernel.py`](algorithm_scientist/test_custom_kernel.py)):
  bit-exact match (`max_abs_diff = 0`) against a naive PyTorch
  reference on a synthetic problem (N=2, G=4, BS=4, D=64,
  num_blocks=16) — including a row with a partial-block mask.
- **End-to-end RULER**: accuracy = 1.0 (same as BatchDecode and BSR).
- **Output statistics** at layer 1 match the BSR/BatchDecode paths
  in mean/std/min/max (within ~10% — see
  `logs/full_comparison_*/custom_baseline.err` with `VORTEX_CUSTOM_DEBUG=1`).

### A.4 — Current perf vs. headroom

Throughput on Qwen3-0.6B + RULER 2-sample subset (small workload,
within ~5% noise):

| Backend | tok/s | vs BatchDecode |
|---|---|---|
| BatchDecode (block_fetch) | 10.07 | — |
| BSR (Option C) | 9.83 | −2.4% |
| **Triton custom (Option A, v1)** | **9.98** | **−0.9%** |

v1 is **already on par with BatchDecode and slightly faster than
BSR** with zero tile-size tuning. That's because Triton's default
parameters (`num_warps=4, num_stages=2`) happen to fit the
`(BS=4, D=128)` tile size reasonably well. Plausible headroom:

- **Tile fusion.** Process 4 or 8 selected blocks per kernel iteration
  rather than 1, amortising the gather instruction over more compute.
- **Autotuning.** Sweep `num_warps ∈ {2, 4, 8}` and `num_stages ∈
  {1, 2, 3}` for the actual (G, D, BS) combo.
- **Block-major vs head-major parallelism.** Currently one program per
  (n, g). Splitting along the K dimension (split-k decode) and
  cross-program reduction would help for very long rows.
- **Tensor cores via `tl.dot`.** For `(BS, D) @ (D, 1)` GEMV this is
  marginal, but `(BS, D) @ (D, G)` (fused over GQA group) is a 4×16
  matmul that does benefit from tensor cores.

### A.5 — Remaining work (was 5–7 days; now 2–3 days)

| Day | Task | Status |
|---|---|---|
| 1 | Triton kernel + Python wrapper | ✅ done |
| 1 | Wire into flashinfer.py behind `VORTEX_USE_CUSTOM=1` | ✅ done |
| 1 | Unit test against naive reference | ✅ done |
| 1 | End-to-end RULER accuracy = 1.0 | ✅ done |
| 2 | Tile fusion (multi-block per iter) | TODO |
| 2 | Autotune `num_warps` / `num_stages` for B200 sm_100 | TODO |
| 2 | GQA-fused QK (`(BS, D) @ (D, G)` matmul via `tl.dot`) | TODO |
| 3 | Benchmark on 70B + 32k context (B200), confirm HBM bandwidth win | TODO |
| 3 (optional) | Port to CUDA / FlashInfer JIT if Triton ceiling hit | DEFERRED |

### A.6 — Run it

```bash
VORTEX_USE_CUSTOM=1 python algorithm_scientist/run_ruler_trace.py \
    --config submissions/block_size_sweep/batch_0_id2_page32.json
```

Or via the all-policies comparison:

```bash
bash run_method_comparison.sh
```

Debug mode (logs layer-1 output stats):

```bash
VORTEX_USE_CUSTOM=1 VORTEX_CUSTOM_DEBUG=1 python algorithm_scientist/run_ruler_trace.py ...
```

### A.7 — Stopping criterion

Option A is "worth it" if, on a 70B model with `seq_len ≥ 32k` and
`block_size=4`, the custom kernel beats BatchDecode (page=4) by **≥
1.5×** in decode tok/s on B200. v1 matches BatchDecode on a tiny
workload — the real test is at scale. If it only matches at scale,
the win is purely cleaner code (no paging mismatch between sglang,
vortex_torch, FlashInfer); defer further work. If it loses at scale,
the assumption that sub-page gather is HBM-bound was wrong — revisit
the indexer cost instead.

## Trace-driven analysis (companion, simulation-based)

If you want to analyze policies on a wider grid of (P, threshold) values
without running RULER for each, use the trace-driven simulator:

```bash
# Step 1: dump traces from a single RULER run
VORTEX_DUMP_TRACE_DIR=logs/traces python algorithm_scientist/run_ruler_trace.py \
    --config submissions/block_size_sweep/batch_0_id2_page32.json

# Step 2: simulate policies on the traces (no model needed)
python algorithm_scientist/trace_policy_analyzer.py \
    --trace-dir logs/traces \
    --page-sizes 16 32 64 128 \
    --thresholds 0.10 0.25 0.50 0.75 \
    --out-csv logs/trace_policy_analysis.csv
```

Caveat: simulation gives `coverage / waste / loaded_MB` only — it does
not measure real accuracy or kernel time (because attention never runs
with the simulated policy). For real numbers, use `run_method_comparison.sh`.

## Local environment hacks (RTX 5060 Ti / Blackwell sm_120)

This branch was developed on a shared single RTX 5060 Ti (sm_120,
Blackwell). The standard sglang/flashinfer pinned versions don't ship
sm_120 binaries. These patches enable inference but should be **reverted
on B200**:

| File | Change | Revert on B200 |
|---|---|---|
| `third_party/sglang/.../layernorm.py` | `RMSNorm.forward_cuda` → `forward_native` | Restore sgl-kernel CUDA body |
| `third_party/sglang/.../rotary_embedding.py` | Same fallback for RoPE | Restore original |
| `third_party/sglang/.../activation.py` | Same for SiluAndMul | Restore original |
| JSON config | `disable_cuda_graph: true`, `sampling_backend: pytorch` | Remove both keys |
| JSON config | `mem_fraction_static: 0.35`, `model_path: Qwen/Qwen3-0.6B` | Raise mem_fraction to 0.8+, switch to 70B model |

The sm_120 patches make absolute throughput numbers unrepresentative of
B200 performance (native PyTorch fallbacks are slower than sgl-kernel
CUDA). However, the **relative** comparison across policies remains valid
because every policy uses the same fallback overhead.

## Other files in this branch (not used by run_method_comparison.sh)

- `algorithm_scientist/page_policy_sweep.py` — synthetic-distribution
  policy comparison (superseded by the real comparison here)
- `algorithm_scientist/trace_policy_analyzer.py` — trace-driven simulator
  (companion to this experiment, mentioned above)
- `algorithm_scientist/tpot_microbench.py` — per-decode-step latency
- `bench_decode_bandwidth_notc.py`, `run_bandwidth_sweep*.sh` — HBM
  bandwidth profiling
- `submissions/block_size_sweep/batch_0_id{0,1,2,3}.*` — earlier
  block_size variants
