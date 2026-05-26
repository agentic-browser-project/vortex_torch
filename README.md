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
| **A** — custom CUDA fetch kernel | Replace FlashInfer entirely with a hand-written tiled-gather + flash-attention kernel that fuses sub-page gather into the K/V load loop | 5–10 days | **PLAN ONLY** (see end of this file) |

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
| **block_fetch** (baseline) | 1.0 | **10.30** | — |
| method1_p32 | 1.0 | 10.42 | +1.2% |
| method2_p32_t25 | 1.0 | 9.97 | −3.2% |
| method2_p32_t50 | 1.0 | 10.45 | +1.5% |
| method2_p32_t75 | 1.0 | 10.43 | +1.3% |
| **bsr_baseline** (Option C) | 1.0 | 10.16 | −1.4% |

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

## Option A — Custom CUDA fetch + flash-attention kernel (PLAN ONLY)

**Why A is needed.** Both B and C use existing FlashInfer kernels. They
issue HBM loads at the kernel's native granularity (BatchDecode →
page_size=4 tile loads; BSR → C=4 block tile loads). Real HBM bandwidth
savings from "block-granularity gather" would require fusing the gather
into the same kernel that computes attention, with a memory access
pattern designed for sub-page tiles. This is a multi-day effort.

### A.1 — Anatomy of what we want

The kernel should, for each (request, kv_head) row, perform:

```
for each selected 4-token block_id in this row:
    load 4 tokens × head_dim of K from cache[k]   (one tile load)
    load 4 tokens × head_dim of V from cache[v]   (one tile load)
    accumulate q·k.T into online-softmax state
    accumulate softmax·v into output
write o
```

This is **textbook flash-attention** with one twist: K/V are gathered
by an indices list, not stored contiguously. The indices list is the
already-existing `sparse_kv_indices` buffer.

### A.2 — Where to put the kernel

Two reasonable implementation paths:

| Path | Reuse base | Effort | Risk |
|---|---|---|---|
| **A-FlashInfer** | Fork `flashinfer/csrc/single_decode.cu` and add a `gather_kv` path that reads block_ids from an indptr+indices buffer instead of computing block_ids from contiguous offsets | 5–7 days | low (kernel + JIT plumbing already proven) |
| **A-FlashAttn** | Start from `flash-attn`'s decode kernel (cutlass) and add the same gather | 7–10 days | medium (we'd be vendoring more of flash-attn) |

**Recommend A-FlashInfer.** vortex_torch is already wired into FlashInfer;
the JIT pipeline, plan/run protocol, and head-dim instantiations are in
place. The smallest possible diff: add a new kernel name (e.g.
`block_sparse_decode`) alongside the existing `BatchDecodeWithPagedKVCacheWrapper`,
copy-paste the decode kernel, and replace the page-block-offset address
computation with `indices[indptr[row]+i] * block_size * head_dim` lookups.

### A.3 — Day-by-day plan

| Day | Task | Deliverable |
|---|---|---|
| 1 | Read `flashinfer/csrc/decode/decode_kernel.cuh` (or wherever decode tiles are emitted in your FlashInfer version). Identify the *exact* line where K/V global addresses are computed from page_table and `kv_indices`. | Annotated diff plan |
| 2 | Copy decode kernel to `block_sparse_decode_kernel.cuh`. Replace the page address computation with `indices[i] * (block_size * num_kv_heads * head_dim)`. Keep block_size as a kernel template parameter (default 4). | Kernel compiles, single-row unit test |
| 3 | Add the Python wrapper (mirror `BatchDecodeWithPagedKVCacheWrapper` API). `plan()` records indptr layout only — indices are read at run() from a registered buffer pointer (same trick BatchDecode uses for cuda graph). | `BlockSparseDecodeWrapper` class with `.plan()` + `.run()` |
| 4 | Wire into `vortex_torch/engine/sgl/attention_backend/flashinfer.py` behind `VORTEX_USE_CUSTOM=1`. Mirror the BSR integration (Option C) but with the new wrapper. | RULER runs end-to-end with custom kernel |
| 5 | Match BatchDecode accuracy bit-exactly on the same indices. Tune tile sizes for block_size=4 (the existing decode kernel may be tuned for block_size=16 or 32). | Profile-guided tile size |
| 6 | Benchmark on B200 vs BatchDecode and BSR. Verify HBM bandwidth savings show up in `nsys` profile (lower HBM bytes read per decode step). | Benchmark report |
| 7 (slack) | Polish: cuda graph support, multi-stream, FA3 path | Production-ready |

### A.4 — What can go wrong

- **Register pressure with small tiles.** block_size=4 is unusually
  small. The existing decode kernel is likely tuned for `BLOCK_N=64`
  or `128`. Going to 4 may waste warp-level parallelism. Tile-fusing
  (process 16 blocks per warp = 64 tokens) is the natural fix but
  requires more thought on the gather pattern.
- **L2 cache thrashing.** Gathering 4-token blocks from arbitrary
  positions has worse spatial locality than reading contiguous 32-token
  pages. May need a software-prefetching pass.
- **GQA layout.** Q has `group_size` heads per KV head; the kernel must
  broadcast K/V across the group. This is standard but the existing
  BatchDecode does it one way and BSR another — the custom kernel
  needs to pick one and be explicit.
- **Last-block partial fill.** The last block of the sequence has
  `last_page_len ≤ block_size` valid tokens. The mask logic must zero
  out the invalid positions in the softmax. BatchDecode handles this
  with `kv_last_page_len`; we'd need to thread that through too.

### A.5 — Stopping criterion

Option A is "worth it" if, on a 70B model with `seq_len ≥ 32k` and
`block_size=4`, the custom kernel beats BatchDecode (page=4) by **≥
1.5×** in decode tok/s on B200. If it only matches, the win is purely
cleaner code — defer to a future quarter. If it loses, the assumption
that sub-page gather is HBM-bound was wrong; revisit the indexer cost
instead.

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
