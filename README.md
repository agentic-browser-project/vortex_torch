# bench1: Block Fetch vs Method 1 vs Method 2

A **real** comparison (not simulation) of three KV-fetch policies on
vortex_torch's `block_sparse_attention` upstream, with `block_size=4`
and `page_size=32`.

## The three policies

| Policy | What it does | Granularity |
|---|---|---|
| **Block Fetch** | Load only the selected 4-token blocks (vortex_torch's actual current behavior) | block_size=4 |
| **Method 1** (`method1_pP`) | If any selected block falls inside a P-token page, load ALL blocks of that page | P=16 or 32 |
| **Method 2** (`method2_pP_tTT`) | Load a P-page only when hit ratio ≥ TT/100, otherwise drop | P=16/32, TT=25/50/75 |

For Method 1 / Method 2 to differ from Block Fetch, `P` must be a strict
multiple of `block_size`. Method 1 is lossless (recall=1.0 always);
Method 2 may drop content (recall < 1).

**This is real, not simulation**: an in-flight hook in
[`flashinfer.py`](vortex_torch/engine/sgl/attention_backend/flashinfer.py)
rewrites the `sparse_kv_indices` buffer after the indexer fills it but
before the attention kernel runs. The attention call actually sees the
transformed indices.

## Headline result

On vortex_torch's `block_sparse_attention` algorithm (block_size=4) +
RULER 2-sample subset, **Block Fetch is the fastest** at 7.88 tok/s,
while every Method 1/2 variant is ~7.5–9.0% slower. All policies preserve
accuracy=1.0 on this task.

| Policy | Accuracy | Throughput (tok/s) | vs Block Fetch |
|---|---|---|---|
| **block_fetch** | 1.0 | **7.88** | — |
| method1_p16 | 1.0 | 7.26 | −7.9% |
| method1_p32 | 1.0 | 7.29 | −7.5% |
| method2_p16_t50 | 1.0 | 7.22 | −8.4% |
| method2_p32_t50 | 1.0 | 7.24 | −8.1% |
| method2_p32_t25 | 1.0 | 7.17 | −9.0% |

**Caveat**: the policy transformation hook runs in Python (CPU↔GPU
roundtrip per attention call), which contributes most of the ~8% slowdown.
A production implementation would move this to a CUDA kernel and shrink
the gap. The relative ordering (Block Fetch fastest) is still meaningful,
but the absolute throughput delta overstates the true HBM-bandwidth cost.

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

Expected time on B200: ~10–15 min (6 policies × ~2 min each).

Override Python interpreter:
```bash
PYTHON=/path/to/env/bin/python bash run_method_comparison.sh
```

What the script does:
1. For each of 6 policies (`block_fetch`, `method1_p16`, `method1_p32`,
   `method2_p16_t50`, `method2_p32_t50`, `method2_p32_t25`):
   - Sets `VORTEX_POLICY` env var to that policy
   - Runs RULER on a 2-prompt subset with vortex_torch's
     `block_sparse_attention` algorithm
2. Aggregates `(accuracy, throughput)` into a single TSV summary

## Output

`logs/method_comparison_<timestamp>/`:
- `summary.tsv` — one row per policy, with `accuracy`, `throughput`, status
- `<policy>.out` / `<policy>.err` — per-policy logs
- `<policy>_result.json` — full RULER summary per policy

### How to read the results table

| Column | Meaning |
|---|---|
| **policy** | The fetch policy name (e.g. `method1_p32` = Method 1 with page_size=32) |
| **accuracy** | RULER substring-match accuracy: did the model find the magic UUID? Range [0, 1] |
| **throughput** | Tokens/sec generated end-to-end (includes prefill, decode, sampling) |
| **rc** | Return code (`OK` / `FAIL`) |

Expected pattern: Block Fetch ≥ Method 1 ≥ Method 2 in throughput; all
should preserve accuracy on this easy needle-in-haystack task. If
accuracy drops for Method 2 at high threshold, the policy is dropping
too much context.

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

## How the policies are actually implemented

The hook lives in
[`vortex_torch/engine/sgl/attention_backend/flashinfer.py`](vortex_torch/engine/sgl/attention_backend/flashinfer.py)
between the indexer call and the attention call. It reads `VORTEX_POLICY`,
calls `apply_policy()` from
[`vortex_torch/engine/sgl/policy_transform.py`](vortex_torch/engine/sgl/policy_transform.py)
which rewrites `sparse_kv_indptr` and the indices buffer in place.

`apply_policy()` logic, per (request, kv_head) row:
1. Decode each physical block_id into a (kv_head-local) logical block index
2. Group logical blocks into P-pages (one P-page = `P / block_size`
   consecutive logical blocks)
3. **Method 1**: keep every P-page that has ≥1 selected block; expand
   to all its blocks
4. **Method 2**: keep P-pages with hit ratio ≥ threshold; expand
   to all blocks; on empty selection, fall back to first P-page (to
   avoid crashing FlashInfer)
5. Convert expanded logical block list back to physical block_ids
6. Write back the new indices + indptr

FlashInfer is **not** modified — it still uses
`BatchDecodeWithPagedKVCacheWrapper` with `page_size=block_size=4`.
The "policy" lives entirely in the indices rewrite.

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
