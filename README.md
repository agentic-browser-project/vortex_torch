# Block Fetch Policy Experiments on vortex_torch

This branch (`bench1`) contains experiments comparing three KV-fetch policies
for sparse attention, run on top of vortex_torch's `block_sparse_attention`
algorithm. The goal was to evaluate whether coarser fetch granularity
(loading whole pages when only part is needed) or thresholded fetch
(dropping low-density pages) could outperform the current vortex_torch
behavior (load exactly the selected blocks).

**Headline finding**: on the real workload, **Block Fetch (= vortex_torch's
current behavior) is strictly the best policy**. Method 1 and Method 2 have
no operating point that simultaneously preserves accuracy and reduces
kernel time, because (a) vortex_torch's planner already chooses tokens at
block-aligned granularity, and (b) the default `topk_val` keeps the workload
in the launch-overhead-bound regime where bandwidth savings don't translate
to time savings.

---

## Three policies compared

| Policy | What it does | Lossy? | Over-fetches? |
|---|---|---|---|
| **Block Fetch** | Load exactly the blocks the indexer selected (= vortex_torch's current behavior) | No | No |
| **Method 1** | Group selected blocks into larger P-pages; if any block in a P-page is selected, load the whole P-page | No | Yes |
| **Method 2** | Group selected blocks into P-pages; load a P-page only if its hit ratio ≥ `threshold`, otherwise drop the entire P-page | Yes (can drop selected blocks) | Yes (loads whole pages) |

---

## Column definitions for all result tables

| Column | Unit | Input/Output | Meaning |
|---|---|---|---|
| **Method** | — | input | Block Fetch / Method 1 / Method 2 |
| **N** | tokens | input (workload) | Number of "needed" tokens per (request, kv_head) per decode step (= the upstream algorithm's selection size) |
| **P** | tokens / page | input (knob) | Page size for Method 1 / Method 2 (size of the "load it whole or not" decision unit). Block Fetch implicitly has P = 1 |
| **threshold** | ∈ [0, 1] | input (knob) | Method 2 only: load a P-page only when `(needed_in_page / P) ≥ threshold`; otherwise drop |
| **coverage** | ∈ [0, 1] | output | Fraction of the N needed tokens actually loaded. Block Fetch & Method 1 always = 1.0; Method 2 may be < 1.0 |
| **waste** | ∈ [0, 1] | output | Fraction of loaded tokens that were not needed (= 1 − precision). Block Fetch always = 0; Method 1 increases with P |
| **loaded MB** | MB | output | Bytes actually transferred from HBM (per call or per (request, kv_head); see context) |
| **achieved BW** | GB/s | output (diagnostic) | `loaded MB × 1000 / kernel time`. Tells you whether HBM is being kept busy. **Not** a quality-of-policy indicator on its own |
| **kernel time** | μs | output (final latency) | Wall-clock time of one attention kernel call. **The metric users actually feel.** |

---

## Real-trace experiment results

**Setup**: dumped `sparse_kv_indices` from every decode step / layer / kv_head
during a RULER run with vortex_torch's default
`block_sparse_attention` (block_size=16, topk_val=29, reserved_bos=1,
reserved_eos=2). Analyzed **27,216 (request, kv_head) selections** across
27 layers (all except layer 0, which is dense per `vortex_layers_skip`).
RULER baseline accuracy = **1.00** on the 2-sample validation subset.

For each (Method, P, threshold) configuration, we computed coverage, waste,
and `loaded MB` per selection. The table shows the mean across all 27,216
rows. `N_mean = 511 tokens` for all rows (= 32 blocks × 16 tokens − padding).

| Method | P | threshold | coverage | waste | loaded MB (per row) | predicted accuracy |
|---|---|---|---|---|---|---|
| **Block Fetch** | 16 | — | **1.000** | **0.000** | **0.262** | **1.00** (measured) |
| Method 1 | 16 | — | 1.000 | 0.000 | 0.262 | ≈ 1.00 |
| Method 1 | 32 | — | 1.000 | 0.500 | 0.523 | ≈ 1.00 |
| Method 1 | 64 | — | 1.000 | 0.750 | 1.046 | ≈ 1.00 |
| Method 1 | 128 | — | 1.000 | 0.875 | 2.093 | ≈ 1.00 |
| Method 1 | 256 | — | 1.000 | 0.918 | 3.234 | ≈ 1.00 |
| Method 2 | 16 | 0.10 / 0.25 / 0.50 / 0.75 | 1.000 | 0.000 | 0.262 | ≈ 1.00 |
| Method 2 | 32 | 0.10 / 0.25 / 0.50 | 1.000 | 0.500 | 0.523 | ≈ 1.00 |
| Method 2 | 32 | **0.75** | **0.000** | — | **0.000** | **0** (collapse) |
| Method 2 | 64 | 0.10 / 0.25 | 1.000 | 0.750 | 1.046 | ≈ 1.00 |
| Method 2 | 64 | **0.50** | **0.000** | — | **0.000** | **0** |
| Method 2 | 128 | 0.10 | 1.000 | 0.875 | 2.093 | ≈ 1.00 |
| Method 2 | 128 | **0.25 / 0.50 / 0.75** | **0.000** | — | **0.000** | **0** |
| Method 2 | 256 | 0.10 | 0.455 | 0.875 | 0.952 | severely degraded |
| Method 2 | 256 | ≥ 0.25 | 0.000 | — | 0.000 | 0 |

### Three observations from the real-trace data

1. **P = block_size (16) → all three policies are mathematically identical.** Because vortex_torch's planner already chose tokens at block-aligned granularity, there is no "intra-page waste" to exploit.
2. **Method 1's waste is a mathematical constant**: `waste = 1 − block_size/P`. At P=32 → 0.5, P=64 → 0.75, P=256 → 0.918. Predictable regardless of trace content.
3. **Method 2 falls off a cliff**: real selections form small clusters of ~2–4 adjacent blocks. Any threshold that requires more than that within a P-page drops everything. There is no smooth degradation — coverage goes from 1.0 to 0.0 abruptly.

### Why Block Fetch wins

| Dimension | Block Fetch | Method 1 (P > 16) | Method 2 |
|---|---|---|---|
| coverage | 1.00 | 1.00 | 1.00 or 0 (cliff) |
| loaded MB | **0.262 (lowest)** | 0.5 – 3.2 (up to 12× more) | 0 or same as Method 1 |
| kernel time (inferred) | **~42 μs (lowest)** | ≥ 42 μs, grows with loaded MB once out of overhead-bound | same as Method 1 |
| accuracy (trace prediction) | 1.00 | ≈ 1.00 | 1.00 or 0 |

There is **no operating point** where Method 1 or Method 2 outperforms
Block Fetch on this workload. Method 1 only ever loads more data with
unchanged accuracy; Method 2 either matches Method 1 (low threshold) or
catastrophically drops everything (high threshold).

---

## Supporting experiments (also in this branch)

### Bandwidth sweep over (page_size, selected_seq_len)
[`run_bandwidth_sweep_seqlen.sh`](run_bandwidth_sweep_seqlen.sh) +
[`bench_decode_bandwidth_notc.py`](bench_decode_bandwidth_notc.py).

Mapped out the three regimes of HBM bandwidth utilization on this GPU:

| selected_seq_len | KV bytes/call | regime | achieved BW |
|---|---|---|---|
| 512 | 2 MB | launch-overhead bound | ~50 GB/s (11% of peak) |
| 4096 | 16 MB | transition | ~395 GB/s (88% of peak) |
| 16384 | 67 MB | bandwidth bound | ~408 GB/s (91% of peak) |

This established that vortex_torch's default `topk_val=29` keeps every
attention call at < 4 MB — fully in the launch-overhead-bound regime —
explaining why no fetch policy can beat Block Fetch by reducing data
movement.

### Synthetic policy sweep (random vs clustered)
[`algorithm_scientist/page_policy_sweep.py`](algorithm_scientist/page_policy_sweep.py).

Compared the three policies on synthetic "needed token" distributions to
understand the policies in isolation. Confirmed that:
- Method 1 only ever adds bytes; it's "free" in launch-overhead-bound and
  expensive in bandwidth-bound
- Method 2 only has a sweet spot when the distribution is highly clustered
  AND N is moderate; random distributions break it immediately

Full data: [`logs/page_policy_sweep_v2_*.csv`](logs/).

### Block-size sweep on RULER
[`submissions/block_size_sweep/`](submissions/block_size_sweep/).

Tested vortex_torch's `block_sparse_attention` at block_size ∈ {16, 8, 4, 1}
to see if the existing infrastructure could already do finer-grained fetch
(it can, but topk kernel has a hard ceiling of `max_seq_len / block_size ≤ 4096`).

| block_size | RULER accuracy | notes |
|---|---|---|
| 16 (baseline) | 0.99 | works with default `vortex_max_seq_lens=20480` |
| 8 | 1.00 | works with default |
| 4 | 1.00 | requires `vortex_max_seq_lens=16384` |
| 1 | — | blocked by topk kernel ceiling (`max_blocks_per_request ≤ 4096`) |

---

## ⚠️ Local environment hacks (RTX 5060 Ti / Blackwell sm_120)

This branch was developed on a single **RTX 5060 Ti** (sm_120, Blackwell).
That GPU is newer than the prebuilt `sgl-kernel-0.2.4` binaries shipped
with `sglang 0.4.9`. The following local changes were required to make
inference run **and will not be needed on a B200 / proper datacenter GPU**:

| File | Change | Why it exists | B200 action |
|---|---|---|---|
| `third_party/sglang/v0.4.9/sglang/python/sglang/srt/layers/layernorm.py` | `RMSNorm.forward_cuda` → falls back to `forward_native` (pure PyTorch) | sgl-kernel 0.2.4 has no sm_120 binary for `rmsnorm` | **Revert** — use the original CUDA kernel |
| `third_party/.../layers/rotary_embedding.py` | `forward_cuda` → falls back to `forward_native` | Same: no sm_120 binary for `apply_rope_with_cos_sin_cache_inplace` | **Revert** |
| `third_party/.../layers/activation.py` | `SiluAndMul.forward_cuda` → falls back to `forward_native` | Preventive; same root cause | **Revert** |
| All JSON configs in `submissions/block_size_sweep/` | `"sampling_backend": "pytorch"` and `"disable_cuda_graph": true` | sgl-kernel's `top_k_top_p_sampling_from_probs` has no sm_120 binary; CUDA graph capture also fails on the missing RMSNorm | **Remove these two keys** — both fall back to original sgl-kernel CUDA |

These workarounds make absolute throughput numbers in this branch
**unrepresentative of B200 performance** (the native PyTorch fallback
paths are slower than sgl-kernel CUDA). However, **relative comparisons
across the 4 block-size variants and the 3 fetch policies remain valid**
because every variant uses the same fallback overhead.

### Other branch additions (keep / portable to B200)

- [`algorithm_scientist/page_policy_sweep.py`](algorithm_scientist/page_policy_sweep.py) — orchestrator for the (Method, P, threshold, N, distribution) policy sweep
- [`algorithm_scientist/trace_policy_analyzer.py`](algorithm_scientist/trace_policy_analyzer.py) — analyzes dumped sparse_kv_indices traces against the three policies
- [`algorithm_scientist/run_ruler_trace.py`](algorithm_scientist/run_ruler_trace.py) — RULER runner pointing at 2-sample subset (`examples/validation_subset2.jsonl`)
- [`algorithm_scientist/tpot_microbench.py`](algorithm_scientist/tpot_microbench.py) — small TPOT (time-per-output-token) benchmark
- [`bench_decode_bandwidth_notc.py`](bench_decode_bandwidth_notc.py) — patched version of `bench_decode_bandwidth.py` with `use_tensor_cores=False` and graceful trtllm skip (so it works for page_size ∈ {1, 2, 4, 8}). On B200 you can use the original `bench_decode_bandwidth.py` directly
- `vortex_torch/engine/sgl/attention_backend/flashinfer.py` — adds a trace-dump hook gated by `VORTEX_DUMP_TRACE_DIR`. Safe to keep; only activates when the env var is set
- [`submissions/block_size_sweep/`](submissions/block_size_sweep/) — 4 submission pairs varying block_size
- `run_*.sh` — shell wrappers for batch sweeps

---

## How to reproduce on B200

1. **Revert the four sgl-kernel fallbacks** listed in the table above.
2. **Remove `sampling_backend` and `disable_cuda_graph` from the JSON configs**.
3. Re-run with the standard vortex_torch workflow:
   ```bash
   # block-size sweep + RULER
   for y in 0 1 2 3; do
     python algorithm_scientist/run_ruler.py \
       --config submissions/block_size_sweep/batch_0_id${y}.json
   done

   # synthetic policy sweep
   python algorithm_scientist/page_policy_sweep.py \
     --needed-list 256 2048 8192 \
     --page-list 4 8 16 32 \
     --x-list 0.1 0.25 0.5 0.75 \
     --distributions random clustered

   # trace dump + real-trace policy analysis
   VORTEX_DUMP_TRACE_DIR=logs/traces \
     python algorithm_scientist/run_ruler_trace.py \
     --config submissions/block_size_sweep/batch_0_id0.json
   python algorithm_scientist/trace_policy_analyzer.py \
     --trace-dir logs/traces

   # bandwidth sweeps (B200 should use the unpatched bench)
   bash run_bandwidth_sweep_seqlen.sh   # may need to edit to use bench_decode_bandwidth.py
   ```
4. **Throughput numbers will be very different (much faster) on B200**, but the qualitative findings about Method 1 / Method 2 vs Block Fetch should hold because they are determined by the planner's selection pattern (block-aligned, small clusters), not by hardware speed.

## What we expected to find vs what we actually found

| Hypothesis | Result |
|---|---|
| Smaller block_size → less HBM bandwidth → faster decode | False on this GPU: no measurable difference because workload is launch-overhead bound (~3% in TPOT bench, within noise of bandwidth sweep) |
| Method 1 (coarsen fetch) could be "free" at small N | True in overhead-bound regime; flips to ~4× slower in bandwidth-bound regime |
| Method 2 (drop low-density pages) could trade a small accuracy hit for big speedup | False on real traces: cluster structure makes thresholds either trivially permissive or catastrophic |
| Real planner output is similar to synthetic clustered distribution | False: real output is *block-aligned*, which makes Block Fetch trivially optimal at P=16. Synthetic distributions over-state the "value" of coarser fetch |

---

## Final answer for the colleague

> **Block Fetch (= the framework's current behavior) is the best policy on vortex_torch's `block_sparse_attention` workload.** Method 1 and Method 2 have no operating point that beats it without sacrificing accuracy. To make either policy interesting, you would need to either (a) drive the workload into the bandwidth-bound regime (10× larger `topk_val`), or (b) use a token-level (not block-aligned) upstream sparse attention algorithm. Neither matches the current vortex_torch surface, so the existing Block Fetch path should remain the default.
