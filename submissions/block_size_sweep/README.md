# Block-size sweep (4 variants) — Results & Setup Notes

Purpose: empirically test whether vortex_torch's existing block-fetch
infrastructure works at `block_size < page_size` (which in vortex_torch
means shrinking the physical KV page itself — see
[api.py:48](../../vortex_torch/engine/sgl/api.py#L48)), and measure the
accuracy/throughput tradeoff as fetch granularity gets finer.

All four variants use the **same algorithm** (centroid + GeMV + topK,
identical to `example_block_sparse_attention.py`) and keep the **same
total selected-token budget** (~512 per decode step). Only `block_size`
and the proportionally-scaled `topk_val` / `reserved_bos` / `reserved_eos`
differ.

| Variant | block_size | topk_val | reserved_bos | reserved_eos | total selected | mem_fraction |
|---------|------------|----------|--------------|--------------|----------------|---------------|
| A (id0) | 16         | 29       | 1            | 2            | 32 × 16 = 512  | 0.80          |
| B (id1) | 8          | 58       | 2            | 4            | 64 × 8  = 512  | 0.80          |
| C (id2) | 4          | 116      | 4            | 8            | 128 × 4 = 512  | 0.80          |
| D (id3) | 1          | 464      | 16           | 32           | 512 × 1 = 512  | 0.70          |

`vortex_topk_ratio = 0.0625` for all (fraction-based, scales naturally).

## Results — RULER quality gate (≥0.85)

| Variant | block_size | max_seq_lens used | RULER accuracy | Status |
|---|---|---|---|---|
| **A** | 16 | 20480 (default) | **0.99** | ✅ PASS |
| **B** | 8  | 20480 (default) | **1.00** | ✅ PASS |
| **C** | 4  | **16384** (reduced) | **1.00** | ✅ PASS (only with reduced max_seq) |
| **D** | 1  | — | — | ❌ BLOCKED (see "Hard ceiling" below) |

**Conclusion**: Sub-page block fetch infrastructure is algorithmically
correct down to `block_size=4` (accuracy holds at 1.0 on RULER for B and
C). Below that, a framework limit prevents testing.

## Hard ceiling — `topk` kernel limits `max_seq_lens / block_size ≤ 4096`

vortex_torch's `topk` kernel
([custom_ops/topk_output/flashinfer/default/kernel.cu:174-186](../../vortex_torch/custom_ops/topk_output/flashinfer/default/kernel.cu#L174-L186))
has a hardcoded dispatch tower with a hard `TORCH_CHECK(false, "topk:
max_num_pages > 4096 not supported")` ceiling. The argument it receives
is `ctx.max_num_blocks_per_request = max_seq_lens / block_size` (because
vortex_torch forces `page_size = block_size`, so `num_blocks_per_page = 1`).

Consequence at default `vortex_max_seq_lens=20480`:

| Variant | block_size | max_seq / block_size | Within 4096? |
|---|---|---|---|
| A | 16 | 1280 | ✅ |
| B | 8  | 2560 | ✅ |
| C | 4  | **5120** | ❌ — need `max_seq ≤ 16384` |
| D | 1  | **20480** | ❌ — need `max_seq ≤ 4096`, but RULER inputs are ~4500 tok |

So:
- **AIME24** (needs `max_seq_lens = 20480`): only A and B fit
- **RULER** (max input ~4500 tok): A, B, C fit; D doesn't (its 4096-token
  ceiling is below RULER's minimum input)
- **D cannot be tested on any meaningful workload** without rebuilding
  the topk kernel with a higher dispatch ceiling.

To raise the ceiling, extend the dispatch tower in
`kernel.cu:174-186` with additional `LAUNCH_TOPK(threads, blocks)`
entries beyond 4096, and rebuild sgl-kernel from source.

## Results — TPOT microbench (AIME24 was too slow on this consumer GPU)

AIME24's protocol (30 problems × 16 trials × up to 16K tokens generated)
proved infeasible on RTX 5060 Ti + native PyTorch fallback paths:
~10 hours per variant. Replaced with a tight TPOT micro-benchmark in
[algorithm_scientist/tpot_microbench.py](../../algorithm_scientist/tpot_microbench.py):
single prompt of ~1646 tokens, 128 generated tokens, 3 reps, median.

| Variant | block_size | median TPOT (ms/tok) | throughput (tok/s) | vs A baseline |
|---------|------------|----------------------|--------------------|---------------|
| A (id0) | 16 | **89.12** | 11.2 | — |
| B (id1) | 8  | **86.13** | 11.6 | **3.4% faster** |
| C (id2) | 4  | **86.70** | 11.5 | **2.7% faster** |
| D (id3) | 1  | (blocked — see above) | — | — |

Per-rep numbers (very consistent, ±2%):

| Variant | rep 0 | rep 1 | rep 2 |
|---------|------|------|------|
| A | 91.66 ms | 89.12 ms | 89.07 ms |
| B | 89.56 ms | 86.13 ms | 86.04 ms |
| C | 88.59 ms | 86.16 ms | 86.70 ms |

### Interpretation

1. **Block-level sub-page fetch does reduce TPOT, but only marginally
   on this setup** (~3% A→B). The trend is real (B's rep 1/2 beat all of
   A's reps), but the magnitude is small.
2. **B and C are essentially tied** — going from block_size=8 to 4
   buys nothing extra in this configuration.
3. **Why the gain is small here**:
   - Short prompt (~1.6K tokens) → KV bandwidth is a smaller fraction
     of total decode work
   - Native PyTorch fallbacks for RMSNorm/rotary/silu/sampling dominate
     per-step cost; bandwidth savings get masked
   - No CUDA graph → per-step Python overhead also dominates
   - At default `topk_ratio=0.0625` and small budget (~512 tokens), the
     attention compute itself is already small relative to other costs
4. **Where to look for bigger effects**: longer context (8K+ tokens),
   tighter topk so KV reads dominate, production setup with sgl-kernel
   CUDA ops + CUDA graph, and a card with proper sm_120 sgl-kernel
   binaries (not this RTX 5060 Ti workaround).

## Environment notes (2026-05-24 setup on RTX 5060 Ti / sm_120 Blackwell)

The standard sglang/flashinfer pinned versions don't ship full sm_120
support. The patches that got things running:

1. **CUDA 12.9 toolkit** (conda) — flashinfer warns
   `SM 12.x requires CUDA >= 12.9`; CUDA 12.8 alone won't JIT.
2. **Symlink** `lib64 → lib` in the env — flashinfer's JIT link step
   uses `-L$CUDA_HOME/lib64` which conda's CUDA layout doesn't create.
3. **PyTorch 2.7.1+cu128** — does include sm_120 in arch list; works.
4. **sgl-kernel 0.2.4** — NO sm_120 binaries. Three custom ops fail at
   runtime: `RMSNorm`, `apply_rope_with_cos_sin_cache_inplace`,
   `top_k_top_p_sampling_from_probs`. Patches applied in this repo:
   - `third_party/.../layers/layernorm.py`: `RMSNorm.forward_cuda` →
     `self.forward_native(x, residual)`
   - `third_party/.../layers/rotary_embedding.py`: `forward_cuda` →
     `self.forward_native(...)`
   - `third_party/.../layers/activation.py`: `SiluAndMul.forward_cuda` →
     `self.forward_native(x)` (preventive)
   - All 4 JSON configs: `"sampling_backend": "pytorch"` to bypass
     the failing top-k sampling kernel via sglang's built-in fallback.
   - All 4 JSON configs: `"disable_cuda_graph": true` (sgl-kernel
     RMSNorm is also called during graph capture).
5. **Upgrading sgl-kernel to 0.3.x** — has sm_120 prebuilt but ABI
   incompatible with PyTorch 2.7.1 (`_ZN3c104cuda9SetDeviceEab`
   undefined symbol). Would need PyTorch ≥ 2.8.

**Performance caveat**: The native-PyTorch fallback paths are slower
than sgl-kernel's CUDA kernels. Absolute throughput will be lower
than a production sgl-kernel + CUDA-graph setup. But the relative
comparison across A/B/C is still valid (all variants use identical
fallback overhead).

## How to run

```bash
# env vars (one-time per shell)
export VORTEX_ENV_ROOT="$HOME/miniforge3/envs/vortex_v04"
export CUDA_HOME="$VORTEX_ENV_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
PY="$VORTEX_ENV_ROOT/bin/python"

# Pre-flight (~10 sec)
for y in 0 1 2 3; do
  $PY -c "from vortex_torch.engine.sgl import check_engine_config; \
    check_engine_config('submissions/block_size_sweep/batch_0_id${y}.json')"
done

# RULER (~2-3 min each)
LOGDIR=logs/submission/block_size_sweep; mkdir -p $LOGDIR
for y in 0 1 2; do  # D blocked
  CUDA_VISIBLE_DEVICES=0 $PY algorithm_scientist/run_ruler.py \
    --config "submissions/block_size_sweep/batch_0_id${y}.json" \
    > "$LOGDIR/ruler_id${y}.out" 2> "$LOGDIR/ruler_id${y}.err"
done

# AIME24 (~20-60 min each); only A and B fit max_seq=20480
for y in 0 1; do
  CUDA_VISIBLE_DEVICES=0 $PY algorithm_scientist/run_submission_aime24.py \
    --config "submissions/block_size_sweep/batch_0_id${y}.json" \
    > "$LOGDIR/aime_id${y}.out" 2> "$LOGDIR/aime_id${y}.err"
done

# Collect
for y in 0 1 2 3; do
  echo "=== id${y} ==="
  cat summary_ruler_submissions/block_size_sweep/batch_0_id${y}/latest.json 2>/dev/null
  cat summary_submissions/block_size_sweep/batch_0_id${y}/latest.json       2>/dev/null
done
```

## Diagnostic interpretation

| Outcome | Diagnosis |
|---|---|
| RULER ✓ at all tested block_size, AIME24 throughput **↑** as bs ↓ | Sub-page works, kernel overhead doesn't dominate — task succeeds; next step: raise topk ceiling for bs ∈ {1, 2} |
| RULER ✓, AIME24 throughput **↓** as bs ↓ | Sub-page logically works but per-kernel overhead dominates the bandwidth savings — need to switch decode kernel (BlockSparseAttentionWrapper, custom Triton, etc.) |
| RULER ✓, AIME24 **same** | Bandwidth not the current bottleneck on this workload/GPU — re-test at larger batch / longer context |

## Files

- `batch_0_id0.{py,json}` — A: block_size=16 (baseline)
- `batch_0_id1.{py,json}` — B: block_size=8
- `batch_0_id2.{py,json}` — C: block_size=4 (note `vortex_max_seq_lens=16384`)
- `batch_0_id3.{py,json}` — D: block_size=1 (cannot run; documented for reference)
