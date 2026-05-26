# bench1: sub-page block-level KV fetch for vortex_torch

Comparing three implementations (A / B / C) of block-granularity
(BS=4) KV gather on top of vortex_torch's `block_sparse_attention`,
plus two index-rewrite policies (Method 1 / Method 2) layered on B.

## Reproduce on B200

Prereqs: vortex_torch + sglang installed, a 70B-class model checkpoint.

```bash
# 1. Revert the sm_120 / co-tenant workarounds (this branch was developed
#    on RTX 5060 Ti). On B200, restore:
#      - third_party/sglang/.../layernorm.py        : forward_native → forward_cuda
#      - third_party/sglang/.../rotary_embedding.py : forward_native → forward_cuda
#      - third_party/sglang/.../activation.py       : forward_native → forward_cuda
#    and in submissions/block_size_sweep/batch_0_id2_page32.json:
#      - remove "disable_cuda_graph": true
#      - remove "sampling_backend": "pytorch"
#      - mem_fraction_static : 0.35 → 0.85
#      - model_path          : "Qwen/Qwen3-0.6B" → "Qwen/Qwen2.5-72B-Instruct"
#      - vortex_max_seq_lens : 6144 → 32768  (or whatever your prompts need)

# 2. Run the 7-config sweep (B + C + A). Expect ~10-15 min total.
bash run_method_comparison.sh

# 3. Read results
cat logs/method_comparison_*/summary.tsv
```

Per-config debugging / tracing:

```bash
# pick one backend
VORTEX_POLICY=block_fetch         python algorithm_scientist/run_ruler_trace.py --config <cfg>   # Option B
VORTEX_POLICY=method1_p32         python ...                                                     # Method 1
VORTEX_POLICY=method2_p32_t50     python ...                                                     # Method 2, threshold 0.5
VORTEX_USE_BSR=1                  python ...                                                     # Option C
VORTEX_USE_CUSTOM=1               python ...                                                     # Option A (auto: v2 if G≥8, else v1)
VORTEX_USE_CUSTOM=1 VORTEX_CUSTOM_KERNEL_VERSION=v1   python ...
VORTEX_USE_CUSTOM=1 VORTEX_CUSTOM_KERNEL_VERSION=v2   python ...
VORTEX_USE_CUSTOM=1 VORTEX_CUSTOM_KERNEL_VERSION=cuda python ...

# strict numerical check (logs per-layer MAE vs BatchDecode)
VORTEX_USE_CUSTOM=1 VORTEX_CUSTOM_VALIDATE=1 python ...

# HBM bytes accounting (dumps JSON at the given path, flushed every 32 calls)
VORTEX_HBM_TRACE=logs/hbm.json VORTEX_USE_CUSTOM=1 python ...
```

Standalone kernel test (no sglang import):

```bash
python algorithm_scientist/test_custom_kernel.py
```

## Metrics

- **accuracy**: RULER substring-match score on the magic-UUID needle-in-haystack task. Range [0, 1].
- **throughput** (tok/s): end-to-end tokens generated per second, including prefill + decode + sampling.
- **avg idx / call**: average number of CSR indices the kernel gathers per layer × decode step (= number of selected 4-token blocks). Bigger = more HBM work.
- **KV MB / call**: HBM bytes the kernel pulls per layer × decode step ≈ `idx_count × BS × head_dim × 2 (K+V) × sizeof(bf16)`. The directly-measurable proxy for memory bandwidth.
- **per-layer MAE** (with `VORTEX_CUSTOM_VALIDATE=1`): max / mean absolute diff vs FlashInfer BatchDecode on identical indices. Sanity-checks numerical correctness of the custom path.

## The three options

All three load only the 4-token blocks the indexer selects. The
difference is *which kernel implements the gather*.

- **Option B** — index-transformation hook. FlashInfer's
  `BatchDecodeWithPagedKVCacheWrapper(page_size=4)` is unchanged; an
  in-flight hook in `flashinfer.py` rewrites `sparse_kv_indices`
  before the kernel runs. Lets us swap fetch *policies* without
  touching CUDA. Implements `block_fetch` (passthrough), `method1_p32`
  (lossless page expansion), and `method2_p32_t{25,50,75}` (lossy
  page filter at hit-ratio threshold TT/100).

- **Option C** — BSR wrapper swap. Drop the BatchDecode kernel for
  FlashInfer's `BlockSparseAttentionWrapper(C=4)`. No paging
  mismatch between layers; uses a prefill-style kernel. Toggled by
  `VORTEX_USE_BSR=1`.

- **Option A** — custom kernel. We own the K/V load loop. Three
  implementations behind `VORTEX_USE_CUSTOM=1`:
  - `v1` Triton — one program per `(row, query head)`, one block per
    inner iter. Simple.
  - `v2` Triton — one program per row, `BLOCK_BLOCKS` blocks per
    iter, GQA-fused QK + PV via `tl.dot` (tensor cores). autotune'd.
    Auto-selected only when `G ≥ 8`.
  - `cuda` — hand-written CUDA port of v1. JIT-compiled via
    `torch.utils.cpp_extension.load_inline`. Falls back to Triton v1
    if the build fails.

## Method 1 / Method 2 (layered on Option B)

- **Method 1** (`method1_p32`): if any 4-token block in a 32-token
  page is selected, load **all 8** blocks of that page. Lossless
  over-fetch.
- **Method 2** (`method2_p32_t{TT}`): load a 32-token page only when
  its hit ratio ≥ TT/100, otherwise drop. Lossy.

## Files

- `vortex_torch/engine/sgl/attention_backend/flashinfer.py` — backend dispatcher
- `vortex_torch/engine/sgl/policy_transform.py` — Method 1 / 2 index rewriter
- `vortex_torch/engine/sgl/attention_backend/block_sparse_decode_triton.py` — Option A Triton (v1, v2)
- `vortex_torch/engine/sgl/attention_backend/block_sparse_decode_cuda.py` — Option A CUDA
- `vortex_torch/engine/sgl/hbm_trace.py` — HBM bytes tracer
- `run_method_comparison.sh` — 7-config sweep driver
- `algorithm_scientist/test_custom_kernel.py` — standalone kernel unit test
