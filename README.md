# bench1: KV-fetch policy comparison on vortex_torch

This branch compares three KV-fetch policies — **Block Fetch** (load only the
selected blocks, vortex_torch's current behavior), **Method 1** (load whole
page if any block in it is selected, lossless over-fetch), **Method 2**
(load whole page only if hit ratio ≥ threshold, drop otherwise) — against
the real selection patterns produced by vortex_torch's `block_sparse_attention`.

**Headline finding**: on the real workload, Block Fetch is strictly the best.
Method 1 over-fetches without speedup; Method 2 either matches Method 1 or
catastrophically drops content. See `logs/trace_policy_analysis.csv` after
running the experiment for the data behind this claim.

---

## Recommended setup for B200

| Item | Recommendation |
|---|---|
| GPU | NVIDIA B200 (this branch was developed on RTX 5060 Ti; numbers will be very different but the qualitative findings should hold) |
| Model | **A ~70B-class model**, e.g. `Qwen/Qwen2.5-72B-Instruct` or `meta-llama/Llama-3.1-70B-Instruct`. vortex_torch is most thoroughly tested with the Qwen family; if you stick with Qwen3 the biggest dense option is `Qwen/Qwen3-32B`. |
| How to switch model | Add `"model_path": "Qwen/Qwen2.5-72B-Instruct"` to each `submissions/block_size_sweep/batch_0_id*.json`, **or** edit `MODEL_PATH` in [`vortex_torch/engine/sgl/api.py:22`](vortex_torch/engine/sgl/api.py#L22) |

---

## Setup time (one-time, B200, normal network)

| Step | Approx. time |
|---|---|
| 1. Create conda env + Python 3.12 (`conda create -n vortex_v04 python=3.12`) | 1–2 min |
| 2. PyTorch 2.7.1 + cu128 (~3 GB wheel) | 3–5 min |
| 3. `flashinfer-python==0.6.8.post1` | < 1 min |
| 4. sglang editable install (`bash third_party/sglang/v0.4.9/sglang/install.sh`) — pulls + compiles sgl-kernel | 5–10 min |
| 5. vortex_torch editable install (`pip install -e .`) | < 1 min |
| 6. **70B model download (~140 GB safetensors)** | **15 min – 2 h** (network-dependent) |
| 7. First-run JIT compile (FlashInfer attention kernels) | 30 sec – 2 min on first attention call |

Total ~30 min compute + 15 min–2 h model download. Subsequent runs reuse the model + JIT cache.

---

## Reverting the 4 local sm_120 (RTX 5060 Ti / Blackwell) workarounds on B200

These patches let inference run on a card whose architecture has no
prebuilt sgl-kernel binary. B200 has full sm_100 support, so revert all four:

### (a) `third_party/sglang/v0.4.9/sglang/python/sglang/srt/layers/layernorm.py`
The block marked `# PATCH (block_size_sweep, sm_120)` inside `RMSNorm.forward_cuda`
replaces the body with `return self.forward_native(x, residual)`. **Restore the
original body** (calls `fused_add_rmsnorm` / `rmsnorm` from sgl-kernel) — search
for "PATCH (block_size_sweep, sm_120)" in the file.

### (b) `third_party/sglang/v0.4.9/sglang/python/sglang/srt/layers/rotary_embedding.py`
Same pattern in `forward_cuda`: replace `return self.forward_native(...)` with
the original two-branch implementation (sgl-kernel `apply_rope_with_cos_sin_cache_inplace`
when `head_size ∈ {64,128,256,512}`, vllm fallback otherwise).

### (c) `third_party/sglang/v0.4.9/sglang/python/sglang/srt/layers/activation.py`
Same pattern in `SiluAndMul.forward_cuda`: replace native fallback with the
sgl-kernel `silu_and_mul(x, out)` call.

### (d) Remove sm_120 workarounds from the 4 submission JSONs
In `submissions/block_size_sweep/batch_0_id{0,1,2,3}.json`, delete:
```json
"disable_cuda_graph": true,
"sampling_backend": "pytorch"
```

The fastest way is `git diff bench1~1 -- third_party/sglang/v0.4.9 submissions/`
on the original (pre-bench1) tree to see the exact deltas, then reverse them.

---

## Run the experiment

```bash
bash run_all_experiments.sh
```

This runs three stages back-to-back. Expected total time on B200 + 70B + warm
caches: **~20–35 min**.

| Stage | What | Time on B200 (70B) | Output |
|---|---|---|---|
| 1 | Bandwidth regime sweep (page_size × selected_seq_len, no model) | 3–5 min | `logs/bandwidth_sweep_seqlen_*/` |
| 2 | Synthetic policy sweep (random + clustered distributions, no model) | 10–15 min | `logs/page_policy_sweep.csv` |
| 3 | Real-trace experiment (RULER 2-sample run + trace analyzer) | 5–15 min | `logs/trace_policy_analysis.csv` |

Override the Python interpreter if needed:
```bash
PYTHON=/path/to/your/env/bin/python bash run_all_experiments.sh
```

---

## What each output file tells you

- **`logs/bandwidth_sweep_seqlen_*/summary.tsv`**: at what `selected_seq_len` does HBM bandwidth become the bottleneck? On RTX 5060 Ti this happens around 4096 tokens. Will be different on B200.
- **`logs/page_policy_sweep.csv`**: each policy's (coverage, waste, loaded_MB, kernel_time) under synthetic random vs clustered token distributions.
- **`logs/trace_policy_analysis.csv`** ← **the conclusion**: same metrics under vortex_torch's actual `block_sparse_attention` selection patterns. Confirms Block Fetch is the optimum and shows where Method 1 / Method 2 break down.

For the detailed methodology and column definitions, see
[`submissions/block_size_sweep/README.md`](submissions/block_size_sweep/README.md).

---

## Single-step reproduction (if you want pieces individually)

```bash
PY=python   # or path to your env python

# Stage 1
bash run_bandwidth_sweep_seqlen.sh

# Stage 2
$PY algorithm_scientist/page_policy_sweep.py \
    --needed-list 256 2048 8192 --page-list 4 8 16 32 \
    --x-list 0.10 0.25 0.50 0.75 \
    --distributions random clustered \
    --out-csv logs/page_policy_sweep.csv

# Stage 3
VORTEX_DUMP_TRACE_DIR=logs/traces $PY \
    algorithm_scientist/run_ruler_trace.py \
    --config submissions/block_size_sweep/batch_0_id0.json
$PY algorithm_scientist/trace_policy_analyzer.py \
    --trace-dir logs/traces --out-csv logs/trace_policy_analysis.csv
```
