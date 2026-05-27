# CUDA-graph capability — per-method status

This benchmark exposes a `--enable-cuda-graph` flag on its `sgl.Engine` runs.
Whether the flag actually has an effect — or is even acceptable — depends on
the attention method. This note records the per-method status so the
two-category README ("no-graph" and "CUDA graph") cites a single source.

## dense (sgl.Engine via flashinfer backend) — CUDA graph supported

`sgl.Engine` natively supports CUDA-graph capture of the decode step on the
flashinfer backend. The harness's `disable_cuda_graph=not args.enable_cuda_graph`
flag flows through both engine constructors (`build_engine_kwargs` and
`build_get_engine_kwargs`) and is the documented sglang knob for this.

## quest (sgl.Engine + vortex sparsity) — CUDA graph status TBD-by-smoke

Quest's static-block-budget sparse attention runs through the vortex sparsity
backend. vortex v0.5's `get_engine` hardcodes `disable_cuda_graph=True` as
its built-in default, which means the configuration is not its tested path.
Whether overriding to `disable_cuda_graph=False` actually works at decode
time is verified empirically by the smoke test in Task 2 of the
"cuda-graph-second-category" plan, and the outcome is recorded in the
"Quest + CUDA graph — smoke result" section below.

## TreeSparseAttention (standalone HuggingFace+FlashInfer harness) — CUDA graph NOT supported

TreeSparseAttention is a separate project at
`/vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention`.
It is **not** a vortex/sglang plugin: it has its own Python 3.13 venv, its
own pre-built CUDA kernels, and its own decode harness in `benchmark_batch.py`
that drives a per-step Python loop over layers. The harness as it stands
cannot be wrapped in a `torch.cuda.CUDAGraph` — the decode step contains
multiple host-blocking operations, dynamic Python control flow, and a
FlashInfer plan() that is explicitly configured as graph-disabled.

### Concrete evidence (re-confirm with the greps in Task 1, Step 1)

1. **No torch CUDA-graph code anywhere.** `grep -rn "torch.cuda.graph\|CUDAGraph"`
   over `python/`, `models/`, and `benchmark_batch.py` returns no matches.
   TreeSparse never captures or replays a CUDA graph.
2. **FlashInfer plan() is hardcoded graph-disabled.**
   `models/direct_decode.py` calls FlashInfer's plan with
   `False,  # is_cuda_graph_enabled` as the argument that tells FlashInfer
   whether the kernel will be replayed from a captured graph. Switching that
   flag would also require switching every other graph-incompatible site
   below.
3. **Per-step `torch.cuda.synchronize()` brackets every decode step.**
   `benchmark_batch.py`'s decode loop calls `torch.cuda.synchronize()`
   before `time.perf_counter()` and again after `runner.decode_step(...)`.
   Host-side sync is graph-incompatible: a captured graph cannot contain a
   host-blocking barrier.
4. **Per-step host-blocking token extraction.** After each `decode_step`,
   `benchmark_batch.py` does `sampled = torch.argmax(logits, dim=-1)`
   followed by `token_ids = sampled.tolist()` — a `.tolist()` is a
   GPU-to-host copy that forces a sync. The next step then writes those
   Python ints back into `tok_buf` via `tok_buf[r, 0] = token_ids[r]`,
   another series of host-driven scalar writes. Both patterns are forbidden
   inside a captured CUDA graph.
5. **Per-step Python loop over layers.** `models/direct_decode.py`'s
   `decode_step` iterates `for layer_id, lw in enumerate(self.layer_weights)`
   inside the timed decode step. While the body is GPU work, the loop
   itself is Python control flow that fixes the number of layers into the
   graph at capture time — survivable, but combined with the other
   graph-blockers above the loop is irrelevant: capture cannot proceed.

### What it would take to add CUDA-graph support to TreeSparse

Out of scope for this plan, but for the record: the harness would need a
graph-compatible decode step (no per-step `.tolist()` — keep sampled tokens
on GPU and feed them back as a tensor; no per-step `synchronize()` — replace
with `cuda.Event` timing or external timing of the replay window), a
FlashInfer plan re-issued with `is_cuda_graph_enabled=True`, and a wrapping
`torch.cuda.graph(...)` capture/replay around the steady-state portion of
the decode loop (the first step still runs eager to populate the cache, the
remainder replay the graph). That is a meaningful rewrite of TreeSparse's
own harness, not a tweak.

### Consequence for this benchmark

The CUDA-graph category is a **two-way comparison (dense, quest)** only.
TreeSparse is left in the no-graph three-way table only. Reusing TreeSparse's
no-graph number in the CUDA-graph table would mix categories and mislead the
reader, so the tables stay separate.

## Quest + CUDA graph — smoke result

*Filled in by the Task 2 smoke gate.*
