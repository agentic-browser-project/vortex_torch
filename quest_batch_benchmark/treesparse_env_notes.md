# TreeSparseAttention environment — verified for the three-way comparison

**Verified:** 2026-05-22 (Task 1 of the TreeSparse three-way comparison plan).

TreeSparseAttention is a separate project with its own environment. The
three-way benchmark does NOT rebuild it — it activates the existing env.

## Location & activation

- Project: `/vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention`
- venv: `<project>/.venv` (Python 3.13.2)
- Activation: `cd <project> && source env.sh` (loads Lmod modules, then
  activates `.venv`). A non-interactive script must first source the Lmod
  init — `run_treesparse.sh` does this.

## Verified stack (observed)

- python: 3.13.2 (`/vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention/.venv/bin/python`)
- torch: 2.11.0+cu128, CUDA available: True, device: NVIDIA B200
- flashinfer: 0.6.9
- transformers: 5.7.0
- `_tree_sparse_kernels` C++ extension: imports OK (pre-built at
  `build/_tree_sparse_kernels.cpython-313-x86_64-linux-gnu.so`)

## Smoke test

`benchmark_batch.py` on the shared `quest_batch_benchmark/request.json`
(`--batch-sizes 1 --num-decode-tokens 8 --repeat 1 --page-size 64 --no-kv-sharing`):
PASSED — prefill_len 9661, produced a valid result JSON with
`tpot_median_ms` 137.17 ms.

## Note on stack divergence from the quest benchmark

TreeSparse uses torch 2.11.0; the quest/dense `sgl.Engine` path uses
torch 2.9.1. Each method requires its own engine and cannot share a venv.
This is a documented, unavoidable difference — see README.md.
