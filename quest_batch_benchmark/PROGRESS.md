# Quest Batch Benchmark — Progress / Handoff

**Status:** ✅ COMPLETE — all 9 tasks done, benchmark run, final review approved.
**Last updated:** 2026-05-20.
**Branch:** `quest-batch-benchmark` (repo `/vast/projects/liuv/pennnetworks/xutingl/vortex_torch`, forked from `v0.3` @ `6825ff4`).

Built by executing `IMPLEMENTATION_PLAN.md` task-by-task with the
**superpowers:subagent-driven-development** workflow (per task: implementer
subagent → spec-compliance review → code-quality review → fix loop), then a
final whole-implementation review.

---

## Result

`results/tpot_vs_batchsize.csv` — mean decode TPOT (ms/token), Qwen3-8B,
`request_005` (9,661 input tokens), Quest `topk_val=64`:

| batch | dense | quest | speedup |
|------:|------:|------:|--------:|
| 1  | 5.62  | 5.54 | 1.02× |
| 2  | 5.99  | 5.73 | 1.05× |
| 4  | 6.84  | 6.09 | 1.12× |
| 8  | 8.74  | 6.43 | 1.36× |
| 16 | 11.94 | 6.93 | 1.73× |
| 32 | 19.23 | 8.54 | 2.25× |
| 64 | OOM   | OOM  | —     |

Quest decode TPOT scales far better with batch size than dense attention —
2.25× faster at batch 32. Both modes OOM at batch 64 (recorded `status=oom`).

---

## Task status

| Task | State | Commits |
|------|-------|---------|
| 1. uv venv + stack | ✅ DONE — revised for B200, see below | `aebc440` … `4c6f79d` |
| 2. Download Qwen3-8B | ✅ DONE (model lives outside the repo) | — |
| 3. Scaffold + `prompt_io.py` | ✅ DONE — spec + code-quality reviewed | `81d494c` |
| 4. Benchmark harness | ✅ DONE — reviewed; doc fixes + crash bugfix | `ab361a6`, `5312869`, `099151e` |
| 5. Smoke gate | ✅ PASSED — dense→FlashInfer, quest→VTXGraph backends | `4c6f79d` |
| 6. Results aggregator | ✅ DONE — spec + code-quality reviewed | `11567be` |
| 7. Driver script | ✅ DONE | `dfb9829` |
| 8. Full benchmark run | ✅ DONE — `mem_fraction_static` tuned, results committed | `0be1162`, `6aeb33e` |
| 9. README | ✅ DONE | `8e388a5` |

Final whole-implementation review: **approved** (only minor cleanups, applied).

---

## ⚠️ Major deviation — B200 environment rework

The plan's "known-good" stack (the `vortex` conda env: torch 2.7.1+cu126,
sgl-kernel 0.2.4) is **Hopper-only and cannot run on this B200 cluster**.
sgl-kernel 0.2.x ships no sm_100 kernels and no PTX — `rmsnorm` fails with
"no kernel image is available for execution on the device". The whole cluster
is B200 (`dgx-b200` partition, dgx001–029); no Hopper nodes exist. Fixed
(commit `4c6f79d`):

- torch 2.7.1 → **2.8.0+cu128**, torchvision → 0.23.0, triton → 3.4.0.
- sgl-kernel 0.2.4 → **0.3.17.post1** (first series with an `sm100/` build;
  it requires torch 2.8.0).
- `vortex_torch_C` rebuilt for sm_100 against torch 2.8.0.
- `benchmark_quest_tpot.py` sets `CUDA_HOME` + `PATH` (ninja, nvcc) at import,
  so flashinfer's and vortex's runtime JIT compilation works.

`setup_env.sh` reflects the new pins. The CUDA 12.8 toolkit used to build
`vortex_torch_C` is the spack install
`/vast/parcc/spack/sw/apps/linux-sapphirerapids/cuda-12.8.1-lmm74gnqr2pl2dzbtfjdwoo3fnwbar43`.

Two other in-flight fixes:
- A `build_server_args` crash — `parse_known_args([])` exited the process
  because `ServerArgs` makes `--model-path` required (commit `099151e`).
- `--mem-fraction-static` default 0.9 → **0.6** (commit `0be1162`): at 0.9 the
  static KV pool is over-allocated (~144 GB) and large-batch prefill
  activations OOM at batch ≥ 16; 0.6 keeps enough KV for batch 64 while
  freeing room for the prefill, so the sweep reaches batch 32.

---

## How to re-run

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch/quest_batch_benchmark
GPU=0 bash run_benchmark.sh        # dense + quest sweeps, then aggregate
.venv/bin/python -m pytest         # 8 unit tests
```

The venv (`quest_batch_benchmark/.venv`) and the Qwen3-8B checkpoint
(`/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B`) are already set
up. To recreate the venv: `bash setup_env.sh`. The harness sets `CUDA_HOME`/
`PATH` itself — no external env vars needed beyond `CUDA_VISIBLE_DEVICES`/`GPU`.

## User decisions baked into the benchmark
- Model: Qwen3-8B (text twin of Qwen3-VL-8B). Dense baseline included.
- KV cache stays bf16; any batch size that OOMs is recorded `status=oom`,
  the ascending sweep stops.
- Engine booted with `disable_radix_cache=True` so KV-cache reuse cannot make
  the quest-vs-dense comparison unfair.
