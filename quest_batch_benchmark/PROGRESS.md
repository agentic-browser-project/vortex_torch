# Quest Batch Benchmark — Progress / Resume Handoff

**Last updated:** 2026-05-20, end of session (node stopping).
**Branch:** `quest-batch-benchmark` (in repo `/vast/projects/liuv/pennnetworks/xutingl/vortex_torch`, forked from `v0.3` @ `6825ff4`).

This benchmark is being built by executing `IMPLEMENTATION_PLAN.md` task-by-task with
the **superpowers:subagent-driven-development** workflow: per task, dispatch an
implementer subagent → spec-compliance review → code-quality review → fix loop → next task.

---

## Status at pause

| Task | State | Commits |
|------|-------|---------|
| 1. uv venv + stack | ✅ DONE — implemented, spec-reviewed, code-quality-reviewed, fixes applied | `aebc440`, `23facdc`, `6692b48` |
| 2. Download Qwen3-8B | ✅ DONE — downloaded + verified (no commit; model lives outside the repo) | — |
| 3. Scaffold + `prompt_io.py` + tests | ⚠️ IMPLEMENTED & COMMITTED, **reviews NOT done yet** | `81d494c` |
| 4–9 | ⬜ not started | — |

`IMPLEMENTATION_PLAN.md` is committed alongside this file. Tasks 4–9 are fully specified there.

---

## ▶️ How to resume (next session)

1. `cd /vast/projects/liuv/pennnetworks/xutingl/vortex_torch && git checkout quest-batch-benchmark`
2. Re-invoke the **superpowers:subagent-driven-development** skill.
3. **First action: finish Task 3's reviews** (they were not run before the pause):
   - Dispatch a spec-compliance reviewer for Task 3 (commit `81d494c`, base `6692b48`).
   - Then a code-quality reviewer.
   - Apply any fixes. Note the Task 3 open concern below.
4. Then continue with Tasks 4 → 9 from `IMPLEMENTATION_PLAN.md`, one at a time.
5. Finish with the whole-implementation code review + `superpowers:finishing-a-development-branch`.

### Task 3 open concern (must be checked during its review)
The implementer reported **DONE_WITH_CONCERNS**. Deviations from the plan's literal code,
both believed correct but unverified by review:
- `prompt_io.py` — the plan's `apply_chat_template(tokenize=True)` returns a `BatchEncoding`
  (not a list) under the installed `transformers==5.8.1`. The implementer added
  `return_tensors=None` plus a `hasattr(ids, "input_ids")` unwrap guard. Verify this is
  correct and minimal.
- `quest_batch_benchmark/.gitignore` gained negation rules (`!request.json`, `!tests/`,
  `!tests/**`) because the repo-root `.gitignore` was otherwise hiding those paths. Verify.
- Tests pass: `test_prompt_io.py` 2/2 green. **`request_005` tokenizes to 9,661 tokens**
  (Task 9's README needs this number; the plan estimated ~8K).

---

## Environment facts (already set up — do NOT redo)

- **uv venv:** `quest_batch_benchmark/.venv` — use `quest_batch_benchmark/.venv/bin/python`
  for everything. Created by `quest_batch_benchmark/setup_env.sh` (re-runnable).
- **Stack actually installed** (deviates from the plan, intentionally — see below):
  `torch==2.7.1+cu128` (NOT cu126 — cu126 lacks B200/sm_100 kernels and errors with
  "no kernel image available"), sgl-kernel 0.2.4, flashinfer-python 0.2.7.post1,
  the patched sglang fork 0.4.9 (editable, `third_party/sglang`), `vortex_torch` +
  `vortex_torch_C` CUDA extension (built for `sm_100` / `TORCH_CUDA_ARCH_LIST=10.0`
  using a CUDA 12.8 toolkit), transformers 5.8.1, numpy 2.3.5, xgrammar 0.2.0.
- **B200 CUDA gate passed** — torch runs on the dedicated B200 (`dgx006`, sm_100).
- **Model:** `/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B`
  (`Qwen3ForCausalLM`, 36 layers, 8 KV heads — the text twin of Qwen3-VL-8B; the
  framework's sglang fork has no `qwen3_vl` support, hence the substitution).
- **sglang submodule** has a committed one-line `hasattr` guard patch in
  `python/sglang/srt/utils.py` (harmless with the final torch).
- Known benign warnings at import: `pynvml` deprecation; `NVCC Compiler not found,
  use NVRTC for DeepGEMM JIT` (affects both quest and dense equally — relative TPOT
  comparison unaffected; consider putting nvcc on PATH for Task 8 if absolute numbers
  matter).

## User decisions baked into the plan
- Model: Qwen3-8B (text twin). Include a **dense baseline** alongside Quest.
- KV cache stays **bf16**; any batch size that OOMs is recorded `status=oom`, sweep stops.
- Engine booted with `disable_radix_cache=True` so KV-cache reuse cannot make the
  quest-vs-dense comparison unfair.

## Notes / risks for remaining tasks
- **Task 5 is a hard gate:** it verifies vortex sparsity actually activates through
  `sglang.bench_one_batch`. If it fails, pivot to the `sgl.Engine` delta-method
  fallback documented in `IMPLEMENTATION_PLAN.md` → "Risks & Contingencies" → R2.
- Task 1 used `--no-deps` installs with a hand-curated sglang runtime dep list; if a
  later task hits a missing-module ImportError, add the package (pinned) to
  `setup_env.sh` step 4 and re-run.
