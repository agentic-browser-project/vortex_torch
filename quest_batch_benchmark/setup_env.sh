#!/usr/bin/env bash
# Create the uv venv for the Quest batch benchmark on vortex_torch v0.5.
#
# v0.5 is pure-Python + Triton JIT -- there is NO vortex_torch_C CUDA extension
# to build (csrc/ and setup.py were removed in v0.5).
#
# The serving backend is the vendored sglang v0.5.9, which pins its own
# internally-consistent stack in its base dependencies:
#   torch==2.9.1, torchaudio==2.9.1, torchao==0.9.0, sgl-kernel==0.3.21,
#   flashinfer_python==0.6.3, flashinfer_cubin==0.6.3, torch_memory_saver==0.0.9
# torch 2.9.1's PyPI wheel ships sm_100 cubins and sgl-kernel 0.3.x is the
# Blackwell-capable series -- so this stack runs on a B200 (sm_100) as-is.
# We install sglang's BASE deps only (NOT the [all] extra: that pulls
# diffusion/tracing packages -- diffusers, moviepy, opencv, ... -- that this
# decode benchmark never uses, and dragging them in is what desynced an
# earlier attempt's torch).
#
# Re-runnable: clears and recreates the venv if it already exists.
set -euo pipefail

REPO=/vast/projects/liuv/pennnetworks/xutingl/vortex_torch/.worktrees/quest-batch-benchmark-v0.5
BENCH="$REPO/quest_batch_benchmark"
VENV="$BENCH/.venv"
SGLANG="$REPO/third_party/sglang/v0.5.9/sglang"

command -v uv >/dev/null || { echo "ERROR: 'uv' is not on PATH -- install uv first." >&2; exit 1; }
[ -d "$SGLANG/python" ] || { echo "ERROR: sglang v0.5.9 not found at $SGLANG" >&2; exit 1; }

echo "[1/4] create uv venv (python 3.12)"
uv venv --python 3.12 --clear "$VENV"

echo "[2/4] install sglang v0.5.9 (editable, base deps only)"
# Let uv resolve sglang's base dependency set as a whole -- torch 2.9.1,
# sgl-kernel 0.3.21, flashinfer 0.6.3 etc. are pinned together and consistent.
# Do NOT pre-install or pre-pin torch: a stale torch pin desyncs the stack and
# leaves a half-upgraded, broken torch.
uv pip install --python "$VENV" -e "$SGLANG/python"

echo "[3/4] install vortex_torch v0.5 (editable, pure python, --no-deps)"
# --no-deps: vortex_torch's pyproject pins (torch>=2.7, transformers==4.57.1,
# lighteval, inspect-ai, ...) would otherwise re-resolve and disturb the
# sglang stack. vortex_torch v0.5 is pure python and only needs the sglang
# stack that step 2 already installed.
uv pip install --python "$VENV" --no-deps -e "$REPO"

echo "[4/4] install benchmark harness deps"
# The harness itself needs pytest (unit tests) and pandas (aggregation).
# transformers / accelerate / einops arrive with the sglang stack.
uv pip install --python "$VENV" pytest pandas

echo "[verify] stack imports and sees the B200"
"$VENV/bin/python" - <<'PY'
import torch, sglang, vortex_torch
print("torch  ", torch.__version__, "cuda_ok", torch.cuda.is_available())
print("device ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("sglang ", getattr(sglang, "__version__", "?"))
print("vortex_torch import OK")
PY
echo "[done] venv at $VENV"
