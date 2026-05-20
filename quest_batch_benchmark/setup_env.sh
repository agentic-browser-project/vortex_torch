#!/usr/bin/env bash
# Create the uv venv for the Quest batch benchmark and install the full stack.
# Re-runnable (re-creates the venv from scratch if --clear is needed).
#
# DEVIATION FROM SPEC (torch 2.7.1+cu126):
#   torch 2.7.1+cu126 does NOT run on B200 (sm_100): it emits
#   "no kernel image is available" at runtime.  PTX-JIT fallback was dropped
#   in the cu126 wheel.  We use torch 2.7.1+cu128 instead, which ships the
#   sm_100 cubin images and keeps the same Python ABI as 2.7.1+cu126 (so
#   sgl-kernel==0.2.4 and flashinfer-python==0.2.7.post1 continue to load).
#   The CUDA extension is compiled with the spack CUDA 12.8 toolchain which
#   can target sm_100 natively (no PTX JIT needed).
set -euo pipefail

REPO=/vast/projects/liuv/pennnetworks/xutingl/vortex_torch
BENCH="$REPO/quest_batch_benchmark"
VENV="$BENCH/.venv"
# CUDA 12.8 nvcc from spack — supports sm_100 (B200) natively.
# Used only to BUILD the vortex_torch_C CUDA extension.
BUILD_CUDA_HOME=/vast/parcc/spack/sw/apps/linux-sapphirerapids/cuda-12.8.1-lmm74gnqr2pl2dzbtfjdwoo3fnwbar43
TORCH_INDEX=https://download.pytorch.org/whl/cu128

cd "$REPO"

echo "[1/6] create uv venv (python 3.12)"
uv venv --python 3.12 "$VENV"

echo "[2/6] install torch 2.7.1+cu128 (sm_100 kernel images for B200)"
uv pip install --python "$VENV" \
  --index-url "$TORCH_INDEX" \
  torch==2.7.1 torchvision==0.22.1

echo "[3/6] install the bundled-stack wheels (pinned to the vortex conda env)"
# sgl-kernel==0.2.4 and flashinfer-python==0.2.7.post1 are binary abi3 wheels
# compiled against torch 2.7.x; they remain ABI-compatible with 2.7.1+cu128.
uv pip install --python "$VENV" \
  sgl-kernel==0.2.4 \
  flashinfer-python==0.2.7.post1 \
  flashinfer-cubin==0.6.8.post1 \
  triton==3.3.1 \
  numpy==2.3.5 \
  pandas \
  orjson \
  uvloop \
  xgrammar==0.2.0 \
  "huggingface-hub[cli]" \
  pytest

echo "[4/6] install the patched sglang fork (editable, deps resolved)"
uv pip install --python "$VENV" -e "$REPO/third_party/sglang/python"

echo "[5/6] build + install vortex_torch (editable, CUDA extension)"
# Build against torch 2.7.1+cu128 headers.
# TORCH_CUDA_ARCH_LIST="10.0" targets B200 (sm_100) natively via nvcc 12.8.
# Note: setup.py _normalize_single_arch does not accept the "+PTX" suffix
# used by stock PyTorch; use plain "10.0" which maps to (10,0) = sm_100.
# --no-deps prevents uv from resolving vortex_torch's install_requires
# (which includes torchao, lighteval, etc.) and overwriting our cu128 torch.
CUDA_HOME="$BUILD_CUDA_HOME" \
PATH="$BUILD_CUDA_HOME/bin:$PATH" \
TORCH_CUDA_ARCH_LIST="10.0" \
  uv pip install --python "$VENV" --no-deps -e "$REPO" --no-build-isolation

echo "[6/6] done — venv at $VENV"
