#!/usr/bin/env bash
# Create the uv venv for the Quest batch benchmark and install the full stack.
# Re-runnable (clears and recreates the venv if it already exists).
#
# DEVIATION FROM SPEC (the plan's torch 2.7.1+cu126 stack is Hopper-only):
#   The plan mirrored the `vortex` conda env (torch 2.7.1+cu126, sgl-kernel
#   0.2.4). That stack CANNOT run on a B200 (sm_100):
#     - torch 2.7.1+cu126 has no sm_100 cubins.
#     - sgl-kernel 0.2.4 ships kernels for sm_75..sm_90a only (no sm_100,
#       no PTX) -- its rmsnorm etc. fail with "no kernel image available".
#   The whole 0.2.x sgl-kernel series is Hopper-only; Blackwell support
#   first appears in the 0.3.x series, which requires torch 2.8.0. So:
#     - torch 2.8.0+cu128 / torchvision 0.23.0  (ship sm_100 cubins).
#     - sgl-kernel 0.3.17.post1                 (ships an sm100/ build).
#     - triton 3.4.0                            (required by torch 2.8.0).
#   sgl-kernel / flashinfer wheels are installed with --no-deps so their
#   transitive pins cannot downgrade the cu128 libs. The sglang fork and
#   vortex_torch are installed with --no-deps to skip their torch pins;
#   their runtime deps are installed separately. The vortex_torch_C CUDA
#   extension is compiled with the spack CUDA 12.8 toolchain targeting
#   sm_100 natively (TORCH_CUDA_ARCH_LIST="10.0").
set -euo pipefail

REPO=/vast/projects/liuv/pennnetworks/xutingl/vortex_torch
BENCH="$REPO/quest_batch_benchmark"
VENV="$BENCH/.venv"
# CUDA 12.8 nvcc from spack — supports sm_100 (B200) natively.
# Used only to BUILD the vortex_torch_C CUDA extension. Override if yours is
# elsewhere:  BUILD_CUDA_HOME=<cuda-toolkit-dir> bash setup_env.sh
BUILD_CUDA_HOME="${BUILD_CUDA_HOME:-/vast/parcc/spack/sw/apps/linux-sapphirerapids/cuda-12.8.1-lmm74gnqr2pl2dzbtfjdwoo3fnwbar43}"
TORCH_INDEX=https://download.pytorch.org/whl/cu128

cd "$REPO"

# --- preflight: check prerequisites, fetch the sglang fork submodule -----
# Prerequisites: an NVIDIA B200 (sm_100) GPU, `uv` on PATH, Python 3.12, a
# CUDA >= 12.8 toolkit ($BUILD_CUDA_HOME), and network access (PyPI + the
# PyTorch cu128 index).
command -v uv >/dev/null || { echo "ERROR: 'uv' is not on PATH -- install uv first." >&2; exit 1; }
if [ ! -x "$BUILD_CUDA_HOME/bin/nvcc" ]; then
  echo "ERROR: no CUDA toolkit found at BUILD_CUDA_HOME=$BUILD_CUDA_HOME" >&2
  echo "       vortex_torch_C needs a CUDA >= 12.8 nvcc to build for sm_100." >&2
  echo "       Override: BUILD_CUDA_HOME=<cuda-toolkit-dir> bash setup_env.sh" >&2
  exit 1
fi
echo "[0/6] fetch the patched sglang fork submodule"
git submodule update --init third_party/sglang

echo "[1/6] create uv venv (python 3.12)"
uv venv --python 3.12 "$VENV" 2>/dev/null || uv venv --python 3.12 --clear "$VENV"

echo "[2/6] install torch 2.8.0+cu128 (includes sm_100 cubins for B200)"
uv pip install --python "$VENV" \
  --index-url "$TORCH_INDEX" \
  torch==2.8.0 torchvision==0.23.0

echo "[3/6] install the bundled-stack wheels (pinned, --no-deps to preserve cu128)"
# Install with --no-deps to avoid transitive CUDA 12.6 packages that would
# downgrade our cu128 nvidia-* libs and force torch back to +cu126.
uv pip install --python "$VENV" --no-deps \
  sgl-kernel==0.3.17.post1 \
  flashinfer-python==0.2.7.post1 \
  flashinfer-cubin==0.6.8.post1

# Install non-conflicting deps (pinned to reference_freeze.txt versions)
uv pip install --python "$VENV" \
  triton==3.4.0 \
  numpy==2.4.4 \
  pandas==3.0.3 \
  orjson==3.11.9 \
  uvloop==0.21.0 \
  "huggingface-hub==1.14.0" \
  pytest==9.0.3 \
  ninja \
  pynvml \
  einops==0.8.1

echo "[4/6] install the patched sglang fork (editable, --no-deps to skip torch pin)"
# The sglang fork pins torch==2.7.1 (cu126) in its optional extras.
# Install with --no-deps, then manually pull the runtime deps we need.
uv pip install --python "$VENV" --no-deps -e "$REPO/third_party/sglang/python"

uv pip install --python "$VENV" \
  "aiohttp" "requests" "tqdm==4.67.1" "IPython" "setproctitle" \
  "fastapi" "uvicorn" "hf_transfer" \
  "interegular" "llguidance>=0.7.11,<0.8.0" \
  "msgspec" "packaging==25.0" "partial_json_parser" "pillow" \
  "prometheus-client>=0.20.0" "psutil" "pydantic" "python-multipart" \
  "pyzmq>=25.1.2" "xgrammar==0.2.0" \
  "torchao==0.9.0" "transformers==5.8.1" "timm==1.0.16" \
  "compressed-tensors"

echo "[5/6] build + install vortex_torch (editable, CUDA extension, sm_100)"
# CUDA 12.8 nvcc can emit sm_100 SASS natively (no PTX JIT needed for B200).
# --no-deps skips vortex_torch's install_requires (torchao etc.) which would
# otherwise re-resolve and downgrade the cu128 torch.
# --no-build-isolation ensures the build uses the venv's torch 2.8.0+cu128.
CUDA_HOME="$BUILD_CUDA_HOME" \
PATH="$BUILD_CUDA_HOME/bin:$PATH" \
TORCH_CUDA_ARCH_LIST="10.0" \
  uv pip install --python "$VENV" --no-deps -e "$REPO" --no-build-isolation

echo "[6/6] done — venv at $VENV"
