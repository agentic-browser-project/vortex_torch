#!/usr/bin/env bash
# Source this file or use as a wrapper to run any vortex_torch command
# with the correct env vars + python interpreter.
#
# Usage: bash run_env.sh <any python script and args>
#   or:  source run_env.sh   (sets up env vars for the current shell)

export VORTEX_ENV_ROOT="$HOME/miniforge3/envs/vortex_v04"
export CUDA_HOME="$VORTEX_ENV_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTHON="$VORTEX_ENV_ROOT/bin/python"

if [ $# -gt 0 ]; then
    exec "$PYTHON" "$@"
fi
