#!/usr/bin/env bash
# No-graph batch benchmark for Qwen3-8B on the TRT-LLM-family kernels
# (dense + quest@64 + quest_topk29@29), writing into results/qwen3_8b_trtllm/.
# Thin wrapper over run_benchmark.sh: it pins the model, output dir, both the
# vortex (sparse) and the sglang dense attention backends to TRT-LLM, and skips
# TreeSparse.
#
# Two distinct backends are set so BOTH arms run on TRT-LLM-family kernels:
#   - QUEST (sparse): VORTEX_ATTENTION_BACKEND=trtllm -> vortex's own trtllm
#     sparse-decode kernel.
#   - DENSE: ATTENTION_BACKEND=trtllm_mha -> sglang's TensorRT-LLM MHA dense
#     kernel (NOT the vortex sparse kernel -- a different kernel; dense never
#     goes through the vortex sparse path). This removes the flashinfer-vs-trtllm
#     confound from the dense-vs-quest comparison, but dense and quest still use
#     different (though both TRT-LLM-family) kernels by necessity.
# Note: quest keeps attention_backend=flashinfer internally regardless of
# ATTENTION_BACKEND -- the vortex sparse path is registered under sglang's
# flashinfer backend, so the harness pins it.
#
# TreeSparse is SKIPPED (RUN_TREESPARSE=0) for the same reason as the flashinfer
# Qwen3-8B wrapper -- its harness can't load a text-only Qwen3-8B checkpoint
# correctly. See the README "Why TreeSparse is omitted for Qwen3-8B".
#
# Usage:  GPU=<id> bash quest_batch_benchmark/run_benchmark_qwen3_8b_trtllm.sh
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B}"
export RESULTS_DIR="${RESULTS_DIR:-$BENCH/results/qwen3_8b_trtllm}"
export RUN_TREESPARSE="${RUN_TREESPARSE:-0}"
export VORTEX_ATTENTION_BACKEND="${VORTEX_ATTENTION_BACKEND:-trtllm}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-trtllm_mha}"

exec bash "$BENCH/run_benchmark.sh" "$@"
