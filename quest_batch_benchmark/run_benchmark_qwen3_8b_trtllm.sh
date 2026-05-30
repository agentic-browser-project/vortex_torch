#!/usr/bin/env bash
# No-graph batch benchmark for Qwen3-8B with the TRTLLM vortex attention
# backend (dense + quest@64 + quest_topk29@29), writing into
# results/qwen3_8b_trtllm/. Thin wrapper over run_benchmark.sh: it pins the
# model, output dir, the trtllm vortex backend, and skips TreeSparse.
#
# Backend note: only the QUEST runs differ from the flashinfer Qwen3-8B sweep
# (vortex_attention_backend=trtllm vs flashinfer). DENSE sets
# enable_vortex_sparsity=False, so vortex_attention_backend is never consulted
# -- dense here uses sglang's attention_backend=flashinfer and is numerically
# identical to dense in results/qwen3_8b/. It is still run as the comparison
# baseline so results/qwen3_8b_trtllm/ is a self-contained single physical run.
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

exec bash "$BENCH/run_benchmark.sh" "$@"
