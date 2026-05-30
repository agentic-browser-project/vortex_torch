#!/usr/bin/env bash
# CUDA-graph batch benchmark for Qwen3-8B with the TRTLLM vortex attention
# backend (dense + quest@64 + quest_topk29@29), writing into
# results/qwen3_8b_trtllm/. Thin wrapper over run_benchmark_cudagraph.sh.
# Requires the no-graph trtllm sweep to have run first (the comparison joins
# against results/qwen3_8b_trtllm/tpot_vs_batchsize.csv).
#
# Dense is backend-independent (enable_vortex_sparsity=False), so only the two
# Quest points exercise trtllm; attention_backend stays flashinfer.
#
# Usage:  GPU=<id> bash quest_batch_benchmark/run_benchmark_cudagraph_qwen3_8b_trtllm.sh
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B}"
export RESULTS_DIR="${RESULTS_DIR:-$BENCH/results/qwen3_8b_trtllm}"
export VORTEX_ATTENTION_BACKEND="${VORTEX_ATTENTION_BACKEND:-trtllm}"

exec bash "$BENCH/run_benchmark_cudagraph.sh" "$@"
