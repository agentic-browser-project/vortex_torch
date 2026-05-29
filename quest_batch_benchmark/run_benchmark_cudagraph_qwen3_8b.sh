#!/usr/bin/env bash
# Run the CUDA-graph batch benchmark for Qwen3-8B (dense + quest@64 +
# quest_topk29@29) writing into results/qwen3_8b/. Thin wrapper over
# run_benchmark_cudagraph.sh. Requires the no-graph Qwen3-8B sweep to have run
# first (the comparison joins against results/qwen3_8b/tpot_vs_batchsize.csv).
#
# Usage:  GPU=<id> bash quest_batch_benchmark/run_benchmark_cudagraph_qwen3_8b.sh
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B}"
export RESULTS_DIR="${RESULTS_DIR:-$BENCH/results/qwen3_8b}"

exec bash "$BENCH/run_benchmark_cudagraph.sh" "$@"
