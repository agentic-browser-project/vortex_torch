#!/usr/bin/env bash
# Run the full no-graph batch benchmark for the SECOND model, Qwen3-8B
# (dense + quest@64 + quest_topk29@29 + TreeSparse), writing into
# results/qwen3_8b/. A thin wrapper over run_benchmark.sh: it only pins the
# model + output dir, so all fairness flags and stages stay identical to the
# headline run.
#
# Like the headline Qwen3-VL run, all four methods run on the same model here
# (Qwen3-8B). TSA_MODEL is appended as TreeSparse's --model-path so TreeSparse
# runs Qwen3-8B too (TreeSparse's own hard-coded default is Qwen3-VL).
#
# Usage:  GPU=<id> bash quest_batch_benchmark/run_benchmark_qwen3_8b.sh
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B}"
export RESULTS_DIR="${RESULTS_DIR:-$BENCH/results/qwen3_8b}"
export TSA_MODEL="${TSA_MODEL:-/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B}"
export TSA_MODEL_TAG="${TSA_MODEL_TAG:-Qwen3-8B}"

exec bash "$BENCH/run_benchmark.sh" "$@"
