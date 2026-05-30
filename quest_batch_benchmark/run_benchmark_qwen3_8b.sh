#!/usr/bin/env bash
# Run the no-graph batch benchmark for the SECOND model, Qwen3-8B
# (dense + quest@64 + quest_topk29@29), writing into results/qwen3_8b/.
# A thin wrapper over run_benchmark.sh: it pins the model + output dir, so all
# fairness flags and stages stay identical to the headline run.
#
# TreeSparse is intentionally SKIPPED for Qwen3-8B (RUN_TREESPARSE=0).
# TreeSparse's harness hardcodes the Qwen3-VL model class; given a text-only
# Qwen3-8B checkpoint it silently loads a wrong-architecture, random-weight
# model (the real checkpoint weights load as "unexpected"; the VL weights are
# newly initialized), so any TPOT it produced would be meaningless. The
# Qwen3-8B comparison is therefore dense + quest only. See the README section
# "Why TreeSparse is omitted for Qwen3-8B".
#
# Usage:  GPU=<id> bash quest_batch_benchmark/run_benchmark_qwen3_8b.sh
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B}"
export RESULTS_DIR="${RESULTS_DIR:-$BENCH/results/qwen3_8b}"
export RUN_TREESPARSE="${RUN_TREESPARSE:-0}"

exec bash "$BENCH/run_benchmark.sh" "$@"
