#!/usr/bin/env bash
# CUDA-graph batch benchmark for Qwen3-8B on the TRT-LLM-family kernels
# (dense + quest@64 + quest_topk29@29), writing into results/qwen3_8b_trtllm/.
# Thin wrapper over run_benchmark_cudagraph.sh. Requires the no-graph trtllm
# sweep to have run first (the comparison joins against
# results/qwen3_8b_trtllm/tpot_vs_batchsize.csv).
#
# Both arms run on TRT-LLM-family kernels: QUEST via VORTEX_ATTENTION_BACKEND=trtllm
# (vortex's trtllm sparse-decode kernel), DENSE via ATTENTION_BACKEND=trtllm_mha
# (sglang's TensorRT-LLM MHA dense kernel -- a different kernel; dense never goes
# through the vortex sparse path). quest keeps attention_backend=flashinfer
# internally (the vortex sparse path is registered under it).
#
# Usage:  GPU=<id> bash quest_batch_benchmark/run_benchmark_cudagraph_qwen3_8b_trtllm.sh
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B}"
export RESULTS_DIR="${RESULTS_DIR:-$BENCH/results/qwen3_8b_trtllm}"
export VORTEX_ATTENTION_BACKEND="${VORTEX_ATTENTION_BACKEND:-trtllm}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-trtllm_mha}"

exec bash "$BENCH/run_benchmark_cudagraph.sh" "$@"
