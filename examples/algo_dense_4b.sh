#!/usr/bin/env bash

set -uo pipefail

CUDART_DIR="/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/env312/lib"
export LD_LIBRARY_PATH="${CUDART_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ALGO=full_attention
MODEL=/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-4B

for trial in 16; do
  for k_val in 61; do
    echo ">>> algo=${ALGO} model=${MODEL} trial=${trial} k=${k_val}"
    python examples/verify_algo.py \
        --trials ${trial} \
        --topk-val ${k_val} \
        --page-size 16 \
        --workload-chunk-size 64 \
        --block-size 16 \
        --topk-ratio 0.0625 \
        --vortex-module-name "${ALGO}" \
        --model-name "${MODEL}" \
        --mem 0.85 \
        --data-path examples/aime24.jsonl \
        --generation-max-new-tokens 16384 \
        --max-input-length 4096 \
        --tp-size 1 \
        --summary-dir summary-Qwen3-4B-triton \
      || echo "!!! FAILED: trial=${trial} k=${k_val} — continuing"
  done
done
