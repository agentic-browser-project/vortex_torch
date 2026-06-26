#!/usr/bin/env bash

set -uo pipefail
sparse_algos=(
block_sparse_attention
)

models=(
/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-0.6B

/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-1.7B

/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-4B

/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B

)
trials=(
16 32 64
)
topk_val=(
29 61 93 125 157 189 221 253
)
for algo in "${sparse_algos[@]}"; do
  for model in "${models[@]}"; do
    for trial in "${trials[@]}"; do
      for k_val in "${topk_val[@]}"; do
        echo ">>> Running verify_algo.py with --vortex-module-name ${algo} and --model-name ${model} for ${trial} trials k=${k_val}"
        python examples/verify_algo.py \
            --trials ${trial} \
            --topk-val ${k_val} \
            --page-size 16 \
            --workload-chunk-size 64 \
            --block-size 16 \
            --topk-ratio 0.0625 \
            --vortex-module-name "${algo}" \
            --model-name  "${model}" \
            --mem 0.85 \
            --data-path examples/aime24.jsonl \
            --generation-max-new-tokens 16384 \
            --max-input-length 4096 \
            --tp-size 1 \
            --summary-dir summary-Qwen3-4B \
          || echo "!!! FAILED: algo=${algo} model=${model} trial=${trial} k=${k_val} — continuing"
      done
    done
  done
done