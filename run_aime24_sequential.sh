#!/usr/bin/env bash
# Sequential AIME24 benchmark on all 4 block_size_sweep variants.
# Single GPU, ~20-60 min per variant. Run only after RULER passers identified.
set -u
cd /home/wangxian/vortex_torch
export VORTEX_ENV_ROOT="$HOME/miniforge3/envs/vortex_v04"
export CUDA_HOME="$VORTEX_ENV_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
LOGDIR=/home/wangxian/vortex_torch/logs/submission/block_size_sweep
mkdir -p "$LOGDIR"

for y in 0 1 2 3; do
    echo "==================== AIME24 id${y} START $(date +%H:%M:%S) ===================="
    CUDA_VISIBLE_DEVICES=0 "$VORTEX_ENV_ROOT/bin/python" \
        algorithm_scientist/run_submission_aime24.py \
        --config "submissions/block_size_sweep/batch_0_id${y}.json" \
        > "$LOGDIR/aime_id${y}.out" 2> "$LOGDIR/aime_id${y}.err"
    rc=$?
    echo "==================== AIME24 id${y} DONE rc=$rc $(date +%H:%M:%S) ===================="
    if [ $rc -ne 0 ]; then
        echo "[WARN] id${y} exited rc=$rc — see $LOGDIR/aime_id${y}.err"
    fi
    cat /home/wangxian/vortex_torch/summary_submissions/block_size_sweep/batch_0_id${y}/latest.json 2>/dev/null | grep -E "mean@16|throughput|tokens_per_second|elapsed" | head -5
done

echo "==================== ALL 4 AIME24 VARIANTS DONE $(date +%H:%M:%S) ===================="
