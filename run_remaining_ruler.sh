#!/usr/bin/env bash
# Run RULER sequentially on variants B (id1), C (id2), D (id3).
# Variant A (id0) is launched separately first.
set -u
cd /home/wangxian/vortex_torch
export VORTEX_ENV_ROOT="$HOME/miniforge3/envs/vortex_v04"
export CUDA_HOME="$VORTEX_ENV_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
LOGDIR=/home/wangxian/vortex_torch/logs/submission/block_size_sweep
mkdir -p "$LOGDIR"

for y in 1 2 3; do
    echo "==================== RULER id${y} START $(date +%H:%M:%S) ===================="
    CUDA_VISIBLE_DEVICES=0 "$VORTEX_ENV_ROOT/bin/python" \
        algorithm_scientist/run_ruler.py \
        --config "submissions/block_size_sweep/batch_0_id${y}.json" \
        > "$LOGDIR/ruler_id${y}.out" 2> "$LOGDIR/ruler_id${y}.err"
    rc=$?
    echo "==================== RULER id${y} DONE  $(date +%H:%M:%S) (rc=$rc) ===================="
    if [ $rc -ne 0 ]; then
        echo "[WARN] id${y} exited with $rc — see $LOGDIR/ruler_id${y}.err"
    fi
done

echo "==================== ALL REMAINING RULER VARIANTS DONE $(date +%H:%M:%S) ===================="
ls -lt /home/wangxian/vortex_torch/summary_ruler_submissions/block_size_sweep/ 2>/dev/null
