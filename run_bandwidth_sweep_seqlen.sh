#!/usr/bin/env bash
# Sweep bench_decode_bandwidth_notc.py over (page_size, selected_seq_len)
# Fixed total_seq_len=32768 (large KV pool, overflows L2 + GPU caches).
# Goal: find the point where the kernel becomes bandwidth-bound and
# whether page_size matters once we're there.
set -u
cd /home/wangxian/vortex_torch
export VORTEX_ENV_ROOT="$HOME/miniforge3/envs/vortex_v04"
export CUDA_HOME="$VORTEX_ENV_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
PY="$VORTEX_ENV_ROOT/bin/python"

LOGDIR=/home/wangxian/vortex_torch/logs/bandwidth_sweep_seqlen_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
echo -e "page_size\ttotal_seq_len\tseq_len\tstatus\tlog_path" > "$SUMMARY"

TOTAL_SEQ_LEN=32768
PAGE_SIZES=(16 8 4 1)
SELECTED=(512 1024 2048 4096 8192 16384)

for ps in "${PAGE_SIZES[@]}"; do
    for sel in "${SELECTED[@]}"; do
        # round sel down to multiple of ps and cap at total
        eff=$(( (sel / ps) * ps ))
        [ "$eff" -gt "$TOTAL_SEQ_LEN" ] && eff=$TOTAL_SEQ_LEN
        [ "$eff" -lt "$ps" ] && eff=$ps

        tag="ps${ps}_sel${sel}"
        log="$LOGDIR/${tag}.log"
        echo "=== running ps=$ps sel=$eff total=$TOTAL_SEQ_LEN ($(date +%H:%M:%S)) ===" | tee -a "$LOGDIR/run.log"
        CUDA_VISIBLE_DEVICES=0 "$PY" bench_decode_bandwidth_notc.py \
            --batch-size 1 \
            --total-seq-len "$TOTAL_SEQ_LEN" \
            --seq-len "$eff" \
            --num-qo-heads 16 --num-kv-heads 8 --head-dim 128 \
            --page-size "$ps" \
            --kv-dtype bf16 \
            --warmup 5 --iters 50 \
            > "$log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            echo -e "${ps}\t${TOTAL_SEQ_LEN}\t${eff}\tOK\t${tag}.log" >> "$SUMMARY"
        else
            echo -e "${ps}\t${TOTAL_SEQ_LEN}\t${eff}\tFAIL(rc=${rc})\t${tag}.log" >> "$SUMMARY"
        fi
    done
done

echo "=== sweep complete $(date +%H:%M:%S) ==="
cat "$SUMMARY"
echo "--- LOGDIR = $LOGDIR ---"
