#!/usr/bin/env bash
# Sweep bench_decode_bandwidth_notc.py over (page_size, total_seq_len)
# All other parameters fixed to a Qwen3-1.7B-shaped workload with
# 512 selected tokens (matching our block_size_sweep budget).
set -u
cd /home/wangxian/vortex_torch
export VORTEX_ENV_ROOT="$HOME/miniforge3/envs/vortex_v04"
export CUDA_HOME="$VORTEX_ENV_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
PY="$VORTEX_ENV_ROOT/bin/python"

LOGDIR=/home/wangxian/vortex_torch/logs/bandwidth_sweep_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
echo -e "page_size\ttotal_seq_len\tseq_len\tstatus\tlog_path" > "$SUMMARY"

PAGE_SIZES=(16 8 4 2 1)
TOTAL_SEQS=(4096 8192 16384 32768)
SEQ_LEN=512   # selected tokens per request — matches our 512-token budget

for ps in "${PAGE_SIZES[@]}"; do
    for ctx in "${TOTAL_SEQS[@]}"; do
        # seq_len must be a multiple of page_size and <= total_seq_len
        eff_sel=$(( SEQ_LEN > ctx ? ctx : SEQ_LEN ))
        # round eff_sel down to multiple of ps
        eff_sel=$(( (eff_sel / ps) * ps ))
        [ "$eff_sel" -lt "$ps" ] && eff_sel=$ps

        tag="ps${ps}_ctx${ctx}"
        log="$LOGDIR/${tag}.log"
        echo "=== running ps=$ps ctx=$ctx selected=$eff_sel ($(date +%H:%M:%S)) ===" | tee -a "$LOGDIR/run.log"
        CUDA_VISIBLE_DEVICES=0 "$PY" bench_decode_bandwidth_notc.py \
            --batch-size 1 \
            --total-seq-len "$ctx" \
            --seq-len "$eff_sel" \
            --num-qo-heads 16 --num-kv-heads 8 --head-dim 128 \
            --page-size "$ps" \
            --kv-dtype bf16 \
            --warmup 5 --iters 50 \
            > "$log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            echo -e "${ps}\t${ctx}\t${eff_sel}\tOK\t${tag}.log" >> "$SUMMARY"
        else
            echo -e "${ps}\t${ctx}\t${eff_sel}\tFAIL(rc=${rc})\t${tag}.log" >> "$SUMMARY"
        fi
    done
done

echo "=== sweep complete $(date +%H:%M:%S) ==="
echo "--- summary index ---"
cat "$SUMMARY"
echo "--- one example log tail ---"
head -50 "$LOGDIR/ps16_ctx4096.log" 2>/dev/null
echo "--- LOGDIR = $LOGDIR ---"
