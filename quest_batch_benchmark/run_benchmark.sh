#!/usr/bin/env bash
# Drive the full Quest batch benchmark: dense baseline, then quest, then aggregate.
# Each mode boots its own sgl.Engine and measures decode TPOT in streaming mode,
# matching the sgl baseline `run_batch_experiments_offline.sh tpot-no-share`.
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH"

GPU="${GPU:-0}"
PY="$BENCH/.venv/bin/python"
RAW="$BENCH/results/raw_results.csv"
OUT="$BENCH/results/tpot_vs_batchsize.csv"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p results logs
rm -f "$RAW"   # fresh raw CSV; harness appends per mode

for mode in dense quest; do
  echo ">>> running $mode  (GPU $GPU)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
    --attention "$mode" \
    --raw-csv "$RAW" \
    2>&1 | tee "logs/${mode}_${TS}.log"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "!!! $mode run failed (exit $status) -- see logs/${mode}_${TS}.log" >&2
    exit "$status"
  fi
done

echo ">>> aggregating"
"$PY" aggregate_results.py --raw-csv "$RAW" --out-csv "$OUT"

echo ">>> done"
echo "    raw      : $RAW"
echo "    processed: $OUT"
