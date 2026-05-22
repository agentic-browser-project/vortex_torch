#!/usr/bin/env bash
# Drive the full three-way batch benchmark: dense + quest, then aggregate, then
# the TreeSparseAttention run, then the three-way comparison merge.
# The dense/quest modes each boot their own sgl.Engine and measure decode TPOT
# in streaming mode, matching the sgl baseline
# `run_batch_experiments_offline.sh tpot-no-share`. TreeSparse runs separately
# via run_treesparse.sh in its own environment.
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

echo ">>> aggregating dense + quest"
"$PY" aggregate_results.py --raw-csv "$RAW" --out-csv "$OUT" || exit 1

echo ">>> running treesparse  (TreeSparseAttention's own environment)"
CUDA_VISIBLE_DEVICES="$GPU" bash "$BENCH/run_treesparse.sh" \
  2>&1 | tee "logs/treesparse_${TS}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "!!! treesparse run failed (exit $status) -- see logs/treesparse_${TS}.log" >&2
  exit "$status"
fi

echo ">>> building the three-way comparison"
# --quest-csv / --treesparse-json are passed explicitly so the call still works
# if $OUT is overridden; both equal build_comparison.py's own defaults.
"$PY" build_comparison.py \
  --quest-csv "$OUT" \
  --treesparse-json "$BENCH/results/treesparse_raw.json" \
  2>&1 | tee "logs/comparison_${TS}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "!!! comparison build failed (exit $status) -- see logs/comparison_${TS}.log" >&2
  exit "$status"
fi

echo ">>> done"
echo "    raw (dense+quest) : $RAW"
echo "    aggregated        : $OUT"
echo "    treesparse raw    : $BENCH/results/treesparse_raw.json"
echo "    three-way CSV     : $BENCH/results/tpot_three_way.csv"
echo "    comparison table  : $BENCH/results/comparison_table.md"
