#!/usr/bin/env bash
# Drive the full four-way batch benchmark: dense + quest (topk_val=64) +
# quest_topk29 (topk_val=29), then aggregate, then the TreeSparseAttention
# run, then the four-way comparison merge.
# Each dense/quest mode boots its own sgl.Engine and measures decode TPOT
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

# Three modes: dense, quest with topk_val=64 (headline), quest with topk_val=29
# (the vortex_torch get_engine default). All three share machine state so the
# cross-method comparison is on a single physical run. The --label flag on the
# third invocation puts quest_topk29 into its own (attention, batch_size) group
# in the raw CSV — aggregate_results.py groups by `attention`, so the two quest
# variants don't collide.

echo ">>> running dense  (GPU $GPU)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
  --attention dense \
  --raw-csv "$RAW" \
  2>&1 | tee "logs/dense_${TS}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "!!! dense run failed (exit $status) -- see logs/dense_${TS}.log" >&2
  exit "$status"
fi

echo ">>> running quest (topk_val=64)  (GPU $GPU)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
  --attention quest \
  --topk-val 64 \
  --raw-csv "$RAW" \
  2>&1 | tee "logs/quest_${TS}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "!!! quest (topk=64) run failed (exit $status) -- see logs/quest_${TS}.log" >&2
  exit "$status"
fi

echo ">>> running quest_topk29 (topk_val=29 = vortex_torch get_engine default)  (GPU $GPU)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
  --attention quest \
  --topk-val 29 \
  --label quest_topk29 \
  --raw-csv "$RAW" \
  2>&1 | tee "logs/quest_topk29_${TS}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "!!! quest_topk29 run failed (exit $status) -- see logs/quest_topk29_${TS}.log" >&2
  exit "$status"
fi

echo ">>> aggregating dense + quest + quest_topk29"
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
echo "    raw (dense+quest+quest_topk29) : $RAW"
echo "    aggregated        : $OUT"
echo "    treesparse raw    : $BENCH/results/treesparse_raw.json"
echo "    three-way CSV     : $BENCH/results/tpot_three_way.csv"
echo "    comparison table  : $BENCH/results/comparison_table.md"
