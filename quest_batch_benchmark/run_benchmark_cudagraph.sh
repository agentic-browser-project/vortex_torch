#!/usr/bin/env bash
# Second benchmark category: dense + quest with `--enable-cuda-graph`.
# Mirrors run_benchmark.sh but adds the flag and writes to _cudagraph result
# files. TreeSparse is NOT run here -- see cuda_graph_status.md for why; its
# decode harness is structurally incompatible with torch CUDA graphs.
#
# Final stage: compare_cuda_graph.py joins this sweep's aggregated CSV with
# the existing no-graph aggregate (results/tpot_vs_batchsize.csv) and writes
# a side-by-side graph-vs-no-graph table.
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH"

GPU="${GPU:-0}"
PY="$BENCH/.venv/bin/python"
RAW="$BENCH/results/raw_results_cudagraph.csv"
OUT="$BENCH/results/tpot_vs_batchsize_cudagraph.csv"
NOGRAPH_AGG="$BENCH/results/tpot_vs_batchsize.csv"
CMP_CSV="$BENCH/results/cuda_graph_comparison.csv"
CMP_MD="$BENCH/results/cuda_graph_comparison.md"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p results logs
rm -f "$RAW"   # fresh raw CSV; harness appends per mode

for mode in dense quest; do
    echo ">>> running $mode --enable-cuda-graph  (GPU $GPU)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
        --attention "$mode" \
        --enable-cuda-graph \
        --raw-csv "$RAW" \
        2>&1 | tee "logs/${mode}_cudagraph_${TS}.log"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "!!! $mode (cudagraph) failed (exit $status) -- see logs/${mode}_cudagraph_${TS}.log" >&2
        exit "$status"
    fi
done

echo ">>> aggregating cudagraph"
"$PY" aggregate_results.py --raw-csv "$RAW" --out-csv "$OUT" || exit 1

if [ ! -s "$NOGRAPH_AGG" ]; then
    echo "!!! no-graph aggregate missing at $NOGRAPH_AGG -- run run_benchmark.sh first," >&2
    echo "    then re-run this driver to build the graph-vs-no-graph comparison." >&2
    exit 1
fi

echo ">>> building cuda-graph vs no-graph comparison"
"$PY" compare_cuda_graph.py \
    --nograph-csv "$NOGRAPH_AGG" \
    --cudagraph-csv "$OUT" \
    --out-csv "$CMP_CSV" \
    --out-md "$CMP_MD" \
    2>&1 | tee "logs/cuda_graph_comparison_${TS}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "!!! cuda-graph comparison failed (exit $status) -- see logs/cuda_graph_comparison_${TS}.log" >&2
    exit "$status"
fi

echo ">>> done"
echo "    raw (cudagraph)   : $RAW"
echo "    aggregated        : $OUT"
echo "    comparison CSV    : $CMP_CSV"
echo "    comparison table  : $CMP_MD"
