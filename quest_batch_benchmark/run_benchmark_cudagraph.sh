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
# Model + output dir parameterized (defaults reproduce the Qwen3-VL run).
MODEL_PATH="${MODEL_PATH:-/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct}"
RESULTS_DIR="${RESULTS_DIR:-$BENCH/results}"
VORTEX_ATTENTION_BACKEND="${VORTEX_ATTENTION_BACKEND:-flashinfer}"
RAW="$RESULTS_DIR/raw_results_cudagraph.csv"
OUT="$RESULTS_DIR/tpot_vs_batchsize_cudagraph.csv"
NOGRAPH_AGG="$RESULTS_DIR/tpot_vs_batchsize.csv"
CMP_CSV="$RESULTS_DIR/cuda_graph_comparison.csv"
CMP_MD="$RESULTS_DIR/cuda_graph_comparison.md"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR" logs
rm -f "$RAW"   # fresh raw CSV; harness appends per mode

# Three modes under --enable-cuda-graph: dense, quest topk_val=64, quest
# topk_val=29. Same fairness contract as the no-graph driver: single physical
# run, --label isolates the second quest variant in the raw CSV.

echo ">>> running dense --enable-cuda-graph  (GPU $GPU)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
    --attention dense \
    --enable-cuda-graph \
    --model-path "$MODEL_PATH" \
    --vortex-attention-backend "$VORTEX_ATTENTION_BACKEND" \
    --raw-csv "$RAW" \
    2>&1 | tee "logs/dense_cudagraph_${TS}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "!!! dense (cudagraph) failed (exit $status) -- see logs/dense_cudagraph_${TS}.log" >&2
    exit "$status"
fi

echo ">>> running quest (topk_val=64) --enable-cuda-graph  (GPU $GPU)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
    --attention quest \
    --topk-val 64 \
    --enable-cuda-graph \
    --model-path "$MODEL_PATH" \
    --vortex-attention-backend "$VORTEX_ATTENTION_BACKEND" \
    --raw-csv "$RAW" \
    2>&1 | tee "logs/quest_cudagraph_${TS}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "!!! quest (topk=64, cudagraph) failed (exit $status) -- see logs/quest_cudagraph_${TS}.log" >&2
    exit "$status"
fi

echo ">>> running quest_topk29 (topk_val=29) --enable-cuda-graph  (GPU $GPU)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
    --attention quest \
    --topk-val 29 \
    --label quest_topk29 \
    --enable-cuda-graph \
    --model-path "$MODEL_PATH" \
    --vortex-attention-backend "$VORTEX_ATTENTION_BACKEND" \
    --raw-csv "$RAW" \
    2>&1 | tee "logs/quest_topk29_cudagraph_${TS}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "!!! quest_topk29 (cudagraph) failed (exit $status) -- see logs/quest_topk29_cudagraph_${TS}.log" >&2
    exit "$status"
fi

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
