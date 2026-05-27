#!/usr/bin/env bash
# Drive the two-engine-API comparison sweep: dense + quest, each via the
# 'direct' sgl.Engine(...) path AND Quest's official
# vortex_torch.engine.sgl.get_engine wrapper. Aggregate each pair, then
# diff. Fairness flags are matched in build_get_engine_kwargs; any TPOT
# delta in the final table is attributable to the constructor call itself.
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH"

GPU="${GPU:-0}"
PY="$BENCH/.venv/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p results logs

for api in direct get_engine; do
    RAW="$BENCH/results/raw_results_${api}.csv"
    OUT="$BENCH/results/tpot_vs_batchsize_${api}.csv"
    rm -f "$RAW"
    for mode in dense quest; do
        echo ">>> running $mode via engine_api=$api  (GPU $GPU)"
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" benchmark_quest_tpot.py \
            --attention "$mode" \
            --engine-api "$api" \
            --raw-csv "$RAW" \
            2>&1 | tee "logs/${mode}_${api}_${TS}.log"
        status=${PIPESTATUS[0]}
        if [ "$status" -ne 0 ]; then
            echo "!!! $mode (api=$api) failed (exit $status) -- see logs/${mode}_${api}_${TS}.log" >&2
            exit "$status"
        fi
    done
    echo ">>> aggregating $api"
    "$PY" aggregate_results.py --raw-csv "$RAW" --out-csv "$OUT" || exit 1
done

echo ">>> building engine-API comparison"
"$PY" compare_engine_apis.py \
    --direct-csv "$BENCH/results/tpot_vs_batchsize_direct.csv" \
    --get-engine-csv "$BENCH/results/tpot_vs_batchsize_get_engine.csv" \
    --out-csv "$BENCH/results/engine_api_comparison.csv" \
    --out-md "$BENCH/results/engine_api_comparison.md" \
    2>&1 | tee "logs/engine_api_comparison_${TS}.log"

echo ">>> done"
echo "    direct raw       : $BENCH/results/raw_results_direct.csv"
echo "    direct agg       : $BENCH/results/tpot_vs_batchsize_direct.csv"
echo "    get_engine raw   : $BENCH/results/raw_results_get_engine.csv"
echo "    get_engine agg   : $BENCH/results/tpot_vs_batchsize_get_engine.csv"
echo "    comparison CSV   : $BENCH/results/engine_api_comparison.csv"
echo "    comparison table : $BENCH/results/engine_api_comparison.md"
