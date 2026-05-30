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
# Model + output dir are parameterized so a second model (e.g. Qwen3-8B) can
# reuse this driver via a thin wrapper. Defaults reproduce the headline
# Qwen3-VL-8B-Instruct run exactly.
MODEL_PATH="${MODEL_PATH:-/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct}"
RESULTS_DIR="${RESULTS_DIR:-$BENCH/results}"
# TSA_MODEL: when non-empty, run_treesparse.sh APPENDS it as TreeSparse's
# --model-path (overriding TreeSparse's own hard-coded Qwen3-VL default).
# Empty by default, so the headline run keeps TreeSparse on Qwen3-VL, matched
# to dense/quest. TSA_MODEL_TAG is the cosmetic `model` column label for the
# merged treesparse rows; its default matches the model TreeSparse runs by
# default (Qwen3-VL).
TSA_MODEL="${TSA_MODEL:-}"
TSA_MODEL_TAG="${TSA_MODEL_TAG:-Qwen3-VL-8B-Instruct}"
VORTEX_ATTENTION_BACKEND="${VORTEX_ATTENTION_BACKEND:-flashinfer}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
RAW="$RESULTS_DIR/raw_results.csv"
OUT="$RESULTS_DIR/tpot_vs_batchsize.csv"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR" logs
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
  --model-path "$MODEL_PATH" \
  --vortex-attention-backend "$VORTEX_ATTENTION_BACKEND" \
  --attention-backend "$ATTENTION_BACKEND" \
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
  --model-path "$MODEL_PATH" \
  --vortex-attention-backend "$VORTEX_ATTENTION_BACKEND" \
  --attention-backend "$ATTENTION_BACKEND" \
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
  --model-path "$MODEL_PATH" \
  --vortex-attention-backend "$VORTEX_ATTENTION_BACKEND" \
  --attention-backend "$ATTENTION_BACKEND" \
  --raw-csv "$RAW" \
  2>&1 | tee "logs/quest_topk29_${TS}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "!!! quest_topk29 run failed (exit $status) -- see logs/quest_topk29_${TS}.log" >&2
  exit "$status"
fi

echo ">>> aggregating dense + quest + quest_topk29"
"$PY" aggregate_results.py --raw-csv "$RAW" --out-csv "$OUT" || exit 1

# RUN_TREESPARSE gates the TreeSparse stage + the four-way merge. Default 1
# (the headline Qwen3-VL run is unchanged). Set 0 to skip both -- used by the
# Qwen3-8B wrapper, because TreeSparse's harness hardcodes the Qwen3-VL model
# class and silently loads a wrong-architecture random-weight model for a
# text-only Qwen3 checkpoint (see README "Why TreeSparse is omitted for
# Qwen3-8B"). When skipped, the aggregated dense+quest CSV ($OUT) is the result.
if [ "${RUN_TREESPARSE:-1}" = "1" ]; then
  echo ">>> running treesparse  (TreeSparseAttention's own environment)"
  OUT_JSON="$RESULTS_DIR/treesparse_raw.json" TSA_MODEL="$TSA_MODEL" \
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
    --treesparse-json "$RESULTS_DIR/treesparse_raw.json" \
    --out-csv "$RESULTS_DIR/tpot_three_way.csv" \
    --out-md "$RESULTS_DIR/comparison_table.md" \
    --treesparse-model-tag "$TSA_MODEL_TAG" \
    2>&1 | tee "logs/comparison_${TS}.log"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "!!! comparison build failed (exit $status) -- see logs/comparison_${TS}.log" >&2
    exit "$status"
  fi
else
  echo ">>> skipping treesparse + four-way merge (RUN_TREESPARSE=0)"
fi

echo ">>> done"
echo "    model             : $MODEL_PATH"
echo "    results dir       : $RESULTS_DIR"
echo "    raw (dense+quest+quest_topk29) : $RAW"
echo "    aggregated        : $OUT"
if [ "${RUN_TREESPARSE:-1}" = "1" ]; then
  echo "    treesparse raw    : $RESULTS_DIR/treesparse_raw.json"
  echo "    three-way CSV     : $RESULTS_DIR/tpot_three_way.csv"
  echo "    comparison table  : $RESULTS_DIR/comparison_table.md"
fi
