#!/usr/bin/env bash
# Orchestrator: runs all 3 stages of the bench1 fetch-policy experiment.
#
#   1. Bandwidth regime sweep        (no model, ~3-5 min)
#   2. Synthetic policy sweep        (no model, ~10-15 min)
#   3. Real-trace policy experiment  (needs model, ~5-15 min depending on size)
#
# Override the Python interpreter:
#   PYTHON=/path/to/env/bin/python bash run_all_experiments.sh
#
# All outputs land under logs/ — see README.md for what each file tells you.

set -e

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Python env. On RTX 5060 Ti development the default below works.
# On B200, set PYTHON or VORTEX_ENV_ROOT externally to point at your env.
# ---------------------------------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
    VORTEX_ENV_ROOT="${VORTEX_ENV_ROOT:-$HOME/miniforge3/envs/vortex_v04}"
    PYTHON="$VORTEX_ENV_ROOT/bin/python"
    export CUDA_HOME="$VORTEX_ENV_ROOT"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
echo "Using PYTHON=$PYTHON"

mkdir -p logs

# ---------------------------------------------------------------------------
# Stage 1: bandwidth regime sweep
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  STAGE 1: bandwidth regime sweep  ($(date +%H:%M:%S))"
echo "  page_size x selected_seq_len matrix, no model needed."
echo "============================================================"
# Use the patched bench on sm_120; on B200 you can edit this script to
# call bench_decode_bandwidth.py instead (use_tensor_cores=True path).
PYTHON="$PYTHON" bash run_bandwidth_sweep_seqlen.sh

# ---------------------------------------------------------------------------
# Stage 2: synthetic policy sweep
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  STAGE 2: synthetic policy sweep  ($(date +%H:%M:%S))"
echo "  Block Fetch / Method 1 / Method 2 on random + clustered."
echo "============================================================"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" algorithm_scientist/page_policy_sweep.py \
    --needed-list 256 2048 8192 \
    --page-list 4 8 16 32 \
    --x-list 0.10 0.25 0.50 0.75 \
    --distributions random clustered \
    --out-csv logs/page_policy_sweep.csv

# ---------------------------------------------------------------------------
# Stage 3: real-trace experiment
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  STAGE 3: real-trace experiment  ($(date +%H:%M:%S))"
echo "  RULER 2-sample run with sparse_kv_indices dump, then analyze."
echo "============================================================"
rm -rf logs/traces && mkdir -p logs/traces
export VORTEX_DUMP_TRACE_DIR=/home/wangxian/vortex_torch/logs/traces
CUDA_VISIBLE_DEVICES=0 "$PYTHON" algorithm_scientist/run_ruler_trace.py \
    --config submissions/block_size_sweep/batch_0_id0.json
unset VORTEX_DUMP_TRACE_DIR

"$PYTHON" algorithm_scientist/trace_policy_analyzer.py \
    --trace-dir logs/traces \
    --page-sizes 16 32 64 128 256 \
    --thresholds 0.10 0.25 0.50 0.75 \
    --max-rows-per-layer 100 \
    --out-csv logs/trace_policy_analysis.csv

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  DONE  ($(date +%H:%M:%S))"
echo "============================================================"
echo "Outputs:"
echo "  logs/bandwidth_sweep_seqlen_*/summary.tsv     <- bandwidth regimes"
echo "  logs/page_policy_sweep.csv                    <- synthetic comparison"
echo "  logs/trace_policy_analysis.csv                <- real-trace comparison (headline)"
echo ""
echo "Headline (real-trace): look at the 'Block Fetch' row vs 'Method 1' / 'Method 2'"
echo "rows in trace_policy_analysis.csv. Block Fetch should have coverage=1.0,"
echo "waste=0.0, smallest loaded_MB."
