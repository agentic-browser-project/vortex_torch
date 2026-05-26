#!/usr/bin/env bash
# Simulation analysis: dump real sparse_kv_indices traces from vortex_torch's
# block_sparse_attention, then evaluate three fetch policies on those traces.
#
#   Block Fetch  — load only the selected blocks (vortex_torch's current behavior)
#   Method 1     — for each conceptual page (page_size > block_size), if any
#                  block in it is selected, load the whole page
#   Method 2     — same but only load the page if hit ratio >= threshold,
#                  otherwise drop the entire page
#
# Block Fetch is at block granularity (page_size = block_size = 16). Method 1
# and Method 2 operate at coarser page granularity (page_size > 16). Both are
# post-hoc simulation: vortex_torch actually runs Block Fetch (page_size=16);
# the analyzer computes what Method 1 / Method 2 would have loaded instead.
#
# Override the Python interpreter:
#   PYTHON=/path/to/env/bin/python bash run_all_experiments.sh
#
# Expected total time on B200 + 70B + warm caches: ~5–10 min.

set -e
cd "$(dirname "$0")"

# Default interpreter is the local RTX 5060 Ti env. On B200, set PYTHON.
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
# Step 1: dump real traces by running RULER on vortex_torch's
#         block_sparse_attention (block_size = page_size = 16, default).
# ---------------------------------------------------------------------------
echo ""
echo "[1/2] Dumping real sparse_kv_indices traces from RULER  ($(date +%H:%M:%S))"
echo "      Upstream algorithm: vortex_torch block_sparse_attention"
echo "      block_size = 16 (the granularity of indexer selection)"

rm -rf logs/traces && mkdir -p logs/traces
export VORTEX_DUMP_TRACE_DIR="$(pwd)/logs/traces"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" algorithm_scientist/run_ruler_trace.py \
    --config submissions/block_size_sweep/batch_0_id0.json
unset VORTEX_DUMP_TRACE_DIR

# ---------------------------------------------------------------------------
# Step 2: compute Block Fetch / Method 1 / Method 2 metrics on the traces.
#         page_size sweep is restricted to values > block_size (= 16) so the
#         three methods are actually different from each other.
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Computing fetch policy metrics  ($(date +%H:%M:%S))"
echo "      page_size in {32, 64, 128, 256}  (all > block_size = 16)"
echo "      threshold in {0.10, 0.25, 0.50, 0.75}  (Method 2 only)"

"$PYTHON" algorithm_scientist/trace_policy_analyzer.py \
    --trace-dir logs/traces \
    --page-sizes 32 64 128 256 \
    --thresholds 0.10 0.25 0.50 0.75 \
    --max-rows-per-layer 100 \
    --out-csv logs/trace_policy_analysis.csv

echo ""
echo "=== Done  ($(date +%H:%M:%S)) ==="
echo "Result: logs/trace_policy_analysis.csv"
echo ""
echo "How to read:"
echo "  Block Fetch row    -> baseline (page=block, no over-fetch, no drop)"
echo "  Method 1 rows      -> coverage stays 1, waste grows with page_size"
echo "  Method 2 rows      -> coverage drops to 0 once threshold > cluster density"
