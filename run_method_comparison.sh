#!/usr/bin/env bash
# Compare three fetch policies on real vortex_torch block_sparse_attention runs.
#
# Uses VORTEX_POLICY env var to switch the attention-time fetch behavior:
#   block_fetch       — vortex_torch's actual current behavior (baseline)
#   method1_pP        — load whole P-token "page" if any block in it was selected
#   method2_pP_tTT    — load page only when hit ratio >= TT/100, drop otherwise
#
# Upstream sparse-attention algorithm: vortex_torch's block_sparse_attention
# (cloned as block_size_sweep_id2_cls), with block_size=4, page_size=32.

set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-$HOME/miniforge3/envs/vortex_v04/bin/python}"
echo "Using PYTHON=$PYTHON"

if [ -z "${CUDA_HOME:-}" ]; then
    VORTEX_ENV_ROOT="${VORTEX_ENV_ROOT:-$HOME/miniforge3/envs/vortex_v04}"
    export CUDA_HOME="$VORTEX_ENV_ROOT"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

CONFIG="submissions/block_size_sweep/batch_0_id2_page32.json"
LOGDIR="logs/method_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# label  : run identifier and log filename prefix
# env vars: space-separated KEY=VALUE pairs that switch attention backend / policy
# - block_fetch / method1_p32 / method2_p32_t*: BatchDecode wrapper + index rewrite
# - bsr_baseline                              : BlockSparseAttention wrapper (Option C)
POLICIES=(
    "block_fetch:VORTEX_POLICY=block_fetch"
    "method1_p32:VORTEX_POLICY=method1_p32"
    "method2_p32_t25:VORTEX_POLICY=method2_p32_t25"
    "method2_p32_t50:VORTEX_POLICY=method2_p32_t50"
    "method2_p32_t75:VORTEX_POLICY=method2_p32_t75"
    "bsr_baseline:VORTEX_USE_BSR=1"
    "custom_baseline:VORTEX_USE_CUSTOM=1"
)

SUMMARY="$LOGDIR/summary.tsv"
echo -e "policy\taccuracy\tthroughput\trc" > "$SUMMARY"

for entry in "${POLICIES[@]}"; do
    policy="${entry%%:*}"
    envspec="${entry#*:}"
    echo ""
    echo "==================== $policy ($(date +%H:%M:%S)) ===================="
    echo "  env: $envspec"
    # Clean previous summary so we always read the latest run, not a stale one
    rm -rf summary_ruler_submissions_trace/block_size_sweep/batch_0_id2_page32/
    CUDA_VISIBLE_DEVICES=0 env $envspec \
        "$PYTHON" algorithm_scientist/run_ruler_trace.py \
        --config "$CONFIG" \
        > "$LOGDIR/${policy}.out" 2> "$LOGDIR/${policy}.err"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL] $policy exited $rc; see $LOGDIR/${policy}.err"
        echo -e "${policy}\tFAIL\tFAIL\t${rc}" >> "$SUMMARY"
        continue
    fi
    # Parse summary
    JSON="summary_ruler_submissions_trace/block_size_sweep/batch_0_id2_page32/latest.json"
    if [ -f "$JSON" ]; then
        cp "$JSON" "$LOGDIR/${policy}_result.json"
        acc=$("$PYTHON" -c "import json; print(json.load(open('$JSON')).get('accuracy', 'n/a'))")
        thr=$("$PYTHON" -c "import json; print(json.load(open('$JSON')).get('throughput', 'n/a'))")
        echo -e "${policy}\t${acc}\t${thr}\tOK" >> "$SUMMARY"
        echo "[OK]   $policy: accuracy=$acc throughput=$thr"
    else
        echo "[WARN] $policy: no summary JSON found"
        echo -e "${policy}\tn/a\tn/a\tnojson" >> "$SUMMARY"
    fi
done

echo ""
echo "==================== ALL POLICIES DONE ($(date +%H:%M:%S)) ===================="
cat "$SUMMARY"
echo ""
echo "Per-policy logs + JSON: $LOGDIR/"
