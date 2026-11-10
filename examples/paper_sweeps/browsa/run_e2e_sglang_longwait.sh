#!/bin/bash
# WebVoyager end-to-end evaluation against the VENDORED SGLANG server with a
# BrowserSparseAttention decode backend.  Defaults to bsa_v2; set
# SGLANG_DECODE_BACKEND=bsa for the mixmax_wn-capable reference backend.
#
# Usage: run_e2e_sglang.sh <port_hint> <out_root> [task_num] [batch_size]
# Env:   SGLANG_CUDA_GRAPH=0  -> disable decode CUDA graphs (enabled by default)
#        SGLANG_CUDA_GRAPH_MAX_BS -> largest captured batch bucket (defaults
#                                    to the WebVoyager concurrency)
#        BSA2_REQUIRE_SPARSE=1 -> abort on any decode fallback
#        SGLANG_MEM_FRACTION_STATIC -> leave BSA centroid-cache headroom
#        SGLANG_DECODE_BACKEND -> bsa_v2 (default) or bsa
#        SGLANG_PAGE_SIZE -> KV-cache page size (default 16)
#        BSA2_* knobs pass through to the backend (defaults = r15 flat config)
set -euo pipefail

PORT_HINT="${1:?port hint}"
OUT_ROOT="${2:?out root}"
TASK_NUM="${3:-200}"
BATCH_SIZE="${4:-12}"

DECODE_BACKEND="${SGLANG_DECODE_BACKEND:-bsa_v2}"
MODE="${SGLANG_MODE:-sglang_${DECODE_BACKEND}}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-16}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_DIR_OVERRIDE:-$OUT_ROOT/${TIMESTAMP}_b${BATCH_SIZE}}"

MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-32B"
TSA_DIR="/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention"
BSA_CODE_DIR="${BSA_CODE_DIR:-$TSA_DIR}"
case "$DECODE_BACKEND" in
    bsa) BACKEND_FILE="$BSA_CODE_DIR/bsa_sglang/backend.py" ;;
    bsa_v2) BACKEND_FILE="$BSA_CODE_DIR/bsa_sglang/backend_v2.py" ;;
    # variant backends (bsa_v2_tuned_packing, bsa_v2_postsort, ...) register
    # themselves when backend_v2 loads -- same module file anchors the path.
    bsa_v2_*) BACKEND_FILE="$BSA_CODE_DIR/bsa_sglang/backend_v2.py" ;;
    *) echo "Unsupported SGLANG_DECODE_BACKEND=$DECODE_BACKEND" >&2; exit 2 ;;
esac
SIM_DIR="/vast/projects/liuv/pennnetworks/jiaheng/browser-simulator/browser-use"
CONDA_SH="/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/etc/profile.d/conda.sh"
# The vendored sglang v0.5.9 runs ONLY in vortex-bench (sgl_kernel 0.3.21);
# the 'sglang' env's sgl_kernel is too old and dies at import.
VB_PYTHON="/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/vortex-bench/bin/python"
WV_PYTHON="/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/webvoyager/bin/python"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    PORT_BASE=$(( 21000 + (SLURM_JOB_ID % 40000) ))
else
    PORT_BASE="$PORT_HINT"
fi
PORT=$(python3 -c "
import socket
for p in range(int('$PORT_BASE'), int('$PORT_BASE') + 100):
    try:
        s = socket.socket(); s.bind(('127.0.0.1', p)); s.close(); print(p); break
    except OSError:
        continue
")
[ -n "$PORT" ] || { echo "no free port" >&2; exit 1; }

LOG_DIR="${TSA_DIR}/results/webvoyager_e2e/logs"
mkdir -p "$LOG_DIR" "$OUT_DIR"
RUN_ID="${SLURM_JOB_ID:-local_$$}"
SERVER_LOG="${LOG_DIR}/server_${MODE}_${RUN_ID}.log"
SIM_LOG="${LOG_DIR}/sim_${MODE}_${RUN_ID}.log"
GRADE_VAR="${MODE}_$(basename "$OUT_DIR")_${SLURM_JOB_ID:-$$}"
GRADE_LOG="${LOG_DIR}/grade_${GRADE_VAR}.log"

SERVER_PGID=""
GRADER_PID=""
stop_server() {
    [ -z "${SERVER_PGID:-}" ] && return
    kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    sleep 10
    kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
    SERVER_PGID=""
}
stop_grader() {
    if [ -n "${GRADER_PID:-}" ] && kill -0 "$GRADER_PID" 2>/dev/null; then
        kill "$GRADER_PID" 2>/dev/null || true
        wait "$GRADER_PID" 2>/dev/null || true
    fi
    GRADER_PID=""
}
start_incremental_grader() {
    [ -f "$TSA_DIR/gemini-api-key.txt" ] || return
    ln -sfn "$OUT_DIR" "$TSA_DIR/results/webvoyager_e2e/results/$GRADE_VAR"
    (
        cd "$TSA_DIR/results/webvoyager_e2e"
        if [ "${SIM_HEADLESS:-1}" = "0" ]; then
            provenance=headed
        else
            provenance=headless
        fi
        exec env GRADE_WORKERS="${GRADE_WORKERS:-4}" \
            GRADE_PROVENANCE="$provenance" python3 grade_webvoyager.py \
            --watch --expected-tasks "$TASK_NUM" --poll-seconds 15 "$GRADE_VAR"
    ) > "$GRADE_LOG" 2>&1 &
    GRADER_PID=$!
    echo "[$(date)] Incremental grader PID=$GRADER_PID log=$GRADE_LOG"
}
cleanup() {
    rc=$?
    stop_grader
    stop_server
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ── libcudart.so.13 shim (flashinfer sm100 JIT cache links against cudart 13,
#    which vortex-bench does not ship) ──
SHIM_DIR="/tmp/bsa2_ld_$$"
mkdir -p "$SHIM_DIR"
ln -sf /vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/sglang/lib/libcudart.so.13* "$SHIM_DIR/" 2>/dev/null || true
ln -sf "$SHIM_DIR"/libcudart.so.13.* "$SHIM_DIR/libcudart.so.13" 2>/dev/null || true

# Radix/prefix cache: ON by default for e2e (agent steps re-send a growing
# prompt; prefix reuse skips most prefill). SGLANG_RADIX_CACHE=0 restores the
# old behavior for controlled TPOT measurement runs.
RADIX_ARGS=""
if [ "${SGLANG_RADIX_CACHE:-1}" != "1" ]; then
    RADIX_ARGS="--disable-radix-cache"
fi

GRAPH_ARGS=(--disable-cuda-graph)
if [ "${SGLANG_CUDA_GRAPH:-1}" = "1" ]; then
    GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-$BATCH_SIZE}"
    GRAPH_BUCKETS=()
    for bucket in 1 2 4 8 12 16; do
        [ "$bucket" -le "$GRAPH_MAX_BS" ] && GRAPH_BUCKETS+=("$bucket")
    done
    for ((bucket = 24; bucket <= GRAPH_MAX_BS; bucket += 8)); do
        GRAPH_BUCKETS+=("$bucket")
    done
    if [ "${GRAPH_BUCKETS[-1]:-0}" -ne "$GRAPH_MAX_BS" ]; then
        GRAPH_BUCKETS+=("$GRAPH_MAX_BS")
    fi
    GRAPH_ARGS=(--cuda-graph-bs "${GRAPH_BUCKETS[@]}")
    echo "[e2e-sglang] CUDA graphs ENABLED (buckets=${GRAPH_BUCKETS[*]})"
fi

echo "=========================================="
echo "WebVoyager E2E (sglang $DECODE_BACKEND): port=$PORT tasks=$TASK_NUM conc=$BATCH_SIZE"
echo "Output: $OUT_DIR"
echo "Code root: $BSA_CODE_DIR"
echo "Repo HEAD: $(git -C "$BSA_CODE_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Backend SHA256: $(sha256sum "$BACKEND_FILE" | cut -d' ' -f1)"
echo "Decode backend: $DECODE_BACKEND; page size: $SGLANG_PAGE_SIZE"
echo "Strict sparse: ${BSA2_REQUIRE_SPARSE:-1}"
echo "SGLang static memory fraction: ${SGLANG_MEM_FRACTION_STATIC:-0.80}"
echo "=========================================="

# 1. Launch the vendored sglang server
(
    set +u; source "$CONDA_SH"; conda activate vortex-bench; set -u
    # boundary_registration/sitecustomize.py is INERT unless
    # BSA_SGLANG_DIAGNOSTIC=1: it then registers the bsa_v2_* variant
    # backends (incl. bsa_v2_tuned_packing) and patches the argparse choices
    # before launch_server parses them.
    export PYTHONPATH="$BSA_CODE_DIR/results/webvoyager_e2e/e2e_boundary_registration:$BSA_CODE_DIR:$BSA_CODE_DIR/3rdparty/sglang/python:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="$SHIM_DIR:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
    export CUDA_HOME=/vast/parcc/spack/sw/apps/linux-sapphirerapids/cuda-13.1.0_590.44.01
    export PATH="$CUDA_HOME/bin:$PATH"
    export SGLANG_DISABLE_CUDNN_CHECK=1
    export BSA2_REQUIRE_SPARSE="${BSA2_REQUIRE_SPARSE:-1}"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    exec "$VB_PYTHON" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host 127.0.0.1 --port "$PORT" \
        --attention-backend flashinfer \
        --decode-attention-backend "$DECODE_BACKEND" \
        --page-size "$SGLANG_PAGE_SIZE" \
        "${GRAPH_ARGS[@]}" \
        ${RADIX_ARGS} \
        --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.80}" \
        --max-running-requests "$BATCH_SIZE" \
        --served-model-name "$MODE" \
        ${SGLANG_LOG_REQUESTS:+--log-requests --log-requests-level 0} \
        --trust-remote-code --log-level info
) > "$SERVER_LOG" 2>&1 &
SERVER_PGID=$!
echo "  Server PGID: $SERVER_PGID, log: $SERVER_LOG"

# 2. Wait for health
echo "[$(date)] Waiting for server..."
WAITED=0
while ! curl -fsS "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; do
    sleep 5; WAITED=$((WAITED + 5))
    if [ $WAITED -ge 2400 ]; then
        echo "ERROR: server not healthy in 2400s"; tail -30 "$SERVER_LOG"; exit 1
    fi
    if ! kill -0 "$SERVER_PGID" 2>/dev/null; then
        echo "ERROR: server died"; tail -60 "$SERVER_LOG"; exit 1
    fi
done
echo "[$(date)] Server ready after ${WAITED}s"

# Preflight: real batched requests before launching browsers
if ! "$WV_PYTHON" "$TSA_DIR/results/webvoyager_e2e/preflight_server.py" \
        --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODE" --batch-size 2; then
    echo "ERROR: preflight failed"; tail -80 "$SERVER_LOG"; exit 1
fi
echo "BATCHED_PREFLIGHT_PASSED"

# 3. Simulator (headed via containerized Xvfb — ALWAYS headed)
cd "$SIM_DIR"
export USE_TSA=1
export TSA_BASE_URL="http://127.0.0.1:${PORT}/v1"
export TSA_MODEL="$MODE"
export TSA_API_KEY="EMPTY"
APPT=/vast/parcc/spack/sw/apps/linux-sapphirerapids/apptainer-1.4.1-qpr4lterya7ontg7pkqrx7b3jkab3lcw/bin/apptainer
SIF=/vast/projects/liuv/pennnetworks/jiaheng/bsa_val/xvfb.sif
DISP=$(( 100 + ($$ % 90) )); rm -f /tmp/.X${DISP}-lock 2>/dev/null || true
"$APPT" exec "$SIF" Xvfb :$DISP -screen 0 1280x1024x24 -nolisten tcp > "$LOG_DIR/xvfb_${MODE}.log" 2>&1 &
XVFB_PID=$!; sleep 8
if [ -S /tmp/.X11-unix/X$DISP ]; then
    export DISPLAY=:$DISP SIM_HEADLESS=0
    echo "[xvfb] display :$DISP up — running HEADED"
else
    echo "WARNING: Xvfb failed, falling back to headless"; export SIM_HEADLESS=1
fi
export PLAYWRIGHT_BROWSERS_PATH="/vast/home/j/jiaheng/.cache/ms-playwright"

start_incremental_grader

set +e
$WV_PYTHON -m simulator capture \
    --no-vision --task-num "$TASK_NUM" --batch-size "$BATCH_SIZE" \
    --source "${SIM_SOURCE:-webvoyager}" --shuffle --max-steps 15 \
    ${TASK_IDS_FILE:+--task-ids-file "$TASK_IDS_FILE"} \
    --task-timeout 5400 --llm-timeout 1200 \
    --out-dir "$OUT_DIR" 2>&1 | tee "$SIM_LOG"
SIM_EXIT=${PIPESTATUS[0]}
set -e
kill $XVFB_PID 2>/dev/null || true
echo "[$(date)] Simulator finished rc=$SIM_EXIT"

if ! kill -0 "$SERVER_PGID" 2>/dev/null; then
    echo "ERROR: SGLang server exited before the simulator completed"
    SIM_EXIT=1
fi
if [ "$DECODE_BACKEND" = "bsa_v2" ] && grep -q "STRICT SPARSE VIOLATION" "$SERVER_LOG"; then
    echo "ERROR: strict sparse decode invariant was violated"
    SIM_EXIT=1
fi

stop_server

# 4. Validate + auto-grade
VALIDATE_EXIT=0
"$WV_PYTHON" "$TSA_DIR/results/webvoyager_e2e/validate_run.py" \
    "$OUT_DIR" --expected-tasks "$TASK_NUM" \
    --server-log "$SERVER_LOG" --sim-log "$SIM_LOG" || VALIDATE_EXIT=$?
if [ "$VALIDATE_EXIT" -ne 0 ]; then
    SIM_EXIT="$VALIDATE_EXIT"
fi
if [ -n "${GRADER_PID:-}" ] && [ "$SIM_EXIT" -eq 0 ]; then
    echo "[$(date)] Waiting for incremental grader to flush final tasks..."
    if ! wait "$GRADER_PID"; then
        echo "WARNING: incremental grader failed; retrying terminal tasks once"
        (cd "$TSA_DIR/results/webvoyager_e2e" && \
            python3 grade_webvoyager.py "$GRADE_VAR") || \
            echo "grading retry failed (non-fatal)"
    fi
    GRADER_PID=""
elif [ "$SIM_EXIT" -ne 0 ]; then
    stop_grader
    echo "Skipping grading because the sparse-only run is invalid"
fi
bash /vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention/slack_notify.sh \
    "[$MODE] run finished rc=$SIM_EXIT tasks=$TASK_NUM out=$OUT_DIR" || true
echo "E2E_SGLANG_DONE rc=$SIM_EXIT"
exit "$SIM_EXIT"
