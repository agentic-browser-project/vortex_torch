#!/usr/bin/env bash
# Run the TreeSparseAttention `tpot-no-share` batch benchmark on the SAME
# request.json the quest benchmark uses, so the three-way comparison is fair.
#
# TreeSparseAttention is a separate project with its own Python 3.13 venv and
# pre-built CUDA kernels. This script activates THAT environment (never the
# quest .venv), runs the benchmark, and copies the results JSON back into this
# benchmark's results/ directory as treesparse_raw.json.
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSA="${TSA_DIR:-/vast/projects/liuv/pennnetworks/xutingl/sparse_attn/TreeSparseAttention}"
REQUEST="$BENCH/request.json"
DEST="$BENCH/results/treesparse_raw.json"

[ -f "$REQUEST" ] || { echo "ERROR: request.json missing at $REQUEST" >&2; exit 1; }
[ -d "$TSA" ]     || { echo "ERROR: TreeSparseAttention missing at $TSA" >&2; exit 1; }

mkdir -p "$BENCH/results"

# Lmod's `module` function is installed by the system profile, which a
# non-interactive script does not load -- source it so env.sh can `module load`.
if ! command -v module >/dev/null 2>&1; then
  for init in /etc/profile.d/Z98-lmod.sh /etc/profile.d/modules.sh \
              /usr/share/lmod/lmod/init/bash; do
    # shellcheck disable=SC1090
    [ -f "$init" ] && source "$init" && break
  done
fi

# Marker so we can confirm the run produced a NEW results JSON (not a stale one).
# The EXIT trap removes it on any exit -- normal, error, or signal.
STAMP="$(mktemp)"
trap 'rm -f "$STAMP"' EXIT

echo ">>> activating TreeSparseAttention environment ($TSA)"
cd "$TSA"
# env.sh loads CUDA/toolchain modules, then activates TreeSparse's own .venv.
# shellcheck disable=SC1091
source env.sh
# env.sh ends in `unset`, so `source` returns 0 even if `module load` failed --
# verify the .venv actually activated rather than trusting that exit code.
PY_PATH="$(command -v python || true)"
if [ "$PY_PATH" != "$TSA/.venv/bin/python" ]; then
  echo "ERROR: env.sh did not activate TreeSparse's .venv" >&2
  echo "       (python resolved to: ${PY_PATH:-none}; expected $TSA/.venv/bin/python)" >&2
  exit 1
fi

echo ">>> run_batch_experiments.sh tpot-no-share  (request: $REQUEST)"
# run_batch_experiments.sh forwards every argument after the mode word to
# benchmark_batch.py. The trailing --request-file is therefore appended after
# the script's hardcoded default; argparse keeps the LAST value, so TreeSparse
# runs on the SAME input as the quest/dense runs.
#
# The script's own exit code is intentionally ignored: its final (optional)
# plotting step runs AFTER the results JSON is written and may fail without
# affecting the measurement. Success is verified by locating the JSON below.
bash run_batch_experiments.sh tpot-no-share --request-file "$REQUEST" || true

LATEST="$(ls -t "$TSA"/batch_results/tpot_no_share/results_*/results_*.json \
          2>/dev/null | head -1)"
if [ -z "$LATEST" ]; then
  echo "ERROR: no TreeSparse results JSON under $TSA/batch_results/tpot_no_share/" >&2
  exit 1
fi
if [ ! "$LATEST" -nt "$STAMP" ]; then
  echo "ERROR: newest TreeSparse JSON ($LATEST) predates this run -- the" >&2
  echo "       benchmark produced no fresh results. Check the run log." >&2
  exit 1
fi

cp "$LATEST" "$DEST"
# Use `python` (verified above to be TreeSparse's venv interpreter) rather than
# a bare `python3` that could resolve elsewhere on PATH.
python -c "import json; d=json.load(open('$DEST')); \
print('[treesparse]', len(d), 'batch sizes:', sorted(int(k) for k in d))" || {
  echo "ERROR: $DEST is not valid JSON" >&2; exit 1; }

echo ">>> TreeSparse results: $LATEST"
echo "                     -> $DEST"
