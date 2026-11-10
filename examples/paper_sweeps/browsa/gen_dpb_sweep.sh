#!/bin/bash
# Packed BSA sweep with distinct_page_budget (2026-09-22): precise page-count
# budget (real ratio ~= nominal, floor-based like vortex), count_generation_reservation,
# per-step page telemetry, e2e time. Ratios 6.25% + 12.5% (3.125% skipped for now).
# 2 settings x 2 ratios x 3 runs x {vtx-200, rest-443} = 24 jobs.
#   A: ps64/max32/wn32/C_cap4096/mem0.60   B: ps32/max16/wn16/C_cap4096/mem0.55
# usage: gen_dpb_sweep.sh <A|B> <ratio> <vtx|rest> <run>
set -eo pipefail
E2E=/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention/results/webvoyager_e2e
SB=$E2E/mixmax_sweep_sbatch
ROOT=/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention
emit() {
  local S=$1 R=$2 HALF=$3 RUN=$4
  if [ "$S" = A ]; then PS=64; MAXC=32; WNREF=32; MEMF=0.60
  elif [ "$S" = C ]; then PS=32; MAXC=32; WNREF=32; MEMF=0.55
  else PS=32; MAXC=16; WNREF=16; MEMF=0.55; fi
  local CCAP=4096
  if [ "$HALF" = vtx ]; then TF=$E2E/subset_vortex200.txt; NT=200; TIME=05:00:00
  else TF=$E2E/subset_rest443.txt; NT=443; case "$S" in B) TIME=14:00:00;; C) TIME=12:00:00;; *) TIME=09:00:00;; esac; fi
  local T="r${R//./p}"
  local OUT=$E2E/results/mixmax_wn_sweep/sglang_packed_distinctpage/set${S}_${T}_ps${PS}_${HALF}/run${RUN}
  local f=$SB/dpb_${S}_${T}_${HALF}_r${RUN}.sbatch
  cat > "$f" <<SB
#!/bin/bash
#SBATCH --job-name=claude-dpb${S}${HALF}-${T}-r${RUN}
#SBATCH --output=/vast/projects/liuv/pennnetworks/jiaheng/xqa_bench_log/dpb${S}_${HALF}_${T}_r${RUN}_%A.out
#SBATCH --time=${TIME}
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
cd $E2E
export SGLANG_DECODE_BACKEND=bsa_v2_tuned_packing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BSA2_E2E_BOUNDARY=1
export BSA_GRID_ROOT=$ROOT
export BSA_TUNED_ATTENTION=tc_no_split
export SGLANG_MEM_FRACTION_STATIC=${MEMF}
export SGLANG_CUDA_GRAPH=1
export SGLANG_CUDA_GRAPH_MAX_BS=16
export SGLANG_RADIX_CACHE=0
export SGLANG_PAGE_SIZE=${PS}
export BSA2_KERNELS_DIR=$ROOT/build
export BSA2_CHUNK_MODE=semantic
export BSA2_CHUNK_EXTRACT=semantic_v2
export BSA2_MIN_CHUNK=6
export BSA2_MAX_CHUNK=${MAXC}
export BSA2_SCORE_METHOD=mixmax_wn
export BSA2_WN_ALPHA=0.25
export BSA2_WN_REF=${WNREF}
export BSA2_HEAD_SELECT=per_kv
export BSA2_QUERY_COMBINE=mean_q
export BSA2_GEN_CHUNKS=0
export BSA2_RESERVED_BOS_PAGES=0
export BSA2_RESERVED_RECENT_PAGES=0
export BSA2_STATIC_PACK=d_worst
export BSA2_PACK_PRESERVE_RESERVED=1
export BSA2_ALWAYS_INCLUDE_START_PAGES=1
export BSA2_ALWAYS_INCLUDE_RECENT_PAGES=2
export BSA2_PACK_CLUSTERING=gpu
export BSA2_GRAPH_MAX_CHUNKS=${CCAP}
export BSA_SELECTION_POLICY=distinct_page_budget
export BSA2_BUDGET_CONVENTION=count_generation_reservation
export BSA2_TOKEN_SELECT_RATIO=${R}
export BSA2_REQUIRE_SPARSE=1
export BSA2_LOG_EVERY=20
export SIM_STEP_TIMEOUT=1200
export TASK_IDS_FILE=${TF}
OUT=${OUT}
mkdir -p "\$OUT"
export BSA2_PAGE_TELEMETRY=\$OUT/packed_selection.jsonl
export BSA2_PAGE_TELEMETRY_EVERY=1
bash run_e2e_sglang_longwait.sh 10011 "\$OUT" ${NT} 16
SB
  echo "$f"
}
emit "$@"
