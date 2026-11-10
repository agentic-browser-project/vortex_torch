#!/bin/bash
#SBATCH --job-name=claude-gaia-defjudge
#SBATCH --output=/vast/projects/liuv/pennnetworks/jiaheng/xqa_bench_log/claude_gaia_defjudge_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
# Judge the BrowSA GAIA cells (Setting B, 6.25% / 12.5%) with the standard success
# protocol; writes webvoyager_eval.json into every task dir (resumable).
set -uo pipefail
cd /vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention/results/webvoyager_e2e
PY=/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/webvoyager/bin/python
i=0
for cfg in dp_gaia_setB_r0p0625 dp_gaia_setB_r0p125; do
  for r in 1 2 3; do
    sess=$(ls -dt results/mixmax_wn_sweep/sglang_packed/$cfg/run$r/2026* | head -1)
    $PY claude_default_judge_gaia.py "$sess" > /vast/projects/liuv/pennnetworks/jiaheng/xqa_bench_log/defjudge_${cfg}_r${r}.log 2>&1 &
    i=$((i+1)); [ $((i % 2)) -eq 0 ] && wait
  done
done
wait
grep -h DEFAULTJUDGE_CELL /vast/projects/liuv/pennnetworks/jiaheng/xqa_bench_log/defjudge_dp_gaia_*.log
echo DEFJUDGE_DONE
