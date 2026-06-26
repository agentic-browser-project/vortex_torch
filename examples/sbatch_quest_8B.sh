#!/bin/bash
#SBATCH --job-name=quest-8B
#SBATCH --output=logs/sbatch/quest_8B_%j.out
#SBATCH --time=72:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G

hostname
nvidia-smi || true

CONDA_BASE=/vast/projects/liuv/pennnetworks/jiaheng/miniconda3
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate vortex-bench

cd /vast/projects/liuv/pennnetworks/jiaheng/vortex_torch
bash examples/algo_quest_8b.sh
