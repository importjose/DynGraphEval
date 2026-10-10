#!/usr/bin/env bash
#SBATCH --job-name=dge-tpnet
#SBATCH --partition=shortq7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Submit:
#   sbatch hpc/slurm/train_tpnet.sh
#   DATA_DIR=/work/$USER/dge DATASET=tgbl-review SEED=1 sbatch hpc/slurm/train_tpnet.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

hpc_train tpnet
hpc_eval  tpnet

echo "[done] TPNet train+eval complete."
