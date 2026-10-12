#!/usr/bin/env bash
#SBATCH --job-name=dge-tgn-ne
#SBATCH --partition=longq7-eng
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.out
#SBATCH --export=ALL
#
# Submit:
#   sbatch hpc/slurm/train_tgn_no_enc.sh
#   DATA_DIR=/work/$USER/dge DATASET=tgbl-review SEED=1 sbatch hpc/slurm/train_tgn_no_enc.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/hpc/slurm/_common.sh"

hpc_train tgn_no_enc
hpc_eval  tgn_no_enc

echo "[done] TGN+NoEnc train+eval complete."
