#!/usr/bin/env bash
#SBATCH --job-name=dge-gm-ms
#SBATCH --partition=shortq7
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:45:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.out
#SBATCH --export=ALL
#
# Submit:
#   sbatch hpc/slurm/train_graphmixer_multiscale.sh
#   DATA_DIR=/work/$USER/dge DATASET=tgbl-review SEED=1 sbatch hpc/slurm/train_graphmixer_multiscale.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/hpc/slurm/_common.sh"

hpc_train graphmixer_multiscale
hpc_eval  graphmixer_multiscale

echo "[done] MultiScaleGraphMixer train+eval complete."
