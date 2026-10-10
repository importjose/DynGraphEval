#!/usr/bin/env bash
#SBATCH --job-name=dge-tgn
#SBATCH --partition=shortq7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# Usage: sbatch hpc/slurm/train_tgn.sh [--dataset tgbl-wiki] [--seed 0]
#        Override defaults via env vars: DATASET, SEED, EPOCHS, PATIENCE

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"

DATASET="${DATASET:-tgbl-wiki}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"

echo "[slurm] Job $SLURM_JOB_ID on $(hostname)"
echo "[slurm] TGN | dataset=$DATASET seed=$SEED epochs=$EPOCHS"

"$PYTHON" "$REPO_DIR/models/tgn/train.py" \
    --dataset       "$DATASET" \
    --seed          "$SEED" \
    --epochs        "$EPOCHS" \
    --patience      "$PATIENCE" \
    --datasets_cache "$REPO_DIR/data/datasets" \
    --checkpoints_dir "$REPO_DIR/data/checkpoints/tgn"

echo "[slurm] Done."
