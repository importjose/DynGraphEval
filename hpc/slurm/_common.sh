#!/usr/bin/env bash
# hpc/slurm/_common.sh — sourced by every model SLURM script
#
# Storage layout
# ──────────────
# DATA_DIR defaults to $HOME/dyngrapheval-data, which keeps large files
# out of the repo clone. Override at submission time:
#
#   DATA_DIR=/work/$USER/dyngrapheval sbatch hpc/slurm/train_tgn.sh
#   DATA_DIR=/scratchnvme/$USER/dyngrapheval sbatch hpc/slurm/train_tgn.sh
#
# On Athene you likely want /work or /scratchnvme rather than $HOME
# to avoid quota issues (datasets ~5GB, checkpoints ~500MB each).
#
# Directory layout under DATA_DIR:
#   datasets/          TGB dataset downloads (shared across all models)
#   checkpoints/       one subdir per model, e.g. checkpoints/tgn/tgbl-wiki/run0.pkl
#   results/           eval JSON outputs

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"

DATA_DIR="${DATA_DIR:-$HOME/dyngrapheval-data}"
DATASETS_DIR="$DATA_DIR/datasets"
CHECKPOINTS_DIR="$DATA_DIR/checkpoints"
RESULTS_DIR="$DATA_DIR/results"

mkdir -p "$DATASETS_DIR" "$CHECKPOINTS_DIR" "$RESULTS_DIR"

# Per-run defaults (override via env vars at submission)
DATASET="${DATASET:-tgbl-wiki}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"

# ── helpers ───────────────────────────────────────────────────────────────────
hpc_train() {
    local MODEL="$1"; shift          # e.g. "tgn"
    local EXTRA_ARGS=("$@")          # any extra args forwarded to train.py

    echo "========================================================"
    echo "[train] Job $SLURM_JOB_ID on $(hostname)"
    echo "[train] model=$MODEL dataset=$DATASET seed=$SEED epochs=$EPOCHS"
    echo "[train] DATA_DIR=$DATA_DIR"
    echo "========================================================"

    "$PYTHON" "$REPO_DIR/models/$MODEL/train.py" \
        --dataset       "$DATASET" \
        --seed          "$SEED" \
        --epochs        "$EPOCHS" \
        --patience      "$PATIENCE" \
        --datasets_cache  "$DATASETS_DIR" \
        --checkpoints_dir "$CHECKPOINTS_DIR/$MODEL" \
        "${EXTRA_ARGS[@]}"
}

hpc_eval() {
    local MODEL="$1"

    echo "========================================================"
    echo "[eval]  model=$MODEL dataset=$DATASET seed=$SEED"
    echo "========================================================"

    "$PYTHON" "$REPO_DIR/hpc/eval.py" \
        --model      "$MODEL" \
        --dataset    "$DATASET" \
        --seed       "$SEED" \
        --datasets_dir  "$DATASETS_DIR" \
        --results_dir   "$RESULTS_DIR"
}
