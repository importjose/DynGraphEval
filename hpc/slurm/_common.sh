#!/usr/bin/env bash
# hpc/slurm/_common.sh — sourced by every model SLURM script
#
# Storage layout
# ──────────────
# DATA_DIR defaults to $HOME/dyngrapheval-data. Override at submission:
#
#   DATA_DIR=/work/$USER/dyngrapheval sbatch hpc/slurm/train_tgn.sh
#
# On Athene use /work or /scratchnvme to avoid home quota (~5GB datasets,
# ~500MB per checkpoint).
#
# Directory layout under DATA_DIR:
#   datasets/                        TGB downloads, shared across all jobs
#   checkpoints/{model}/{dataset}/   run{seed}.pkl per trained model
#   results/                         eval JSON outputs

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"

# Guard: fail clearly if setup hasn't been run
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: .venv not found at $REPO_DIR/.venv"
    echo "       Run 'bash hpc/setup.sh' on the login node first."
    exit 1
fi

DATA_DIR="${DATA_DIR:-$HOME/dyngrapheval-data}"
DATASETS_DIR="$DATA_DIR/datasets"
CHECKPOINTS_DIR="$DATA_DIR/checkpoints"
RESULTS_DIR="$DATA_DIR/results"
OUTPUTS_DIR="$REPO_DIR/outputs"

mkdir -p "$DATASETS_DIR" "$CHECKPOINTS_DIR" "$RESULTS_DIR" "$OUTPUTS_DIR"

# Per-run defaults (override via env vars at submission time)
DATASET="${DATASET:-tgbl-wiki}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"

# Safe job ID — $SLURM_JOB_ID is undefined when running outside SLURM
JOB_ID="${SLURM_JOB_ID:-local}"

# ── Per-job working copy of src/TGB_TPNet ─────────────────────────────────────
# _apply_patches() writes model-specific files into the repo working dir.
# Multiple concurrent jobs would race over the shared src/TGB_TPNet/ if they
# all used it directly. Each job gets its own copy under /tmp so patches never
# collide. /tmp is node-local storage (fast, not shared).
JOB_REPO_DIR="/tmp/TGB_TPNet_${JOB_ID}"

_setup_job_repo() {
    if [[ -d "$JOB_REPO_DIR" ]]; then
        echo "[common] Re-using existing job repo: $JOB_REPO_DIR"
    else
        echo "[common] Copying src/TGB_TPNet → $JOB_REPO_DIR"
        cp -r "$REPO_DIR/src/TGB_TPNet" "$JOB_REPO_DIR"
    fi
}

_cleanup_job_repo() {
    if [[ -d "$JOB_REPO_DIR" ]]; then
        echo "[common] Cleaning up job repo: $JOB_REPO_DIR"
        rm -rf "$JOB_REPO_DIR"
    fi
}

# ── helpers ───────────────────────────────────────────────────────────────────
hpc_train() {
    local MODEL="$1"; shift          # e.g. "tgn"
    local EXTRA_ARGS=("$@")          # any extra args forwarded to train.py

    # ── Named log file ────────────────────────────────────────────────────────
    # #SBATCH --output writes to outputs/%x_%j.out (model name + job ID).
    # Tee to a second file with full context in the name.
    local TIMESTAMP
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    local LOG_FILE="$OUTPUTS_DIR/${MODEL}_seed${SEED}_${DATASET}_${TIMESTAMP}_${JOB_ID}.log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "[log] $LOG_FILE"

    _setup_job_repo

    echo "========================================================"
    echo "[train] Job $JOB_ID on $(hostname)"
    echo "[train] model=$MODEL  dataset=$DATASET  seed=$SEED  epochs=$EPOCHS"
    echo "[train] DATA_DIR=$DATA_DIR"
    echo "[train] repo_dir=$JOB_REPO_DIR"
    echo "========================================================"

    "$PYTHON" "$REPO_DIR/models/$MODEL/train.py" \
        --dataset         "$DATASET" \
        --seed            "$SEED" \
        --epochs          "$EPOCHS" \
        --patience        "$PATIENCE" \
        --repo_dir        "$JOB_REPO_DIR" \
        --datasets_cache  "$DATASETS_DIR" \
        --checkpoints_dir "$CHECKPOINTS_DIR/$MODEL" \
        "${EXTRA_ARGS[@]}"
}

hpc_eval() {
    local MODEL="$1"

    echo "========================================================"
    echo "[eval]  Job $JOB_ID | model=$MODEL  dataset=$DATASET  seed=$SEED"
    echo "========================================================"

    "$PYTHON" "$REPO_DIR/hpc/eval.py" \
        --model            "$MODEL" \
        --dataset          "$DATASET" \
        --seed             "$SEED" \
        --datasets_dir     "$DATASETS_DIR" \
        --checkpoints_dir  "$CHECKPOINTS_DIR" \
        --results_dir      "$RESULTS_DIR"

    _cleanup_job_repo
}
