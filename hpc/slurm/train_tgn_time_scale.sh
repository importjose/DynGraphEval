#!/usr/bin/env bash
#SBATCH --job-name=dge-tgn-ts
#SBATCH --partition=shortq7
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:55:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.out
#SBATCH --export=ALL
#
# Runs ONE TGN granularity per job to fit the 6-hour GPU partition limit.
# Submit with: sbatch --export=ALL,GRANULARITY=hour hpc/slurm/train_tgn_time_scale.sh
#
# Submit:
#   sbatch hpc/slurm/train_tgn_time_scale.sh
#   DATA_DIR=/work/$USER/dge DATASET=tgbl-wiki SEED=1 sbatch hpc/slurm/train_tgn_time_scale.sh
set -euo pipefail
REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
source "$REPO_DIR/hpc/slurm/_common.sh"

case "${GRANULARITY:-}" in
    hour|day|week|month) ;;
    *) echo "Set GRANULARITY=hour|day|week|month at submission" >&2; exit 2 ;;
esac
GRANULARITIES=("$GRANULARITY")

_setup_job_repo

for GRAN in "${GRANULARITIES[@]}"; do
    MODEL_NAME="tgn_time_${GRAN}"
    RUN_DIR="$DATA_DIR/runs/$DATASET/$MODEL_NAME/seed${SEED}/${JOB_ID}"
    mkdir -p "$RUN_DIR"
    LOG_FILE="$RUN_DIR/run.log"

    echo "════════════════════════════════════════════════════════"
    echo "  Variant: $MODEL_NAME  ($(date))"
    echo "  Log: $LOG_FILE"
    echo "════════════════════════════════════════════════════════"

    "$PYTHON" "$REPO_DIR/models/tgn_time_scale/train.py" \
        --granularity     "$GRAN" \
        --run_id          "$JOB_ID" \
        --dataset         "$DATASET" \
        --seed            "$SEED" \
        --epochs          "$EPOCHS" \
        --patience        "$PATIENCE" \
        --repo_dir        "$JOB_REPO_DIR" \
        --datasets_cache  "$DATASETS_DIR" \
        --run_dir         "$RUN_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    "$PYTHON" "$REPO_DIR/hpc/eval.py" \
        --model           "$MODEL_NAME" \
        --checkpoint      "$RUN_DIR/checkpoint.pkl" \
        --output          "$RUN_DIR/metrics.json" \
        --dataset         "$DATASET" \
        --seed            "$SEED" \
        --datasets_dir    "$DATASETS_DIR" \
        --results_dir     "$RESULTS_DIR" \
        --neg_cache_dir   "$NEG_CACHE_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    echo "[done] $MODEL_NAME complete at $(date)" | tee -a "$LOG_FILE"
done

_cleanup_job_repo
echo "[done] All TGN time-scale variants complete."
