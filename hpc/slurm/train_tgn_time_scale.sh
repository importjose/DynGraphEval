#!/usr/bin/env bash
#SBATCH --job-name=dge-tgn-ts
#SBATCH --partition=shortq7
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=9:00:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.out
#SBATCH --export=ALL
#
# Runs all 4 TGN temporal-granularity variants sequentially (hour→day→week→month).
# Each variant trains TGN with timestamps scaled to that resolution, then evaluates.
#
# NOTE: 4 × ~2h = ~8h. If shortq7 max time (6h) is exceeded on your cluster,
#       split into two jobs (e.g. GRANULARITIES=("hour" "day") and ("week" "month")).
#
# Submit:
#   sbatch hpc/slurm/train_tgn_time_scale.sh
#   DATA_DIR=/work/$USER/dge DATASET=tgbl-wiki SEED=1 sbatch hpc/slurm/train_tgn_time_scale.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_DIR/hpc/slurm/_common.sh"

GRANULARITIES=("hour" "day" "week" "month")

_setup_job_repo

for GRAN in "${GRANULARITIES[@]}"; do
    MODEL_NAME="tgn_time_${GRAN}"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$OUTPUTS_DIR/${MODEL_NAME}_seed${SEED}_${DATASET}_${TIMESTAMP}_${JOB_ID}.log"

    echo "════════════════════════════════════════════════════════"
    echo "  Variant: $MODEL_NAME  ($(date))"
    echo "  Log: $LOG_FILE"
    echo "════════════════════════════════════════════════════════"

    "$PYTHON" "$REPO_DIR/models/tgn_time_scale/train.py" \
        --granularity     "$GRAN" \
        --dataset         "$DATASET" \
        --seed            "$SEED" \
        --epochs          "$EPOCHS" \
        --patience        "$PATIENCE" \
        --repo_dir        "$JOB_REPO_DIR" \
        --datasets_cache  "$DATASETS_DIR" \
        --checkpoints_dir "$CHECKPOINTS_DIR/$MODEL_NAME" \
        2>&1 | tee -a "$LOG_FILE"

    "$PYTHON" "$REPO_DIR/hpc/eval.py" \
        --model           "$MODEL_NAME" \
        --dataset         "$DATASET" \
        --seed            "$SEED" \
        --datasets_dir    "$DATASETS_DIR" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --results_dir     "$RESULTS_DIR" \
        --neg_cache_dir   "$NEG_CACHE_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    echo "[done] $MODEL_NAME complete at $(date)" | tee -a "$LOG_FILE"
done

_cleanup_job_repo
echo "[done] All TGN time-scale variants complete."
