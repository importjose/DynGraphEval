#!/usr/bin/env bash
#SBATCH --job-name=dge-bin-occupancy
#SBATCH --partition=shortq7
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.out
#SBATCH --export=ALL
# CPU-only data diagnostic. Run on a compute node, not the login node.
set -euo pipefail
REPO_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}" && pwd)"
source "$REPO_DIR/hpc/slurm/_common.sh"
DATASET="${DATASET:-tgbl-wiki}"
"$PYTHON" "$REPO_DIR/scripts/analyze_time_bins.py" \
    --dataset "$DATASET" \
    --datasets-dir "$DATASETS_DIR" \
    --num-neighbors 20 \
    --output "$DATA_DIR/bin_occupancy/$DATASET.json"
