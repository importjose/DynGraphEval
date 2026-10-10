#!/usr/bin/env bash
# hpc/prefetch.sh — download TGB datasets on the login node before job submission
#
# Compute nodes on Athene have no internet access. Run this once from the
# login node to pre-download datasets into data/datasets/.
#
# Usage:
#   bash hpc/prefetch.sh                        # downloads tgbl-wiki (default)
#   bash hpc/prefetch.sh tgbl-review tgbl-coin  # multiple datasets
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Match the DATA_DIR convention used by the SLURM scripts
DATA_DIR="${DATA_DIR:-$HOME/dyngrapheval-data}"
DATASETS_DIR="$DATA_DIR/datasets"
mkdir -p "$DATASETS_DIR"
echo "[prefetch] DATA_DIR=$DATA_DIR"

PYTHON="${REPO_DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: .venv not found. Run hpc/setup.sh first." >&2
    exit 1
fi

DATASETS=("${@:-tgbl-wiki}")

for DS in "${DATASETS[@]}"; do
    echo "[prefetch] Downloading $DS → $DATASETS_DIR ..."
    "$PYTHON" - <<EOF
from tgb.linkproppred.dataset import LinkPropPredDataset
ds = LinkPropPredDataset(name="$DS", root="$DATASETS_DIR", preprocess=True)
print(f"  {ds.name}: {len(ds.get_TemporalData())} edges")
EOF
    echo "[prefetch] $DS done."
done

echo "[prefetch] All datasets ready in $DATASETS_DIR"
