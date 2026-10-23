#!/usr/bin/env bash
# hpc/setup.sh — one-time environment setup on FAU Athene HPC (login node)
#
# Run once after cloning the repo:
#   bash hpc/setup.sh
#
# Creates a uv-managed virtual environment at .venv/ using Python 3.11
# and installs all project dependencies from pyproject.toml.
# torch + torchvision are installed separately with CUDA 12.1 index.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "[setup] Repo root: $REPO_DIR"

# ── 1. Ensure uv is available ─────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[setup] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv $(uv --version)"

# ── 2. Create venv with Python 3.11 ──────────────────────────────────────────
echo "[setup] Creating .venv with Python 3.11..."
uv venv --python 3.11 .venv

# ── 3. Install PyTorch with CUDA 12.1 ────────────────────────────────────────
# Must be done before the rest of the deps because torch-geometric depends on it.
echo "[setup] Installing torch 2.4.0 (CUDA 12.1)..."
uv pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 4. Install remaining project dependencies ─────────────────────────────────
# Install via -e . to pull in all deps from pyproject.toml, then immediately
# uninstall the package itself. The editable install registers a MetaPathFinder
# hook for DynGraphEval's models/ package, which shadows TGB_TPNet's models/
# directory even when PYTHONPATH is set. Uninstalling removes the hook while
# keeping all transitive dependencies (torch, py-tgb, etc.) in place.
echo "[setup] Installing project dependencies..."
uv pip install -e ".[dev]" 2>/dev/null || uv pip install -e .
echo "[setup] Unregistering DynGraphEval package (prevents models/ import conflict)..."
uv pip uninstall -y dyngrapheval

# ── 5. Install extra TGB_TPNet dependencies ───────────────────────────────────
echo "[setup] Installing TGB_TPNet dependencies..."
uv pip install numba scikit-learn scipy wandb

# ── 6. Smoke-test ─────────────────────────────────────────────────────────────
echo "[setup] Verifying torch + CUDA..."
.venv/bin/python - <<'EOF'
import torch
print(f"  torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
EOF

echo "[setup] Done. Activate with: source .venv/bin/activate"
