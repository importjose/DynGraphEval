MLFLOW_DB := sqlite:///$(CURDIR)/mlflow.db

mlflow:
	MLFLOW_TRACKING_URI=$(MLFLOW_DB) mlflow ui

# ── HPC (FAU Athene) ──────────────────────────────────────────────────────────
# Required: set HPC_HOST to your Athene login node
#   export HPC_HOST=woody.rrze.uni-erlangen.de
#
# DATA_DIR controls where datasets, checkpoints, and results are stored
# on the HPC. Defaults to $HOME/dyngrapheval-data on the remote node.
# Override if your home quota is small:
#   make hpc-train MODEL=tgn HPC_DATA_DIR=/work/$$USER/dyngrapheval
#
# Storage layout on HPC (under DATA_DIR):
#   datasets/                  TGB downloads, shared across models (~5GB)
#   checkpoints/{model}/{ds}/  run{seed}.pkl per trained model
#   results/                   eval JSON outputs
HPC_HOST     ?= woody.rrze.uni-erlangen.de
HPC_DIR      ?= ~/DynGraphEval
HPC_DATA_DIR ?= ~/dyngrapheval-data

# Copy repo to HPC (excludes large/generated artifacts)
hpc-push:
	rsync -avz --exclude='.venv' --exclude='__pycache__' \
	    --exclude='*.pyc' --exclude='.git' --exclude='mlruns' \
	    --exclude='data/' --exclude='results/' \
	    . $(HPC_HOST):$(HPC_DIR)/

# One-time environment setup on HPC login node
hpc-setup:
	ssh $(HPC_HOST) "cd $(HPC_DIR) && bash hpc/setup.sh"

# Download datasets on login node before submitting jobs (compute nodes have no internet)
# Usage: make hpc-prefetch DATASETS="tgbl-wiki tgbl-review"
hpc-prefetch:
	ssh $(HPC_HOST) "cd $(HPC_DIR) && DATA_DIR=$(HPC_DATA_DIR) bash hpc/prefetch.sh $(DATASETS)"

# Submit a train+eval job (MODEL required; each script trains then evaluates)
# Usage: make hpc-train MODEL=tgn
#        make hpc-train MODEL=tgn DATASET=tgbl-review SEED=1
hpc-train:
	ssh $(HPC_HOST) "cd $(HPC_DIR) && DATA_DIR=$(HPC_DATA_DIR) DATASET=$(DATASET) SEED=$(SEED) sbatch hpc/slurm/train_$(MODEL).sh"

# Check status of your HPC jobs
hpc-status:
	ssh $(HPC_HOST) "squeue -u \$$USER"

# Fetch results and checkpoints back to local machine
hpc-fetch:
	rsync -avz $(HPC_HOST):$(HPC_DATA_DIR)/results/     results/
	rsync -avz $(HPC_HOST):$(HPC_DATA_DIR)/checkpoints/ data/checkpoints/
