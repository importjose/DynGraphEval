MLFLOW_DB := sqlite:///$(CURDIR)/mlflow.db

mlflow:
	MLFLOW_TRACKING_URI=$(MLFLOW_DB) mlflow ui

# ── HPC (FAU Athene) ──────────────────────────────────────────────────────────
# Set HPC_HOST to your Athene login node, e.g.:
#   export HPC_HOST=woody.rrze.uni-erlangen.de
HPC_HOST ?= woody.rrze.uni-erlangen.de
HPC_DIR  ?= ~/DynGraphEval

# Copy repo to HPC (rsync, excludes large build artifacts)
hpc-push:
	rsync -avz --exclude='.venv' --exclude='__pycache__' \
	    --exclude='*.pyc' --exclude='.git' --exclude='mlruns' \
	    --exclude='data/' \
	    . $(HPC_HOST):$(HPC_DIR)/

# One-time environment setup on HPC login node
hpc-setup:
	ssh $(HPC_HOST) "cd $(HPC_DIR) && bash hpc/setup.sh"

# Download datasets on HPC login node (must run before submitting jobs)
hpc-prefetch:
	ssh $(HPC_HOST) "cd $(HPC_DIR) && bash hpc/prefetch.sh $(DATASETS)"

# Submit a training job (MODEL required, e.g.: make hpc-train MODEL=tgn)
hpc-train:
	ssh $(HPC_HOST) "cd $(HPC_DIR) && sbatch hpc/slurm/train_$(MODEL).sh"

# Check status of HPC jobs
hpc-status:
	ssh $(HPC_HOST) "squeue -u \$$USER"

# Fetch results and checkpoints back from HPC
hpc-fetch:
	rsync -avz $(HPC_HOST):$(HPC_DIR)/results/ results/
	rsync -avz $(HPC_HOST):$(HPC_DIR)/data/checkpoints/ data/checkpoints/
