MLFLOW_DB := sqlite:///$(CURDIR)/mlflow.db

mlflow:
	MLFLOW_TRACKING_URI=$(MLFLOW_DB) mlflow ui

# ── HPC (FAU Athene) ──────────────────────────────────────────────────────────
# Override HPC_HOST/HPC_DIR/HPC_DATA_DIR if your Athene account uses other paths.
#
# DATA_DIR controls where datasets, checkpoints, and results are stored
# on the HPC. The time-scale experiments use DATA_DIR/runs/<dataset>/<variant>/seed<N>/<job-id>/.
# Override if your home quota is small:
#   make hpc-train MODEL=tgn HPC_DATA_DIR=/work/$$USER/dyngrapheval
#
# Storage layout on HPC (under DATA_DIR):
#   datasets/                  TGB downloads, shared across models (~5GB)
#   checkpoints/{model}/{ds}/  run{seed}.pkl per trained model
#   results/                   eval JSON outputs
HPC_HOST     ?= delgadoj2020@athene-login.hpc.fau.edu
HPC_DIR      ?= /mnt/beegfs/home/delgadoj2020/projects/DynGraphEval
HPC_DATA_DIR ?= /mnt/beegfs/home/delgadoj2020/dyngrapheval-data

# Copy repo to HPC (excludes large/generated artifacts)
hpc-push:
	rsync -avz --exclude='.venv' --exclude='__pycache__' \
	    --exclude='*.pyc' --exclude='.git' --exclude='mlruns' \
	    --exclude='data/' --exclude='results/' --exclude='outputs/' --exclude='runs/' \
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
	@test -n "$(MODEL)" || (echo "ERROR: MODEL is required. Usage: make hpc-train MODEL=tgn"; exit 1)
	ssh $(HPC_HOST) "cd $(HPC_DIR) && \
	    DATA_DIR=$(HPC_DATA_DIR) \
	    $(if $(DATASET),DATASET=$(DATASET),) \
	    $(if $(SEED),SEED=$(SEED),) \
	    $(if $(GRANULARITY),GRANULARITY=$(GRANULARITY),) \
	    sbatch hpc/slurm/train_$(MODEL).sh"

# Check status of your HPC jobs
hpc-status:
	ssh $(HPC_HOST) "squeue -u \$$USER"

# Download only small run metadata and metrics; never pull huge checkpoints by default.
# Historical flat JSON files remain in results/ and are never silently migrated/deleted.
hpc-fetch:
	mkdir -p runs results
	rsync -avz --prune-empty-dirs --include='*/' --include='metrics.json' \
	    --include='manifest.json' --exclude='*' $(HPC_HOST):$(HPC_DATA_DIR)/runs/ runs/
	rsync -avz $(HPC_HOST):$(HPC_DATA_DIR)/results/ results/

hpc-fetch-bin-occupancy:
	mkdir -p bin_occupancy
	rsync -avz $(HPC_HOST):$(HPC_DATA_DIR)/bin_occupancy/ bin_occupancy/

hpc-fetch-logs:
	mkdir -p runs
	rsync -avz --prune-empty-dirs --include='*/' --include='run.log' \
	    --exclude='*' $(HPC_HOST):$(HPC_DATA_DIR)/runs/ runs/

# Fetch a single run's checkpoint when needed (DATASET, MODEL, SEED, RUN_ID required).
hpc-fetch-checkpoint:
	@test -n "$(DATASET)" -a -n "$(MODEL)" -a -n "$(SEED)" -a -n "$(RUN_ID)" || \
	    (echo 'Usage: make hpc-fetch-checkpoint DATASET=tgbl-wiki MODEL=tgn_time_day SEED=0 RUN_ID=<job-id>'; exit 1)
	mkdir -p runs/$(DATASET)/$(MODEL)/seed$(SEED)/$(RUN_ID)
	rsync -avz $(HPC_HOST):$(HPC_DATA_DIR)/runs/$(DATASET)/$(MODEL)/seed$(SEED)/$(RUN_ID)/checkpoint.pkl \
	    runs/$(DATASET)/$(MODEL)/seed$(SEED)/$(RUN_ID)/checkpoint.pkl
