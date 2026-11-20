# HPC time-scale run layout

New time-scale runs use one directory per dataset, variant, seed, and Slurm job:

```text
/mnt/beegfs/home/delgadoj2020/dyngrapheval-data/
├── datasets/                       # Shared TGB cache
├── neg_cache/                      # Shared evaluation negatives
├── checkpoints/                    # Legacy models; not moved or deleted
├── results/                        # Legacy flat JSON files; not moved or deleted
└── runs/
    └── tgbl-wiki/
        └── tgn_time_day/
            └── seed0/
                └── 4849999/       # Slurm job ID (or unique manual run ID)
                    ├── manifest.json       # Experiment identity and train arguments
                    ├── checkpoint.pkl      # Checkpoint evaluated by this run
                    ├── metrics.json         # Evaluation scores + checkpoint SHA-256
                    ├── run.log              # Training and evaluation log
                    └── trainer/             # Vendored trainer's best + resume checkpoints
```

The folder name (`tgn_time_day`) identifies the model and input scaling; the filename (`checkpoint.pkl`) identifies the artifact type. Nothing in `trainer/` is shared with another run. `run.log` is the readable per-variant log. Slurm also writes a scheduler-level `.out` file in the repo's `outputs/` directory because its `#SBATCH --output` path is resolved before the script starts; it is not a second checkpoint or evaluation result. The patch that creates the trainer's internal filenames still uses `run{seed}` **inside this isolated folder**, since that is what the vendored trainer expects.

## Run and fetch

Deploy local code with `make hpc-push`, then submit from the remote repo (or via `make hpc-train MODEL=tgn_time_scale`):

```bash
ssh delgadoj2020@athene-login.hpc.fau.edu \
  'cd /mnt/beegfs/home/delgadoj2020/projects/DynGraphEval && sbatch --export=ALL,GRANULARITY=day hpc/slurm/train_tgn_time_scale.sh'
```

The GPU nodes are only on `shortq7` (six-hour limit); `longq7` has no GPUs. Run **one granularity per job** using `GRANULARITY`, for example:

```bash
for gran in hour day week month; do
  sbatch --export=ALL,GRANULARITY="$gran" hpc/slurm/train_tgn_time_scale.sh
  sbatch --export=ALL,GRANULARITY="$gran" hpc/slurm/train_graphmixer_time_scale.sh
done
for gran in day week; do
  sbatch --export=ALL,GRANULARITY="$gran" hpc/slurm/train_tgat_time_scale.sh
done
```

Each script requests 5h55m and passes the exact `runs/.../checkpoint.pkl` to `hpc/eval.py --checkpoint`; evaluation writes `metrics.json` with `--output`. If one run exceeds the six-hour limit, reduce workload or request an appropriate GPU partition; do not silently resume with a different scale.

```bash
make hpc-fetch       # Only metrics.json + manifest.json for new runs; also fetches legacy results/
make hpc-fetch-logs  # Fetch per-run logs when needed
make hpc-fetch-checkpoint DATASET=tgbl-wiki MODEL=tgn_time_day SEED=0 RUN_ID=4849999
```

Use `python hpc/eval.py --model tgn_time_day --dataset tgbl-wiki --seed 0 --checkpoint /absolute/path/to/checkpoint.pkl --output /absolute/path/to/metrics.json` for manual evaluation of a new run. Do not omit `--checkpoint`: default resolution still targets the **legacy** `checkpoints/{model}/{dataset}/run{seed}.pkl`. Do not mix old flat results with clean runs for causal claims. The old files remain untouched; there is no automatic migration because some older experiments have ambiguous checkpoint provenance.

For interrupted jobs, the Slurm job ID selects the directory. A fresh `sbatch` receives a new ID and starts an isolated run. Resuming the *same* directory intentionally requires passing its run ID and existing run directory, and the resume checkpoint's experiment identity (variant, model, run ID, and scale) must match. A fresh job never resumes a different job's trainer files.
