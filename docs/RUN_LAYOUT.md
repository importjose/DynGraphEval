# HPC time-encoder experiments: run layout

New time-encoder runs use one directory per dataset, variant, seed, and Slurm job. Names `*_bin_*` denote elapsed-time **binning**; older `*_time_*` names denote continuous input scaling and should not be interpreted as binned results.

```text
/mnt/beegfs/home/delgadoj2020/dyngrapheval-data/
├── datasets/                       # Shared TGB cache
├── neg_cache/                      # Shared evaluation negatives
├── checkpoints/                    # Legacy models; not moved or deleted
├── results/                        # Legacy flat JSON files; not moved or deleted
└── runs/
    └── tgbl-wiki/
        └── tgn_bin_day/
            └── seed0/
                └── 4849999/       # Slurm job ID (or unique manual run ID)
                    ├── manifest.json       # Experiment identity and train arguments
                    ├── checkpoint.pkl      # Checkpoint evaluated by this run
                    ├── metrics.json         # Evaluation scores + checkpoint SHA-256
                    ├── run.log              # Training and evaluation log
                    └── trainer/             # Vendored trainer's best + resume checkpoints
```

The folder name (`tgn_bin_day`) identifies the model and age binning; the filename (`checkpoint.pkl`) identifies the artifact type. Nothing in `trainer/` is shared with another run. `run.log` is the readable per-variant log. Slurm also writes a scheduler-level `.out` file in the repo's `outputs/` directory because its `#SBATCH --output` path is resolved before the script starts; it is not a second checkpoint or evaluation result. The patch that creates the trainer's internal filenames still uses `run{seed}` **inside this isolated folder**, since that is what the vendored trainer expects.

## Run and fetch

Deploy local code with `make hpc-push` (or commit/push locally and pull on Athene), then submit from the remote repo (or use `make hpc-train MODEL=tgn_time_scale GRANULARITY=day`):

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
for gran in hour day week month; do
  sbatch --export=ALL,GRANULARITY="$gran" hpc/slurm/train_tgat_time_scale.sh
done
```

Each script requests 5h55m and passes the exact `runs/.../checkpoint.pkl` to `hpc/eval.py --checkpoint`; evaluation writes `metrics.json` with `--output`. For `*_bin_day`, the transformed encoder input is `floor((query_time - event_time) / 86400)` when the source timestamps are in seconds. This buckets **elapsed-time ages**, not absolute wall-clock dates, and leaves historical-neighbor selection and memory event order unchanged. The transform is applied to both TGN's message and attention time paths, and consistently during training, memory warmup, and scoring. The bin width is stored in `manifest.json` and `metrics.json`. If absolute calendar-day bins were intended instead, that is a different experiment.

The older TGN `*_time_*` run results are **continuous scaling only**. Older TGAT/GraphMixer `*_time_*` results are invalid as scale ablations because their training code did not apply the scaling patch. They are never migrated into `*_bin_*` directories. If one run exceeds the six-hour limit, reduce workload or request an appropriate GPU partition; do not silently resume with a different scale.

```bash
make hpc-fetch       # Only metrics.json + manifest.json for new runs; also fetches legacy results/
make hpc-fetch-logs  # Fetch per-run logs when needed
make hpc-fetch-checkpoint DATASET=tgbl-wiki MODEL=tgn_bin_day SEED=0 RUN_ID=4849999
```

Use `python hpc/eval.py --model tgn_bin_day --dataset tgbl-wiki --seed 0 --checkpoint /absolute/path/to/checkpoint.pkl --output /absolute/path/to/metrics.json` for manual evaluation of a new run. Do not omit `--checkpoint`: default resolution still targets the **legacy** `checkpoints/{model}/{dataset}/run{seed}.pkl`. Do not mix old flat results with clean runs for causal claims. The old files remain untouched; there is no automatic migration because some older experiments have ambiguous checkpoint provenance.

## Bin-occupancy diagnostic

The CPU-only `scripts/analyze_time_bins.py` inspects the TGB dataset without loading checkpoints. It groups same-timestamp events so histories include only interactions **strictly earlier** than each query, then records the valid first-hop ages for the last 20 historical neighbors of each positive edge's source and destination. Padding is excluded. It reports bin-0, bin-1, bins 2–9, and bins 10+ for each width, separately for train, validation, and test. The TGN message-age summary is explicitly a **proxy** based on the last endpoint interaction: real TGN memory updates use deferred messages and can have different last-update times. Recursive neighbor expansions and negative-candidate histories are not counted. The JSON marks the timestamp unit unverified.

After deploying the script on Athene, use `sbatch hpc/slurm/analyze_time_bins.sh`, then `make hpc-fetch-bin-occupancy` locally to get `bin_occupancy/tgbl-wiki.json`. This job requests only CPUs (not an A100).

For interrupted jobs, the Slurm job ID selects the directory. A fresh `sbatch` receives a new ID and starts an isolated run. Resuming the *same* directory intentionally requires passing its run ID and existing run directory, and the resume checkpoint's experiment identity (variant, model, run ID, and scale) must match. A fresh job never resumes a different job's trainer files.
