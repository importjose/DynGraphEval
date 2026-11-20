"""
Temporal granularity ablation for TGAT: scale TimeEncoder input timestamps.

This variant trains TGAT (TPNet implementation) with timestamps pre-scaled
by a fixed factor before they enter the TimeEncoder's cosine encoding.  Scaling
controls the temporal resolution the model's learned frequencies operate at:

  - day  (1/86400):  encoder unit ≈ 1 day
  - week (1/604800): encoder unit ≈ 1 week

Only day and week granularities are included for TGAT as a focused control
experiment.  The underlying model weights and architecture are identical to
plain TGAT; only the granularity of the temporal signal fed to the TimeEncoder
changes.

Note: tgat_components.py imports TimeEncoder from tpnet_components.py, so
patching models.tgn.tpnet_components.TimeEncoder in the env also patches TGAT.

Usage:
    python models/tgat_time_scale/train.py --granularity day --dataset tgbl-wiki
    python models/tgat_time_scale/train.py --granularity week --dataset tgbl-wiki --epochs 50

Artifacts saved to:
    run_dir/{checkpoint.pkl, trainer/} where run_dir defaults to
    data/runs/{dataset}/tgat_time_{granularity}/seed{seed}/{run_id}/

Run from the DynGraphEval root directory.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import uuid
import re
import json


SCALE_MAP = {
    "day":  1.0 / 86400,
    "week": 1.0 / 604800,
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: str, extra_env: dict = None) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, shell=True, check=True, env=env)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--granularity",    choices=["day", "week"], required=True,
                   help="Temporal granularity: scale factor applied to timestamps before TimeEncoder")
    p.add_argument("--run_id",         default=None,
                   help="Unique job identifier; defaults to SLURM_JOB_ID or a random ID")
    p.add_argument("--dataset",        default="tgbl-wiki")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--patience",       type=int,   default=5)
    p.add_argument("--batch_size",     type=int,   default=200)
    p.add_argument("--num_layers",     type=int,   default=2)
    p.add_argument("--num_heads",      type=int,   default=2)
    p.add_argument("--output_dim",     type=int,   default=100)
    p.add_argument("--time_feat_dim",  type=int,   default=100)
    p.add_argument("--num_neighbors",  type=int,   default=20)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--lr",             type=float, default=0.0001)
    p.add_argument("--gpu",            type=int,   default=None,
                   help="GPU index to use (default: 0 if CUDA available, else CPU)")
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--repo_dir",       default=None,
                   help="Path to vendored TGB_TPNet repo (default: src/TGB_TPNet)")
    p.add_argument("--datasets_cache", default=None,
                   help="Persistent directory for TGB datasets (symlinked into repo).")
    p.add_argument("--run_dir", default=None,
                   help="Directory for this run's checkpoint and trainer state. "
                        "Defaults to <data_root>/runs/{dataset}/{variant}/seed{seed}/{run_id}.")
    return p.parse_args()


def _apply_patches(repo_dir: str) -> None:
    patches_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")
    patch_map = {
        "train_link_prediction.py": "train_link_prediction.py",
        "evaluate_models_utils.py": os.path.join("utils", "evaluate_models_utils.py"),
        "EarlyStopping.py": os.path.join("utils", "EarlyStopping.py"),
    }
    for src_name, dst_rel in patch_map.items():
        src = os.path.join(patches_dir, src_name)
        dst = os.path.join(repo_dir, dst_rel)
        shutil.copy(src, dst)
        print(f"  [patch] {dst_rel}")


def main():
    args = parse_args()

    model_name = f"tgat_time_{args.granularity}"
    scale      = SCALE_MAP[args.granularity]
    run_id = args.run_id or os.environ.get("SLURM_JOB_ID") or uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, underscores, and hyphens")

    # ── Auto-detect GPU ───────────────────────────────────────────────────────
    import torch
    if args.gpu is None:
        args.gpu = 0 if torch.cuda.is_available() else -1
    if args.gpu < 0 or not torch.cuda.is_available():
        raise SystemExit("ERROR: No CUDA GPU found. TGAT training requires a GPU.")
    _log(f"GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    # All artifacts for one experiment live under one unambiguous run directory.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_root = (os.path.dirname(os.path.realpath(args.datasets_cache))
                 if args.datasets_cache else os.path.join(project_root, "data"))
    run_dir = os.path.abspath(args.run_dir or os.path.join(
        data_root, "runs", args.dataset, model_name, f"seed{args.seed}", run_id))
    os.makedirs(run_dir, exist_ok=True)
    _log(f"Run directory: {run_dir}")
    manifest = {"dataset": args.dataset, "variant": model_name, "seed": args.seed,
                "run_id": run_id, "time_scale": scale, "training_args": vars(args)}
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if args.repo_dir is None:
        root_dir      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.repo_dir = os.path.join(root_dir, "src", "TGB_TPNet")
    _log(f"TPNet repo dir: {args.repo_dir}")

    for d in ["logs", "saved_models", "saved_results"]:
        os.makedirs(os.path.join(args.repo_dir, d), exist_ok=True)

    if args.datasets_cache:
        repo_datasets = os.path.join(args.repo_dir, "datasets")
        if not os.path.exists(repo_datasets):
            os.makedirs(args.datasets_cache, exist_ok=True)
            os.symlink(args.datasets_cache, repo_datasets)
            _log(f"Dataset cache symlinked: {repo_datasets} → {args.datasets_cache}")

    # The vendored trainer needs ./saved_models; point it at this run only.
    drive_models = os.path.join(run_dir, "trainer")
    repo_models = os.path.join(args.repo_dir, "saved_models")
    os.makedirs(drive_models, exist_ok=True)
    if os.path.islink(repo_models):
        os.unlink(repo_models)
    elif os.path.isdir(repo_models):
        if os.listdir(repo_models):
            raise RuntimeError(f"Refusing to replace nonempty saved_models: {repo_models}")
        os.rmdir(repo_models)
    os.symlink(drive_models, repo_models)
    _log(f"Isolated saved_models: {repo_models} → {drive_models}")

    # ── Apply patches ─────────────────────────────────────────────────────────
    _log("Applying patches...")
    _apply_patches(args.repo_dir)
    _log("Patches applied.")

    # ── Run training ──────────────────────────────────────────────────────────
    prefix = f"run{args.seed}"
    _log(f"Starting training: {model_name} on {args.dataset} | "
         f"scale={scale:.2e} ({args.granularity}) | "
         f"epochs={args.epochs} patience={args.patience} batch={args.batch_size}")

    train_cmd = (
        f"cd {args.repo_dir} && "
        f"PYTHONPATH={args.repo_dir} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"{sys.executable} _launch.py "
        f"  --model_name TGAT "
        f"  --dataset_name {args.dataset} "
        f"  --num_runs 1 "
        f"  --num_epochs {args.epochs} "
        f"  --patience {args.patience} "
        f"  --batch_size {args.batch_size} "
        f"  --num_layers {args.num_layers} "
        f"  --num_heads {args.num_heads} "
        f"  --output_dim {args.output_dim} "
        f"  --time_feat_dim {args.time_feat_dim} "
        f"  --num_neighbors {args.num_neighbors} "
        f"  --dropout {args.dropout} "
        f"  --learning_rate {args.lr} "
        f"  --sample_neighbor_strategy recent "
        f"  --gpu {args.gpu} "
        f"  --prefix {prefix}"
    )
    # TIME_SCALE signals the patch to scale TimeEncoder input timestamps
    run(train_cmd, extra_env={"TIME_SCALE": str(scale),
                              "TIME_ENCODER_VARIANT": model_name,
                              "TIME_ENCODER_RUN_ID": run_id})
    _log("Training complete.")

    # ── Copy checkpoint to our models/ structure ──────────────────────────────
    src = os.path.join(
        args.repo_dir, "saved_models",
        f"{prefix}_link_{args.dataset}_TGAT_seed{args.seed}.pkl"
    )
    dst = os.path.join(run_dir, "checkpoint.pkl")

    if not os.path.exists(src):
        saved = os.listdir(os.path.join(args.repo_dir, "saved_models"))
        _log(f"ERROR: Expected checkpoint not found at {src}")
        _log(f"Files in saved_models: {saved}")
        sys.exit(1)

    _log(f"Copying checkpoint → {dst}")
    shutil.copy(src, dst)
    _log(f"Done. Checkpoint saved to: {dst}")
    print(f"\nTo evaluate:")
    print(f"  python hpc/eval.py --model {model_name} "
          f"--dataset {args.dataset} --seed {args.seed} --checkpoint '{dst}'")


if __name__ == "__main__":
    main()
