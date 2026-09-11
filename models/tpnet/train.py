"""
Train TPNet (SOTA on tgbl-wiki leaderboard) on a TGB link prediction dataset.

Usage:
    python models/tpnet/train.py --dataset tgbl-wiki
    python models/tpnet/train.py --dataset tgbl-wiki --epochs 50 --patience 5

Checkpoint saved to:
    models/tpnet/checkpoints/{dataset}/run0.pkl

Run from the DynGraphEval root directory.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: str) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, shell=True, check=True, env=env)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",            default="tgbl-wiki")
    p.add_argument("--epochs",             type=int,   default=50)
    p.add_argument("--patience",           type=int,   default=5)
    p.add_argument("--batch_size",         type=int,   default=200)
    p.add_argument("--num_layers",         type=int,   default=2)
    p.add_argument("--output_dim",         type=int,   default=100)
    p.add_argument("--time_feat_dim",      type=int,   default=100)
    p.add_argument("--num_neighbors",      type=int,   default=20)
    p.add_argument("--dropout",            type=float, default=0.1)
    p.add_argument("--lr",                 type=float, default=0.0001)
    p.add_argument("--rp_dim_factor",      type=int,   default=10)
    p.add_argument("--rp_num_layer",       type=int,   default=2)
    p.add_argument("--rp_time_decay",      type=float, default=1e-6)
    p.add_argument("--gpu",                type=int,   default=None)
    p.add_argument("--seed",               type=int,   default=0)
    p.add_argument("--repo_dir",           default="/tmp/TGB_TPNet")
    p.add_argument("--datasets_cache",     default=None)
    p.add_argument("--checkpoints_dir",    default=None)
    return p.parse_args()


def _apply_patches(repo_dir: str) -> None:
    patches_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")
    patch_map = {
        "train_link_prediction.py": "train_link_prediction.py",
        "evaluate_models_utils.py": os.path.join("utils", "evaluate_models_utils.py"),
    }
    for src_name, dst_rel in patch_map.items():
        src = os.path.join(patches_dir, src_name)
        dst = os.path.join(repo_dir, dst_rel)
        shutil.copy(src, dst)
        print(f"  [patch] {dst_rel}")


def main():
    args = parse_args()

    import torch
    if args.gpu is None:
        args.gpu = 0 if torch.cuda.is_available() else -1
    if args.gpu < 0 or not torch.cuda.is_available():
        raise SystemExit("ERROR: No CUDA GPU found. TPNet training requires a GPU.")
    _log(f"GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    # ── Checkpoint destination ────────────────────────────────────────────────
    if args.checkpoints_dir:
        ckpt_dir = os.path.join(args.checkpoints_dir, args.dataset)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir   = os.path.dirname(os.path.dirname(script_dir))
        ckpt_dir   = os.path.join(root_dir, "models", "tpnet", "checkpoints", args.dataset)
    os.makedirs(ckpt_dir, exist_ok=True)
    _log(f"Checkpoint dir: {ckpt_dir}")

    # ── Clone / update TPNet repo ─────────────────────────────────────────────
    repo_url = "https://github.com/lxd99/TGB_TPNet.git"
    if os.path.exists(args.repo_dir):
        _log(f"Updating TPNet repo at {args.repo_dir}...")
        run(f"git -C {args.repo_dir} pull --quiet")
    else:
        _log(f"Cloning TPNet repo to {args.repo_dir}...")
        run(f"git clone --quiet {repo_url} {args.repo_dir}")
    _log("Repo ready.")

    for d in ["logs", "saved_models", "saved_results"]:
        os.makedirs(os.path.join(args.repo_dir, d), exist_ok=True)

    if args.datasets_cache:
        repo_datasets = os.path.join(args.repo_dir, "datasets")
        if not os.path.exists(repo_datasets):
            os.makedirs(args.datasets_cache, exist_ok=True)
            os.symlink(args.datasets_cache, repo_datasets)
            _log(f"Dataset cache symlinked: {repo_datasets} → {args.datasets_cache}")

        drive_root   = os.path.dirname(os.path.realpath(args.datasets_cache))
        drive_models = os.path.join(drive_root, "tpnet_saved_models")
        repo_models  = os.path.join(args.repo_dir, "saved_models")
        if not os.path.islink(repo_models):
            shutil.rmtree(repo_models, ignore_errors=True)
            os.makedirs(drive_models, exist_ok=True)
            os.symlink(drive_models, repo_models)
            _log(f"saved_models symlinked: {repo_models} → {drive_models}")

    # ── Apply patches ─────────────────────────────────────────────────────────
    _log("Applying patches...")
    _apply_patches(args.repo_dir)
    _log("Patches applied.")

    # ── Run training ──────────────────────────────────────────────────────────
    prefix = f"run{args.seed}"
    _log(f"Starting training: TPNet on {args.dataset} | "
         f"epochs={args.epochs} patience={args.patience} batch={args.batch_size}")

    train_cmd = (
        f"cd {args.repo_dir} && "
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"{sys.executable} train_link_prediction.py "
        f"  --model_name TPNet "
        f"  --dataset_name {args.dataset} "
        f"  --num_runs 1 "
        f"  --num_epochs {args.epochs} "
        f"  --patience {args.patience} "
        f"  --batch_size {args.batch_size} "
        f"  --num_layers {args.num_layers} "
        f"  --output_dim {args.output_dim} "
        f"  --time_feat_dim {args.time_feat_dim} "
        f"  --num_neighbors {args.num_neighbors} "
        f"  --dropout {args.dropout} "
        f"  --learning_rate {args.lr} "
        f"  --sample_neighbor_strategy recent "
        f"  --use_random_projection "
        f"  --rp_dim_factor {args.rp_dim_factor} "
        f"  --rp_num_layer {args.rp_num_layer} "
        f"  --rp_time_decay_weight {args.rp_time_decay} "
        f"  --gpu {args.gpu} "
        f"  --prefix {prefix}"
    )
    run(train_cmd)
    _log("Training complete.")

    # ── Copy checkpoint to our models/ structure ──────────────────────────────
    src = os.path.join(
        args.repo_dir, "saved_models",
        f"{prefix}_link_{args.dataset}_TPNet_seed{args.seed}.pkl"
    )
    dst = os.path.join(ckpt_dir, f"run{args.seed}.pkl")

    if not os.path.exists(src):
        saved = os.listdir(os.path.join(args.repo_dir, "saved_models"))
        _log(f"ERROR: Expected checkpoint not found at {src}")
        _log(f"Files in saved_models: {saved}")
        sys.exit(1)

    _log(f"Copying checkpoint → {dst}")
    shutil.copy(src, dst)
    _log(f"Done. Checkpoint saved to: {dst}")


if __name__ == "__main__":
    main()
