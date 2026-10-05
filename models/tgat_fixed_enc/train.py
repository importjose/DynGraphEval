"""
Train TGAT with a fixed (non-trainable) time encoder on a TGB link prediction dataset.

This is an ablation of the standard TGAT (models/tgat/train.py) that freezes the
TimeEncoder weights during training. The architecture is identical to TGAT; only the
time encoder's parameters are excluded from gradient updates.

Hypothesis: removing the recency bias (learned by the time encoder) while keeping the
attention mechanism should reduce the Return/Explore gap, completing the 2×2 matrix.

Usage:
    modal run modal/train.py --model tgat_fixed_enc

Checkpoint saved to:
    /data/checkpoints/tgat_fixed_enc/{dataset}/run{seed}.pkl
"""

import argparse
import os
import shutil
import subprocess
import sys
import time


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
    p.add_argument("--gpu",            type=int,   default=None)
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--repo_dir",       default="/tmp/TGB_TPNet")
    p.add_argument("--datasets_cache", default=None)
    p.add_argument("--checkpoints_dir", default=None)
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
        raise SystemExit("ERROR: No CUDA GPU found. Training requires a GPU.")
    _log(f"GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    if args.checkpoints_dir:
        ckpt_dir = os.path.join(args.checkpoints_dir, args.dataset)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir   = os.path.dirname(os.path.dirname(script_dir))
        ckpt_dir   = os.path.join(root_dir, "models", "tgat_fixed_enc", "checkpoints", args.dataset)
    os.makedirs(ckpt_dir, exist_ok=True)
    _log(f"Checkpoint dir: {ckpt_dir}")

    repo_url = "https://github.com/lxd99/TGB_TPNet.git"
    if os.path.exists(args.repo_dir):
        _log(f"Updating TPNet repo at {args.repo_dir}...")
        run(f"git -C {args.repo_dir} pull --quiet")
    else:
        _log(f"Cloning TPNet repo to {args.repo_dir}...")
        run(f"git clone --quiet {repo_url} {args.repo_dir}")

    for d in ["logs", "saved_models", "saved_results"]:
        os.makedirs(os.path.join(args.repo_dir, d), exist_ok=True)

    if args.datasets_cache:
        repo_datasets = os.path.join(args.repo_dir, "datasets")
        if not os.path.exists(repo_datasets):
            os.makedirs(args.datasets_cache, exist_ok=True)
            os.symlink(args.datasets_cache, repo_datasets)
        drive_root   = os.path.dirname(os.path.realpath(args.datasets_cache))
        drive_models = os.path.join(drive_root, "tpnet_saved_models")
        repo_models  = os.path.join(args.repo_dir, "saved_models")
        if not os.path.islink(repo_models):
            shutil.rmtree(repo_models, ignore_errors=True)
            os.makedirs(drive_models, exist_ok=True)
            os.symlink(drive_models, repo_models)

    _log("Applying patches...")
    _apply_patches(args.repo_dir)
    _log("Patches applied.")

    prefix = f"run{args.seed}"
    _log(f"Starting training: TGAT+FixedTimeEncoder on {args.dataset} | "
         f"epochs={args.epochs} patience={args.patience} batch={args.batch_size}")

    train_cmd = (
        f"cd {args.repo_dir} && "
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"{sys.executable} train_link_prediction.py "
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
    # FREEZE_TIME_ENCODER=1 signals the patch to freeze TimeEncoder params after model init
    run(train_cmd, extra_env={"FREEZE_TIME_ENCODER": "1"})
    _log("Training complete.")

    src = os.path.join(
        args.repo_dir, "saved_models",
        f"{prefix}_link_{args.dataset}_TGAT_seed{args.seed}.pkl"
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
    print(f"\nTo evaluate:")
    print(f"  modal run modal/eval.py --model tgat --checkpoint /data/checkpoints/tgat_fixed_enc/{args.dataset}/run{args.seed}.pkl")


if __name__ == "__main__":
    main()
