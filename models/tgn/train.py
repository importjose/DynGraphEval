"""
Train TGN (TPNet implementation) on a TGB link prediction dataset.

Usage (Colab or local):
    python models/tgn/train.py --dataset tgbl-wiki
    python models/tgn/train.py --dataset tgbl-wiki --epochs 50 --patience 5

Checkpoint saved to:
    models/tgn/checkpoints/{dataset}/run0.pkl

Run from the DynGraphEval root directory.

Dependencies:
    pip install torch-geometric py-tgb numba wandb
"""

import argparse
import os
import shutil
import subprocess
import sys


def run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


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
    p.add_argument("--device",         default=None,
                   help="cuda / cpu (default: auto-detect)")
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--repo_dir",       default="/tmp/TGB_TPNet",
                   help="Where to clone the TPNet repo")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Auto-detect device ────────────────────────────────────────────────────
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    # ── Checkpoint destination (relative to DynGraphEval root) ───────────────
    # train.py lives at models/tgn/train.py; root is two levels up
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    root_dir     = os.path.dirname(os.path.dirname(script_dir))
    ckpt_dir     = os.path.join(root_dir, "models", "tgn", "checkpoints", args.dataset)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Clone / update TPNet repo ─────────────────────────────────────────────
    repo_url = "https://github.com/lxd99/TGB_TPNet.git"
    if os.path.exists(args.repo_dir):
        run(f"git -C {args.repo_dir} pull --quiet")
    else:
        run(f"git clone --quiet {repo_url} {args.repo_dir}")

    # ── Create required directories inside the repo ───────────────────────────
    for d in ["logs", "saved_models", "saved_results"]:
        os.makedirs(os.path.join(args.repo_dir, d), exist_ok=True)

    # ── Run their training script ─────────────────────────────────────────────
    prefix = f"run{args.seed}"
    train_cmd = (
        f"cd {args.repo_dir} && {sys.executable} train_link_prediction.py "
        f"  --model_name TGN "
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
        f"  --device {args.device} "
        f"  --prefix {prefix}"
    )
    print(f"\nStarting training on {args.dataset}...\n")
    run(train_cmd)

    # ── Copy checkpoint to our models/ structure ──────────────────────────────
    src = os.path.join(
        args.repo_dir, "saved_models",
        f"{prefix}_link_{args.dataset}_TGN_seed{args.seed}.pkl"
    )
    dst = os.path.join(ckpt_dir, f"run{args.seed}.pkl")

    if not os.path.exists(src):
        # Fallback: list what was saved
        saved = os.listdir(os.path.join(args.repo_dir, "saved_models"))
        print(f"Expected checkpoint not found at {src}")
        print(f"Files in saved_models: {saved}")
        sys.exit(1)

    shutil.copy(src, dst)
    print(f"\nCheckpoint saved to: {dst}")
    print(f"\nTo evaluate, set config.yaml:")
    print(f"  model: tgn")
    print(f"  checkpoints:")
    print(f"    - models/tgn/checkpoints/{args.dataset}/run{args.seed}.pkl")


if __name__ == "__main__":
    main()
