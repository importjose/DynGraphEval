"""
HPC evaluation script for DynGraphEval (no Modal dependency).

Runs Standard MRR + Recency MRR for a trained checkpoint directly on the
HPC node. Results are saved to results/{model}_{dataset}_{timestamp}.json.

Usage:
    python hpc/eval.py --model tgn --dataset tgbl-wiki --seed 0
    python hpc/eval.py --model edgebank --dataset tgbl-wiki
    python hpc/eval.py --model tgn --checkpoint data/checkpoints/tgn/tgbl-wiki/run0.pkl

Run from the DynGraphEval repo root.
"""

import argparse
import json
import os
import sys
import datetime
import time

# Ensure repo root is on sys.path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="DynGraphEval — HPC evaluation")
    p.add_argument("--model",          default="tgn",
                   choices=["tgn", "tgn_fixed_enc", "tgn_no_enc", "tgat",
                            "tgat_fixed_enc", "graphmixer", "tpnet", "edgebank"],
                   help="Model to evaluate")
    p.add_argument("--checkpoint",     default=None,
                   help="Explicit path to .pkl checkpoint. "
                        "Defaults to data/checkpoints/{model}/{dataset}/run{seed}.pkl")
    p.add_argument("--dataset",        default="tgbl-wiki")
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--num_neg",        type=int, default=999,
                   help="Negatives per edge for both metrics")
    p.add_argument("--skip_standard",  action="store_true",
                   help="Skip Standard MRR (reuse --standard_mrr value)")
    p.add_argument("--standard_mrr",   type=float, default=None,
                   help="Pre-computed Standard MRR (used with --skip_standard)")
    p.add_argument("--datasets_dir",    default=None,
                   help="TGB datasets directory. Default: data/datasets relative to repo root")
    p.add_argument("--checkpoints_dir", default=None,
                   help="Root checkpoints directory. Checkpoint resolved as "
                        "{checkpoints_dir}/{model}/{dataset}/run{seed}.pkl. "
                        "Default: data/checkpoints relative to repo root")
    p.add_argument("--results_dir",    default=None,
                   help="Output directory for JSON results. Default: results/ in repo root")
    return p.parse_args()


def main():
    args = parse_args()

    datasets_dir    = args.datasets_dir    or os.path.join(REPO_DIR, "data", "datasets")
    checkpoints_dir = args.checkpoints_dir or os.path.join(REPO_DIR, "data", "checkpoints")
    results_dir     = args.results_dir     or os.path.join(REPO_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── Resolve checkpoint ─────────────────────────────────────────────────────
    checkpoint = args.checkpoint
    if checkpoint is None and args.model != "edgebank":
        checkpoint = os.path.join(
            checkpoints_dir, args.model, args.dataset, f"run{args.seed}.pkl",
        )
        _log(f"Checkpoint: {checkpoint}")
        if not os.path.exists(checkpoint):
            _log(f"ERROR: checkpoint not found at {checkpoint}")
            _log(f"  Expected: {checkpoint}")
            _log(f"  Available in {os.path.dirname(checkpoint)}: "
                 f"{os.listdir(os.path.dirname(checkpoint)) if os.path.isdir(os.path.dirname(checkpoint)) else 'directory does not exist'}")
            sys.exit(1)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    # ── Load dataset ───────────────────────────────────────────────────────────
    _log(f"Loading dataset {args.dataset}...")
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    dataset_obj = PyGLinkPropPredDataset(name=args.dataset, root=datasets_dir)
    data       = dataset_obj.get_TemporalData()
    train_data = data[dataset_obj.train_mask]
    val_data   = data[dataset_obj.val_mask]
    test_data  = data[dataset_obj.test_mask]
    num_nodes  = int(max(data.src.max(), data.dst.max())) + 1
    min_dst    = int(data.dst.min())
    max_dst    = int(data.dst.max())
    _log(f"Dataset loaded: {len(data)} edges, {num_nodes} nodes")

    # ── Instantiate model ──────────────────────────────────────────────────────
    from models.tgn.model        import TPNetTGN
    from models.graphmixer.model import GraphMixerModel
    from models.tgat.model       import TGATModel
    from models.tpnet.model      import TPNetModel
    from models.edgebank.model   import EdgeBankModel

    _log(f"Loading model {args.model}...")
    if args.model in ("tgn", "tgn_fixed_enc", "tgn_no_enc"):
        m = TPNetTGN(checkpoint=checkpoint, num_nodes=num_nodes, device=device)
    elif args.model in ("tgat", "tgat_fixed_enc"):
        m = TGATModel(checkpoint=checkpoint, num_nodes=num_nodes, device=device)
    elif args.model == "graphmixer":
        m = GraphMixerModel(checkpoint=checkpoint, num_nodes=num_nodes, device=device)
    elif args.model == "tpnet":
        m = TPNetModel(checkpoint=checkpoint, num_nodes=num_nodes, device=device)
    elif args.model == "edgebank":
        m = EdgeBankModel(num_nodes=num_nodes, device=device)
    else:
        _log(f"Unknown model: {args.model}")
        sys.exit(1)

    # ── Run evaluation ─────────────────────────────────────────────────────────
    from evaluate.evaluator import Evaluator
    ev = Evaluator(
        model=m,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        num_nodes=num_nodes,
        min_dst=min_dst,
        max_dst=max_dst,
        num_neg=args.num_neg,
        neg_seed=args.seed,
        dataset_obj=dataset_obj,
    )

    results = ev.run(
        skip_standard=args.skip_standard,
        standard_mrr=args.standard_mrr,
    )
    results["model"]   = args.model
    results["dataset"] = args.dataset
    results["seed"]    = args.seed

    # ── Save results ───────────────────────────────────────────────────────────
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{args.model}_{args.dataset}_{ts}.json"
    out   = os.path.join(results_dir, fname)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    _log(f"Results saved to: {out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
