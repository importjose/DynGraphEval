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
import hashlib

# Ensure repo root is on sys.path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="DynGraphEval — HPC evaluation")
    p.add_argument("--model",          default="tgn",
                   choices=["tgn", "tgn_fixed_enc", "tgn_no_enc",
                            "tgn_time_hour", "tgn_time_day", "tgn_time_week", "tgn_time_month",
                            "tgn_bin_hour", "tgn_bin_day", "tgn_bin_week", "tgn_bin_month",
                            "tgat", "tgat_fixed_enc", "tgat_no_enc",
                            "tgat_time_hour", "tgat_time_day", "tgat_time_week", "tgat_time_month",
                            "tgat_bin_hour", "tgat_bin_day", "tgat_bin_week", "tgat_bin_month",
                            "graphmixer", "graphmixer_multiscale",
                            "graphmixer_time_hour", "graphmixer_time_day",
                            "graphmixer_time_week", "graphmixer_time_month",
                            "graphmixer_bin_hour", "graphmixer_bin_day",
                            "graphmixer_bin_week", "graphmixer_bin_month",
                            "tpnet", "edgebank"],
                   help="Model to evaluate")
    p.add_argument("--checkpoint",     default=None,
                   help="Explicit path to .pkl checkpoint. "
                        "Defaults to {checkpoints_dir}/{model}/{dataset}/run{seed}.pkl")
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
    p.add_argument("--output",         default=None,
                   help="Exact output JSON path (e.g. runs/<dataset>/<variant>/seed0/<run-id>/metrics.json). "
                        "Overrides --results_dir; legacy flat-file output remains the default.")
    p.add_argument("--results_dir",    default=None,
                   help="Output directory for JSON results. Default: results/ in repo root")
    p.add_argument("--neg_cache_dir",  default=None,
                   help="Directory to cache recency negatives. Default: neg_cache/ in DATA_DIR")
    return p.parse_args()


def main():
    args = parse_args()

    datasets_dir    = args.datasets_dir    or os.path.join(REPO_DIR, "data", "datasets")
    checkpoints_dir = args.checkpoints_dir or os.path.join(REPO_DIR, "data", "checkpoints")
    results_dir     = args.results_dir     or os.path.join(REPO_DIR, "results")
    neg_cache_dir   = args.neg_cache_dir   or os.path.join(REPO_DIR, "neg_cache")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(neg_cache_dir, exist_ok=True)

    # ── Resolve checkpoint ─────────────────────────────────────────────────────
    checkpoint = args.checkpoint
    if checkpoint is None and args.model != "edgebank":
        checkpoint = os.path.join(
            checkpoints_dir, args.model, args.dataset, f"run{args.seed}.pkl",
        )
        _log(f"Checkpoint: {checkpoint}")
        if not os.path.exists(checkpoint):
            ckpt_dir = os.path.dirname(checkpoint)
            available = os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else "directory does not exist"
            _log(f"ERROR: checkpoint not found at {checkpoint}")
            _log(f"  Available in {ckpt_dir}: {available}")
            sys.exit(1)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    # Apply the same transformation during memory warmup and link scoring.
    _TIME_GRAN_WIDTHS = {"hour": 3600, "day": 86400,
                         "week": 604800, "month": 2592000}

    def _apply_time_encoding_patch(model_name: str) -> None:
        for gran, width in _TIME_GRAN_WIDTHS.items():
            mode = ("binned" if model_name.endswith(f"_bin_{gran}") else
                    "scaled" if model_name.endswith(f"_time_{gran}") else None)
            if mode is None:
                continue
            from models.tgn.tpnet_components import TimeEncoder
            _orig = TimeEncoder.forward
            def _forward(self, timestamps, _w=width, _mode=mode, _orig=_orig):
                ages = (torch.floor(timestamps / _w) if _mode == "binned"
                        else timestamps / _w)
                return _orig(self, ages)
            TimeEncoder.forward = _forward
            _log(f"[time-{mode}] TimeEncoder input: "
                 f"{'floor(age / width)' if mode == 'binned' else 'age / width'} "
                 f"(width={width} input units)")
            return

    _apply_time_encoding_patch(args.model)

    # ── Load dataset ───────────────────────────────────────────────────────────
    _log(f"Loading dataset {args.dataset}...")
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    dataset_obj = PyGLinkPropPredDataset(name=args.dataset, root=datasets_dir)
    data       = dataset_obj.get_TemporalData()
    train_data = data[dataset_obj.train_mask]
    val_data   = data[dataset_obj.val_mask]
    test_data  = data[dataset_obj.test_mask]
    num_nodes  = int(max(data.src.max(), data.dst.max())) + 1
    msg_dim    = data.msg.shape[1]
    _log(f"Dataset loaded: {len(data)} edges, {num_nodes} nodes, msg_dim={msg_dim}")

    # ── Instantiate model ──────────────────────────────────────────────────────
    from models.tgn.model                       import TPNetTGN
    from models.graphmixer.model                import GraphMixerModel
    from models.graphmixer_multiscale.model     import MultiScaleGraphMixerModel
    from models.tgat.model                      import TGATModel
    from models.tpnet.model                     import TPNetModel
    from models.edgebank.model                  import EdgeBankModel

    _log(f"Loading model {args.model}...")
    if (args.model in ("tgn", "tgn_fixed_enc", "tgn_no_enc") or
            args.model.startswith(("tgn_time_", "tgn_bin_"))):
        m = TPNetTGN(
            checkpoint_path=checkpoint,
            num_nodes=num_nodes,
            msg_dim=msg_dim,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
        )
        m.load_checkpoint()
    elif (args.model in ("tgat", "tgat_fixed_enc", "tgat_no_enc") or
            args.model.startswith(("tgat_time_", "tgat_bin_"))):
        m = TGATModel(
            checkpoint_path=checkpoint,
            num_nodes=num_nodes,
            msg_dim=msg_dim,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
        )
        m.load_checkpoint()
    elif args.model == "graphmixer" or args.model.startswith(("graphmixer_time_", "graphmixer_bin_")):
        m = GraphMixerModel(
            checkpoint_path=checkpoint,
            num_nodes=num_nodes,
            msg_dim=msg_dim,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
        )
        m.load_checkpoint()
    elif args.model == "graphmixer_multiscale":
        m = MultiScaleGraphMixerModel(
            checkpoint_path=checkpoint,
            num_nodes=num_nodes,
            msg_dim=msg_dim,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
        )
        m.load_checkpoint()
    elif args.model == "tpnet":
        m = TPNetModel(
            checkpoint_path=checkpoint,
            num_nodes=num_nodes,
            msg_dim=msg_dim,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
        )
        m.load_checkpoint()
    elif args.model == "edgebank":
        m = EdgeBankModel()
        m.load_checkpoint()
    else:
        _log(f"Unknown model: {args.model}")
        sys.exit(1)

    # ── Run evaluation ─────────────────────────────────────────────────────────
    from evaluate.evaluator import Evaluator
    ev = Evaluator(
        dataset=dataset_obj,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        dataset_name=args.dataset,
        neg_cache_dir=neg_cache_dir,
        num_neg=args.num_neg,
        seed=args.seed,
    )

    results = ev.run(
        model=m,
        model_name=args.model,
        skip_standard=args.skip_standard,
        standard_mrr=args.standard_mrr,
    )
    results["model"]   = args.model
    results["dataset"] = args.dataset
    results["seed"]    = args.seed
    results["job_id"] = os.environ.get("SLURM_JOB_ID")
    results["checkpoint_path"] = os.path.abspath(checkpoint) if checkpoint else None
    if checkpoint:
        digest = hashlib.sha256()
        with open(checkpoint, "rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        results["checkpoint_sha256"] = digest.hexdigest()
    else:
        results["checkpoint_sha256"] = None
    results["encoding_mode"] = ("binned" if "_bin_" in args.model else
                                "scaled" if "_time_" in args.model else None)
    results["bin_width"] = next(
        (width for gran, width in _TIME_GRAN_WIDTHS.items()
         if args.model.endswith((f"_bin_{gran}", f"_time_{gran}"))), None)
    results["time_scale"] = (1.0 / results["bin_width"]
                             if results["encoding_mode"] == "scaled" else None)

    # ── Save results ───────────────────────────────────────────────────────────
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{args.model}_{args.dataset}_{ts}.json"
    out   = os.path.abspath(args.output or os.path.join(results_dir, fname))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    results["run_id"] = os.path.basename(os.path.dirname(out)) if args.output else None
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    _log(f"Results saved to: {out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
