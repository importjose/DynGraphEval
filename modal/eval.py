"""
Modal evaluation app for DynGraphEval.

Run from the DynGraphEval root directory:
    modal run modal/eval.py --model tgn
    modal run modal/eval.py --model tgn --checkpoint /data/checkpoints/tgn/tgbl-wiki/run0.pkl

Prints Standard MRR + Recency MRR results as JSON.
"""

import os
import sys
import json
import datetime
import time

import modal

# ── Image (same deps as train.py) ─────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torch-geometric==2.6.1",
        "py-tgb>=2.2",
        "numpy>=2.0",
        "pandas>=2.0",
        "tqdm>=4.65",
        "pyyaml>=6.0",
        # Match train image exactly so py-tgb resolves to the same version
        "numba",
        "wandb",
        "scikit-learn",
        "scipy",
    )
    .add_local_dir(".", remote_path="/repo", copy=True)
)

volume          = modal.Volume.from_name("dyngrapheval-data", create_if_missing=True)
VOLUME_PATH     = "/data"
DATASETS_DIR    = f"{VOLUME_PATH}/datasets"
CHECKPOINTS_DIR = f"{VOLUME_PATH}/checkpoints"

app = modal.App("dyngrapheval-eval")


@app.function(
    image=image,
    gpu="T4",           # eval needs less VRAM than training
    timeout=14400,      # 4 hours: warmup×2 (~5min each) + scoring×2 (~35min each)
    volumes={VOLUME_PATH: volume},
)
def evaluate(
    model:          str   = "tgn",
    checkpoint:     str   = None,
    dataset:        str   = "tgbl-wiki",
    num_neg:        int   = 999,
    seed:           int   = 42,
    skip_standard:  bool  = False,
    standard_mrr:   float = None,
) -> dict:
    """
    Run Standard MRR + Recency MRR for a given model checkpoint.

    Parameters
    ----------
    model      : 'edgebank', 'tgn', 'graphmixer', 'tgat', 'fl_tgn', or 'fedlink'
    checkpoint : path to checkpoint on the volume.
                 Defaults to /data/checkpoints/{model}/{dataset}/run0.pkl
    dataset    : TGB dataset name (default 'tgbl-wiki')
    num_neg    : negatives per edge for both metrics (default 100)
    seed       : random seed (default 42)

    Returns
    -------
    dict with keys: model, dataset, standard_mrr, recency_mrr
    """
    import torch
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

    sys.path.insert(0, "/repo")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_obj = PyGLinkPropPredDataset(
        name=dataset,
        root=DATASETS_DIR,
    )
    data       = dataset_obj.get_TemporalData()

    train_data = data[dataset_obj.train_mask]
    val_data   = data[dataset_obj.val_mask]
    test_data  = data[dataset_obj.test_mask]

    min_dst   = int(data.dst.min())
    max_dst   = int(data.dst.max())
    num_nodes = int(max(data.src.max(), data.dst.max())) + 1
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Default checkpoint path (not used by edgebank) ───────────────────────
    if checkpoint is None and model != "edgebank":
        checkpoint = os.path.join(CHECKPOINTS_DIR, model, dataset, "run0.pkl")

    # ── Instantiate and load model ────────────────────────────────────────────
    from models.tgn.model import TPNetTGN
    from models.graphmixer.model import GraphMixerModel
    from models.tgat.model import TGATModel
    from models.tpnet.model import TPNetModel
    from models.fl_tgn.model import FederatedTGN
    from models.fedlink.model import FedLink
    from models.edgebank.model import EdgeBankModel

    if model == "edgebank":
        # No checkpoint needed — warmup() builds memory from train+val
        m = EdgeBankModel()
        m.load_checkpoint()

    elif model == "tgn":
        m = TPNetTGN(
            checkpoint_path=checkpoint,
            num_nodes=num_nodes,
            msg_dim=data.msg.shape[1],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
        )
        m.load_checkpoint()

    elif model == "tpnet":
        m = TPNetModel(
            checkpoint_path=checkpoint,
            num_nodes=num_nodes,
            msg_dim=data.msg.shape[1],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
        )
        m.load_checkpoint()

    elif model == "tgat":
        m = TGATModel(
            checkpoint_path=checkpoint,
            num_nodes=num_nodes,
            msg_dim=data.msg.shape[1],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
        )
        m.load_checkpoint()

    elif model == "graphmixer":
        m = GraphMixerModel(
            checkpoint_path=checkpoint,
            num_nodes=num_nodes,
            msg_dim=data.msg.shape[1],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
        )
        m.load_checkpoint()

    elif model == "fl_tgn":
        # checkpoint may be a single path (replicated across clients) or a list
        paths = checkpoint if isinstance(checkpoint, list) else [checkpoint] * 4
        m = FederatedTGN(
            checkpoint_paths=paths,
            num_nodes=num_nodes,
            msg_dim=data.msg.shape[1],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
        )
        m.load_checkpoint()

    elif model == "fedlink":
        paths    = checkpoint if isinstance(checkpoint, list) else [checkpoint] * 4
        num_users = int(data.src.max()) + 1
        num_pages = max_dst - min_dst + 1
        m = FedLink(
            checkpoint_paths=paths,
            num_users=num_users,
            num_pages=num_pages,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            min_dst_idx=min_dst,
            device=device,
        )
        m.load_checkpoint()

    else:
        raise ValueError(f"Unknown model '{model}'. Choose from: edgebank, tgn, tgat, graphmixer, tpnet, fl_tgn, fedlink")

    # ── Run evaluation ────────────────────────────────────────────────────────
    from evaluate.evaluator import Evaluator

    ev = Evaluator(
        dataset=dataset_obj,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        first_dst_id=min_dst,
        last_dst_id=max_dst,
        dataset_name=dataset,
        neg_cache_dir=os.path.join(VOLUME_PATH, "neg_cache"),
        num_neg=num_neg,
        seed=seed,
    )
    results = ev.run(m, model_name=model, skip_standard=skip_standard, standard_mrr=standard_mrr)

    # Save results to Modal Volume so they persist across runs
    results_dir = os.path.join(VOLUME_PATH, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    volume_path = os.path.join(results_dir, f"{model}_{dataset}_{ts}.json")
    with open(volume_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"[eval] Results saved to volume: {volume_path}")

    return results


@app.local_entrypoint()
def main(
    model:         str   = "tgn",
    checkpoint:    str   = None,
    dataset:       str   = "tgbl-wiki",
    num_neg:       int   = 999,
    skip_standard: bool  = False,
    standard_mrr:  float = None,
):
    """
    CLI entrypoint.

    Examples:
        modal run modal/eval.py --model tgn
        modal run modal/eval.py --model graphmixer
        modal run modal/eval.py --model tgn --checkpoint /data/checkpoints/tgn/tgbl-wiki/run0.pkl
    """
    result = evaluate.remote(
        model=model,
        checkpoint=checkpoint,
        dataset=dataset,
        num_neg=num_neg,
        skip_standard=skip_standard,
        standard_mrr=standard_mrr,
    )
    print(json.dumps(result, indent=2))

    # Save to results/ with a timestamp so runs aren't overwritten
    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("results", f"{model}_{dataset}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")
