"""
Modal evaluation app for DynGraphEval.

Run from the DynGraphEval root directory:
    modal run modal/eval.py --model tgn
    modal run modal/eval.py --model tgn --checkpoint /data/checkpoints/tgbl-wiki/run0.pkl

Prints Standard MRR + Recency MRR results as JSON.
"""

import os
import sys
import json

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
    )
)

volume     = modal.Volume.from_name("dyngrapheval-data", create_if_missing=True)
VOLUME_PATH     = "/data"
DATASETS_DIR    = f"{VOLUME_PATH}/datasets"
CHECKPOINTS_DIR = f"{VOLUME_PATH}/checkpoints"

repo_mount = modal.Mount.from_local_dir(
    local_path=".",
    remote_path="/repo",
    condition=lambda p: not any(
        p.startswith(seg) for seg in [".git/", "__pycache__/", "checkpoints/", "datasets/", "neg_cache/"]
    ),
)

app = modal.App("dyngrapheval-eval")


@app.function(
    image=image,
    gpu="T4",           # eval needs less VRAM than training
    timeout=3600,
    volumes={VOLUME_PATH: volume},
    mounts=[repo_mount],
)
def evaluate(
    model:      str  = "tgn",
    checkpoint: str  = None,
    dataset:    str  = "tgbl-wiki",
    num_neg:    int  = 100,
    seed:       int  = 42,
) -> dict:
    """
    Run Standard MRR + Recency MRR for a given model checkpoint.

    Parameters
    ----------
    model      : 'tgn', 'fl_tgn', or 'fedlink'
    checkpoint : path to checkpoint on the volume.
                 Defaults to /data/checkpoints/{dataset}/run0.pkl
    dataset    : TGB dataset name (default 'tgbl-wiki')
    num_neg    : negatives per edge for both metrics (default 100)
    seed       : random seed (default 42)

    Returns
    -------
    dict with keys: model, dataset, standard_mrr, recency_mrr
    """
    import torch
    from tgb.linkproppred.dataset import PyGLinkPropPredDataset

    sys.path.insert(0, "/repo")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_obj = PyGLinkPropPredDataset(
        name=dataset,
        root=DATASETS_DIR,
    )
    data       = dataset_obj.get_TemporalData()
    split_masks = dataset_obj.get_idx_split()

    train_data = data[split_masks["train"]]
    val_data   = data[split_masks["val"]]
    test_data  = data[split_masks["test"]]

    min_dst   = int(data.dst.min())
    max_dst   = int(data.dst.max())
    num_nodes = int(max(data.src.max(), data.dst.max())) + 1
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Default checkpoint path ───────────────────────────────────────────────
    if checkpoint is None:
        checkpoint = os.path.join(CHECKPOINTS_DIR, dataset, "run0.pkl")

    # ── Instantiate and load model ────────────────────────────────────────────
    from models.tgn.model import TPNetTGN
    from models.fl_tgn.model import FederatedTGN
    from models.fedlink.model import FedLink

    if model == "tgn":
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
        raise ValueError(f"Unknown model '{model}'. Choose from: tgn, fl_tgn, fedlink")

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
    results = ev.run(m, model_name=model)
    return results


@app.local_entrypoint()
def main(
    model:      str = "tgn",
    checkpoint: str = None,
    dataset:    str = "tgbl-wiki",
    num_neg:    int = 100,
):
    """
    CLI entrypoint.

    Examples:
        modal run modal/eval.py --model tgn
        modal run modal/eval.py --model tgn --checkpoint /data/checkpoints/tgbl-wiki/run0.pkl
    """
    result = evaluate.remote(
        model=model,
        checkpoint=checkpoint,
        dataset=dataset,
        num_neg=num_neg,
    )
    print(json.dumps(result, indent=2))
