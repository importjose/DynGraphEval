# DynGraphEval

Evaluation framework for temporal graph link prediction models on the [TGB](https://tgb.complexdatalab.com/) benchmark. Compares centralized and federated TGN variants across two evaluation dimensions.

## Evaluation Dimensions

| Metric | Negatives | What it tests |
|--------|-----------|---------------|
| **Standard MRR** | TGB `hist_rnd` (50% historical + 50% random) | Direct comparison with TGB leaderboard |
| **Recency MRR** | Pages source visited just *before* the query time | Temporal reasoning — can the model distinguish "visiting now" from "visited recently"? |

Both metrics use 100 negatives per edge so they are directly comparable.

## Models

| Model | Type | Description |
|-------|------|-------------|
| `tgn` | Centralized | TPNet-style TGN: 2-layer graph attention + GRU memory |
| `fl_tgn` | Federated (4 clients) | Same TGN architecture, source-node partitioned data, FedAvg weights |
| `fedlink` | Federated (4 clients) | Static GraphSAGE baseline — no temporal reasoning |

## Project Structure

```
DynGraphEval/
├── models/
│   ├── base.py              # BaseModel ABC (load_checkpoint, warmup, evaluate)
│   ├── components.py        # Shared nn.Modules (TGNMemory, GraphAttentionEmbedding, ...)
│   ├── tgn/
│   │   ├── model.py         # Centralized TGN wrapper
│   │   ├── tpnet_components.py  # TPNet MemoryModel, NeighborSampler, LinkPredictor
│   │   ├── train.py         # Training launcher (clones TPNet, applies patches, runs training)
│   │   └── patches/         # Patched TPNet files (memory cleanup, resume checkpoints, logging)
│   ├── fl_tgn/
│   │   └── model.py         # Federated TGN wrapper (4 clients)
│   └── fedlink/
│       └── model.py         # FedLink static GraphSAGE wrapper (4 clients)
├── evaluate/
│   ├── evaluator.py         # Evaluator: runs Standard MRR + Recency MRR
│   ├── negative_sampler.py  # RecencyNegativeGenerator + NegativeSampler
│   └── partition.py         # Source-node partitioning for federated models
├── modal/
│   ├── train.py             # Modal training app (A10 GPU, persistent volume)
│   └── eval.py              # Modal eval app (T4 GPU)
├── checkpoints/             # Model checkpoints — gitignored, stored on Modal Volume
├── results/                 # Eval output JSONs — gitignored
└── eval.ipynb               # Result visualization
```

## Training on Modal

Training runs on [Modal](https://modal.com) (pay-per-second GPU, ~$1.10/hr on A10). Datasets and checkpoints persist across runs via a Modal Volume.

### First-time setup

```bash
pip install modal
modal setup          # authenticate
```

### Run training

```bash
# from DynGraphEval root
modal run modal/train.py
modal run modal/train.py --dataset tgbl-wiki --epochs 50 --seed 1
```

Training automatically resumes from the last completed epoch if interrupted. Checkpoints are saved to the Modal Volume at `/data/checkpoints/{dataset}/`.

### Run evaluation

```bash
modal run modal/eval.py --model tgn
modal run modal/eval.py --model fl_tgn --checkpoint /data/checkpoints/tgbl-wiki/run0.pkl
```

## Local Evaluation

If you already have a checkpoint, evaluation can be run locally (GPU recommended):

```python
import torch
from tgb.linkproppred.dataset import PyGLinkPropPredDataset

from models.tgn.model import TPNetTGN
from evaluate.evaluator import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PyGLinkPropPredDataset(name="tgbl-wiki", root="datasets/")
data = dataset.get_TemporalData()
split = dataset.get_idx_split()
train_data, val_data, test_data = data[split["train"]], data[split["val"]], data[split["test"]]

model = TPNetTGN(
    checkpoint_path="checkpoints/tgbl-wiki/run0.pkl",
    num_nodes=int(max(data.src.max(), data.dst.max())) + 1,
    msg_dim=data.msg.shape[1],
    train_data=train_data, val_data=val_data, test_data=test_data,
    device=device,
)
model.load_checkpoint()

ev = Evaluator(
    dataset=dataset, train_data=train_data, val_data=val_data, test_data=test_data,
    first_dst_id=int(data.dst.min()), last_dst_id=int(data.dst.max()),
)
results = ev.run(model, model_name="tgn")
# {'model': 'tgn', 'dataset': 'tgbl-wiki', 'standard_mrr': 0.497, 'recency_mrr': ...}
```

## Known Results (tgbl-wiki)

| Model | Standard MRR | Recency MRR |
|-------|-------------|-------------|
| TGN (centralized) | 0.497 | TBD |
| FL-TGN (4 clients) | 0.339 | TBD |
| FedLink (4 clients) | 0.017 | TBD |

## Dependencies

```bash
pip install torch torch-geometric py-tgb numpy pandas tqdm pyyaml modal
```

Python 3.11+. Training requires a CUDA GPU.
