# DynGraphEval

Evaluation framework for temporal graph link prediction models on the [TGB](https://tgb.complexdatalab.com/) benchmark. Supports centralized and federated models across a multi-dimensional evaluation framework designed to isolate *what temporal graph models actually learn*.

---

## Research Question

Standard MRR measures whether a model can rank a link above distractors. But temporal graph models are supposed to do something harder: **predict the next event in a stream**, not just recognize that a link exists. This framework decomposes that question along two axes:

1. **Return vs. Explore edges** — Is the true next link a page the source visited before (memory task) or never visited (generalization task)?
2. **Recency horizon K** — At what temporal depth does a model's memory stop being useful?

Together these reveal *where* temporal memory helps and *where* federation hurts — not just by how much.

---

## Evaluation Dimensions

### Current (implemented)

| Metric | Negatives | What it tests |
|--------|-----------|---------------|
| **Standard MRR** | TGB `hist_rnd` (50% historical + 50% random, 999/edge) | Leaderboard-comparable link prediction |
| **Recency MRR** | Last-K unique destinations visited by source before query time (no random fill) | Can the model distinguish "visiting now" from "visited recently"? |

Standard MRR uses TGB's fixed negatives and is directly comparable to published leaderboard results.
Recency MRR uses negatives you construct — edges with sparse history get fewer negatives, so the two metrics are not on the same scale. What matters is the **relative ordering across models**.

### Planned

**Return/Explore split** — label every test edge:
- **Return**: source has visited this destination before → tests temporal recall
- **Explore**: source has never visited this destination → tests structural generalization

Run Standard MRR and Recency MRR separately on each subset. Expected finding: memory banks help on Return edges; no model has an advantage on Explore edges.

**Recency MRR at multiple K values** — compute Recency MRR at K = [10, 20, 50, 100, 500, 999] for each model. Produces a **temporal reach curve**:
- GraphMixer (20-neighbor window): curve drops as K grows past ~20
- TGN (memory bank): flatter curve, retains signal at large K
- Crossover point = the horizon at which accumulated memory beats recency structure

**Combined: K curve × Return/Explore** — the full 2D table per model:

```
                  Return edges    Explore edges
Standard MRR      ?               ?             ← one number each, TGB negatives
Recency K=10      ?               ?
Recency K=20      ?               ?
Recency K=50      ?               ?
Recency K=100     ?               ?
Recency K=999     ?               ?
```

Expected pattern:
- TGN beats GraphMixer on Return × large K (memory bank captures long-range history)
- All models perform similarly on Explore × any K (no model can recall a link that never existed)
- FL-TGN loses most on Return × large K (partitioned memory bank breaks long-range recall)
- FL-TGN loses on Explore (federation prevents cross-client structural generalization)

---

## Models

### Centralized baselines

| Model | Memory | Time Encoder | Standard MRR | Recency MRR | Status |
|-------|--------|-------------|-------------|-------------|--------|
| `tgn` | Yes (GRU) | Trainable | 0.601 | 0.789 | ✅ done |
| `tgat` | No | Trainable | 0.568 | 0.773 | ✅ done |
| `graphmixer` | No | Fixed | 0.549 | 0.821 | ✅ done |
| `tpnet` | No (random proj.) | Fixed | TBD | TBD | 🔄 training |

### Federated models (next phase)

| Model | Type | Status |
|-------|------|--------|
| `fl_tgn` | Federated TGN, 4 clients, FedAvg | ⏳ pending |
| `fedlink` | Federated static GraphSAGE, 4 clients | ⏳ pending |

---

## Key Findings So Far (tgbl-wiki, seed=0)

**Learned temporal weighting helps Standard MRR, hurts Recency MRR:**

| Model | Standard MRR | Recency MRR |
|-------|-------------|-------------|
| TGN (memory + trainable encoder) | **0.601** | 0.789 |
| TGAT (no memory + trainable encoder) | 0.568 | 0.773 |
| GraphMixer (no memory + fixed encoder) | 0.549 | **0.821** |

GraphMixer wins on Recency MRR despite being weakest on Standard MRR. Hypothesis: TGAT and TGN learn to upweight recent neighbors — those are exactly the recency negatives, inflating their scores and making ranking harder. GraphMixer's fixed encoder treats all neighbors uniformly.

The Return/Explore split and K curve are needed to confirm whether this is a memory depth effect or an architectural alignment artifact.

---

## Project Structure

```
DynGraphEval/
├── models/
│   ├── base.py                  # BaseModel ABC (load_checkpoint, warmup, evaluate)
│   ├── tgn/
│   │   ├── model.py             # Centralized TGN wrapper
│   │   ├── tpnet_components.py  # NeighborSampler, TimeEncoder, MemoryModel, LinkPredictor
│   │   ├── train.py             # Training launcher
│   │   └── patches/             # Patched TPNet files
│   ├── graphmixer/
│   │   ├── model.py             # GraphMixerModel wrapper (no memory, fixed encoder)
│   │   ├── graphmixer_components.py
│   │   ├── train.py
│   │   └── patches/
│   ├── tgat/
│   │   ├── model.py             # TGATModel wrapper (no memory, trainable encoder)
│   │   ├── tgat_components.py
│   │   ├── train.py
│   │   └── patches/
│   ├── tpnet/
│   │   ├── model.py             # TPNetModel wrapper (random projections, SOTA)
│   │   ├── tpnet_components.py
│   │   ├── train.py
│   │   └── patches/
│   ├── fl_tgn/
│   │   └── model.py             # Federated TGN (4 clients, source-node partition)
│   └── fedlink/
│       └── model.py             # FedLink static GraphSAGE (4 clients)
├── evaluate/
│   ├── evaluator.py             # Runs Standard MRR + Recency MRR (skip_standard flag)
│   ├── negative_sampler.py      # RecencyNegativeGenerator + NegativeSampler
│   └── partition.py             # Source-node partitioning for federated models
├── modal/
│   ├── train.py                 # Modal training app (A10 GPU, --model flag)
│   └── eval.py                  # Modal eval app (T4 GPU, --skip-standard flag)
├── results/                     # Eval output JSONs
└── README.md
```

---

## Training on Modal

```bash
# Train any model (detach so local client doesn't need to stay connected)
modal run --detach modal/train.py --model tgn
modal run --detach modal/train.py --model graphmixer
modal run --detach modal/train.py --model tgat
modal run --detach modal/train.py --model tpnet

# Checkpoints saved to /data/checkpoints/{model}/{dataset}/run0.pkl on Modal Volume
```

## Evaluation on Modal

```bash
# Full eval (Standard + Recency MRR)
modal run --detach modal/eval.py --model tgn

# Skip Standard MRR for memory-free models (training Test MRR = Standard MRR)
modal run --detach modal/eval.py --model graphmixer --skip-standard --standard-mrr 0.549
modal run --detach modal/eval.py --model tgat --skip-standard --standard-mrr 0.568

# Fetch results from volume
modal volume get dyngrapheval-data results/
```

## Dependencies

```bash
pip install torch==2.4.0 torch-geometric==2.6.1 py-tgb>=2.2 numpy pandas tqdm modal
```

Python 3.11+. Training requires a CUDA GPU.
