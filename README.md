# DynGraphEval

Evaluation framework for temporal graph link prediction models on the [TGB](https://tgb.complexdatalab.com/) benchmark. Designed to isolate *what temporal graph models actually learn* — beyond the standard leaderboard metric.

**[Full documentation →](https://importjose.github.io/DynGraphEval/)**

---

## Motivation

Standard MRR measures whether a model can rank a link above distractors. But temporal graph models are supposed to do something harder: **predict the next event in a stream**, not just recognize that a link has historically existed. A model can score well on Standard MRR by learning a simple recency heuristic, without genuinely reasoning about time.

This framework decomposes evaluation along two axes:

1. **Recency MRR** — negatives are the *K most recently visited* unique destinations for each source, ordered most-recent-first. This directly tests whether a model can separate the true next link from the links it just visited.
2. **Return vs. Explore split** — each test edge is labeled as *Return* (source has visited this destination before) or *Explore* (never visited). Measures whether a model's advantage comes from recall or generalization.
3. **K-curve** — Recency MRR computed at K ∈ {10, 20, 50, 100, 500, 999} from a single evaluation pass. Shows at what temporal depth a model's memory stops being useful.

---

## Key Findings (tgbl-wiki, seed=0)

### The 2×2 Architectural Matrix

|  | Trainable encoder | Fixed encoder |
|--|:--:|:--:|
| **Memory bank** | TGN | TGN+FixedEnc |
| **No memory** | TGAT | GraphMixer |

| Model | Memory Bank | Time Encoder | Standard MRR | Recency MRR | Return MRR | Explore MRR | Return/Explore Gap |
|-------|:-----------:|:------------:|:---:|:---:|:---:|:---:|:---:|
| EdgeBank | ✓ | — | 0.495 | 0.362 | 0.392 | 0.284 | 0.108 |
| TGN | ✓ | Trainable | 0.601 | 0.789 | 0.821 | 0.708 | 0.113 |
| TGN+FixedEnc | ✓ | Fixed | **0.692** | 0.804 | 0.837 | 0.720 | 0.117 |
| TGAT | ✗ | Trainable | 0.568 | 0.773 | 0.797 | 0.711 | 0.086 |
| GraphMixer | ✗ | Fixed | 0.549 | **0.821** | **0.834** | **0.787** | **0.047** |

### Key observations

**Fixed encoder is competitive with trainable** — TGN+FixedEnc matches or exceeds TGN across all metrics. The learnable time encoder provides no meaningful benefit; the memory bank alone drives Standard MRR differences within that row.

**GraphMixer has the smallest Return/Explore gap (0.047)** despite having neither a memory bank nor a learned encoder. It generalizes best to unseen (Explore) edges — the opposite of what a recency-biased model would do.

**EdgeBank collapses on Explore edges (0.284)** — pure memorization fails on novel interactions, as expected. Its Return/Explore gap (0.108) is similar to TGN's (0.113), showing that a full GNN with memory adds little over a lookup table for return edges.

**Standard MRR vs Recency MRR inversion holds** — the model with the highest Standard MRR (TGN+FixedEnc) does not win Recency MRR (GraphMixer). The trainable time encoder learns to upweight recent neighbors, inflating recency negative scores and making the true next link harder to rank.

### K-curve: flat for all models

| Model | K=10 | K=20 | K=50 | K=100 | K=999 | Drop |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| TGN | 0.802 | 0.794 | 0.791 | 0.790 | 0.789 | −0.013 |
| TGN+FixedEnc | 0.816 | 0.809 | 0.806 | 0.805 | 0.804 | −0.012 |
| TGAT | 0.785 | 0.778 | 0.774 | 0.773 | 0.773 | −0.012 |
| GraphMixer | **0.835** | **0.828** | **0.824** | **0.822** | **0.821** | −0.014 |

All models drop only 0.012–0.014 from K=10 to K=999. Architectural bias is fully expressed at K=10 — adding more historically-recent negatives beyond the first 10 does not change the ranking.

---

## Project Structure

```
DynGraphEval/
├── models/
│   ├── base.py                  # BaseModel ABC (load_checkpoint, warmup, evaluate)
│   ├── tgn/
│   │   ├── model.py             # Centralized TGN wrapper
│   │   ├── tpnet_components.py  # NeighborSampler, TimeEncoder, MemoryModel, LinkPredictor
│   │   └── train.py
│   ├── graphmixer/
│   │   ├── model.py             # GraphMixerModel wrapper (no memory, fixed encoder)
│   │   ├── graphmixer_components.py
│   │   └── train.py
│   ├── tgat/
│   │   ├── model.py             # TGATModel wrapper (no memory, trainable encoder)
│   │   ├── tgat_components.py
│   │   └── train.py
│   ├── tpnet/
│   │   ├── model.py             # TPNetModel wrapper
│   │   └── train.py
│   └── tgn_fixed_enc/
│       └── train.py             # TGN ablation: frozen time encoder
├── evaluate/
│   ├── evaluator.py             # Standard MRR + Recency MRR + Return/Explore + K-curve
│   ├── negative_sampler.py      # RecencyNegativeGenerator + NegativeSampler
│   └── partition.py             # Source-node partitioning
├── modal/
│   ├── train.py                 # Modal training app (A10 GPU)
│   └── eval.py                  # Modal eval app (T4 GPU)
├── results/                     # Eval output JSONs
└── docs/                        # GitHub Pages documentation
```

---

## Usage

### Training on Modal

```bash
modal run --detach modal/train.py --model tgn
modal run --detach modal/train.py --model graphmixer
modal run --detach modal/train.py --model tgat
modal run --detach modal/train.py --model tgn_fixed_enc
```

Checkpoints are saved to `/data/checkpoints/{model}/{dataset}/run0.pkl` on a Modal Volume.

### Evaluation on Modal

```bash
# Full eval (Standard + Recency MRR)
modal run modal/eval.py --model tgn

# Skip Standard MRR and pass it directly (training Test MRR = Standard MRR)
modal run modal/eval.py --model graphmixer --skip-standard --standard-mrr 0.549
modal run modal/eval.py --model tgat --skip-standard --standard-mrr 0.568

# Fetch results
modal volume get dyngrapheval-data results/
```

Results are written to `results/{model}_{dataset}_{timestamp}.json`.

---

## Open Issues

- [#3](https://github.com/importjose/DynGraphEval/issues/3) **Multi-dataset**: tgbl-review, tgbl-coin, tgbl-comment
- [#4](https://github.com/importjose/DynGraphEval/issues/4) **Statistical validation**: 3 seeds, confirm inversion is stable
- [#8](https://github.com/importjose/DynGraphEval/issues/8) **Ablation**: train all models with recent historical negatives vs random — does training signal transfer?

---

## Dependencies

```bash
pip install torch==2.4.0 torch-geometric==2.6.1 py-tgb>=2.2 numpy pandas tqdm modal
```

Python 3.11+. Training requires a CUDA GPU.
