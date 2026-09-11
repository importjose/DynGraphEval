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

### Standard MRR vs Recency MRR: a complete inversion

| Model | Memory | Time encoder | Standard MRR | Recency MRR |
|-------|--------|-------------|:---:|:---:|
| TGN | GRU memory bank | Trainable | **0.601** | 0.789 |
| TGAT | None | Trainable | 0.568 | 0.773 |
| GraphMixer | None | Fixed | 0.549 | **0.821** |

The model that wins Standard MRR loses Recency MRR, and vice versa. The trainable time encoder learns to upweight recent neighbors — those are exactly the recency negatives, inflating their scores and making the true next link harder to rank.

### Return vs Explore: the gap scales with temporal learning

| Model | Return MRR | Explore MRR | Gap |
|-------|:---:|:---:|:---:|
| TGN | 0.821 | 0.708 | **0.113** |
| TGAT | 0.797 | 0.711 | 0.086 |
| GraphMixer | **0.834** | **0.787** | 0.047 |

More temporal learning → larger Return/Explore gap. TGN and TGAT score nearly identically on Explore (0.708 vs 0.711) despite TGN having a full memory bank — the trainable encoder, not the memory bank, drives Explore collapse. GraphMixer beats TGN even on Return edges.

### K-curve: flat for all models

| Model | K=10 | K=20 | K=50 | K=100 | K=999 | Drop |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| TGN | 0.802 | 0.794 | 0.791 | 0.790 | 0.789 | −0.013 |
| TGAT | 0.785 | 0.778 | 0.774 | 0.773 | 0.773 | −0.012 |
| GraphMixer | **0.835** | **0.828** | **0.824** | **0.822** | **0.821** | −0.014 |

All models drop only 0.012–0.014 from K=10 to K=999. The predicted GraphMixer drop-off past K=20 did not materialize — architectural bias is fully expressed at K=10. Adding more historically-recent negatives beyond the first 10 does not change the ranking.

---

## The Architectural 2×2

|  | Trainable encoder | Fixed encoder |
|--|:--:|:--:|
| **Memory bank** | TGN ✓ | **— missing —** |
| **No memory** | TGAT ✓ | GraphMixer ✓ |

No published model combines a persistent memory bank with a fixed time encoder. This is the subject of the next ablation study (see [open issues](https://github.com/importjose/DynGraphEval/issues)).

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
│   └── tpnet/
│       ├── model.py             # TPNetModel wrapper
│       └── train.py
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

See the [Evaluation Completeness milestone](https://github.com/importjose/DynGraphEval/milestone/1) for planned work:

- [#1](https://github.com/importjose/DynGraphEval/issues/1) **Ablation**: train memory-bank model with fixed time encoder (fills missing 2×2 cell)
- [#2](https://github.com/importjose/DynGraphEval/issues/2) **Baseline**: evaluate EdgeBank with Recency MRR
- [#3](https://github.com/importjose/DynGraphEval/issues/3) **Multi-dataset**: tgbl-review, tgbl-coin, tgbl-comment
- [#4](https://github.com/importjose/DynGraphEval/issues/4) **Statistical validation**: 3 seeds, confirm inversion is stable

---

## Dependencies

```bash
pip install torch==2.4.0 torch-geometric==2.6.1 py-tgb>=2.2 numpy pandas tqdm modal
```

Python 3.11+. Training requires a CUDA GPU.
