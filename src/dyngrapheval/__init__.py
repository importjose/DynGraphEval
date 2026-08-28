"""
DynGraphEval: A simplified evaluation framework for continuous-time temporal graph models.

Evaluates models on two dimensions:
    1. Standard MRR  — TGB hist_rnd negatives (leaderboard-comparable)
    2. Recency MRR   — negatives from source's recent interaction history

Models:
    TGN          — Centralized Temporal Graph Network
    FederatedTGN — Federated TGN (4 clients, source-node partitioned)
    FedLink      — Static GraphSAGE baseline (no temporal reasoning)

Quick start:
    from dyngrapheval import Evaluator
    from dyngrapheval.models import TGN

    model = TGN(checkpoint_path='saved_models/tgn_best.pt', ...)
    model.load_checkpoint()

    ev = Evaluator(dataset, train_data, val_data, test_data,
                   first_dst_id=min_dst_idx, last_dst_id=max_dst_idx)
    results = ev.run(model, model_name='Centralized TGN')
"""

from .evaluator import Evaluator
from .base_model import BaseModel
from .negative_sampler import RecencyNegativeGenerator, NegativeSampler

__all__ = [
    "Evaluator",
    "BaseModel",
    "RecencyNegativeGenerator",
    "NegativeSampler",
]
