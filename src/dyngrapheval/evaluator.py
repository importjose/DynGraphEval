"""
Evaluation orchestrator for DynGraphEval.

The Evaluator class runs two evaluation dimensions on any BaseModel:

    1. Standard MRR  — TGB's built-in hist_rnd negatives
       Directly comparable with the TGB leaderboard.
       Negatives: 50% from source's historical destinations, 50% random pages.

    2. Recency MRR   — temporally hard negatives from source's recent history
       For each positive edge (u, v, t), negatives are pages u visited in
       [t - W, t), where W is the adaptive per-source median inter-event time.
       A model that ignores time cannot distinguish "u is visiting v NOW" from
       "u just visited this page recently."

Usage
-----
    from dyngrapheval import Evaluator
    from dyngrapheval.models import TGN

    model = TGN(checkpoint_path='...', ...)
    model.load_checkpoint()
    model.warmup(train_data, val_data)

    ev = Evaluator(dataset, train_data, val_data, test_data,
                   first_dst_id=min_dst_idx, last_dst_id=max_dst_idx)
    results = ev.run(model)
    # {'standard_mrr': 0.4971, 'recency_mrr': 0.4850}
"""

import json
import random
import torch
from tgb.linkproppred.evaluate import Evaluator as TGBEvaluator

from .base_model import BaseModel
from .negative_sampler import RecencyNegativeGenerator, NegativeSampler


class _CappedNegSampler:
    """
    Thin wrapper that subsamples TGB negatives to a fixed count per edge.

    TGB tgbl-wiki test provides 999 negatives per edge in py-tgb 2.2+.
    Capping to num_neg (default 100) makes Standard MRR and Recency MRR
    use the same pool size and therefore directly comparable.
    """

    def __init__(self, sampler, num_neg: int, seed: int = 42):
        self._sampler = sampler
        self._num_neg = num_neg
        self._rng     = random.Random(seed)

    def query_batch(self, pos_src, pos_dst, pos_t, split_mode):
        full = self._sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)
        return [
            self._rng.sample(list(neg), min(self._num_neg, len(neg)))
            if len(neg) > self._num_neg else list(neg)
            for neg in full
        ]


class Evaluator:
    """
    Runs Standard MRR and Recency MRR on any BaseModel.

    Parameters
    ----------
    dataset       : PyGLinkPropPredDataset  — TGB dataset object (provides neg sampler)
    train_data    : TemporalData
    val_data      : TemporalData
    test_data     : TemporalData
    first_dst_id  : int   smallest valid destination node ID
    last_dst_id   : int   largest valid destination node ID
    dataset_name  : str   used for logging and filenames (default 'tgbl-wiki')
    neg_cache_dir : str   directory to cache generated negatives (default 'neg_cache')
    num_neg       : int   negatives per positive edge (default 100)
    seed          : int   random seed for negative generation (default 42)
    """

    def __init__(
        self,
        dataset,
        train_data,
        val_data,
        test_data,
        first_dst_id: int,
        last_dst_id: int,
        dataset_name: str = "tgbl-wiki",
        neg_cache_dir: str = "neg_cache",
        num_neg: int = 100,
        seed: int = 42,
    ):
        self.dataset       = dataset
        self.train_data    = train_data
        self.val_data      = val_data
        self.test_data     = test_data
        self.dataset_name  = dataset_name
        self.neg_cache_dir = neg_cache_dir
        self.num_neg       = num_neg
        self.seed          = seed

        # TGB evaluator: computes MRR from y_pred_pos and y_pred_neg
        self.tgb_evaluator = TGBEvaluator(name=dataset_name)

        # Recency negative generator (generates once, reuses from cache)
        self.recency_gen = RecencyNegativeGenerator(
            dataset_name  = dataset_name,
            first_dst_id  = first_dst_id,
            last_dst_id   = last_dst_id,
            num_neg       = num_neg,
            seed          = seed,
        )

    def run(self, model: BaseModel, model_name: str = "model") -> dict:
        """
        Run both evaluation dimensions on the given model.

        The model must already have load_checkpoint() called before run().
        warmup() is called internally before each evaluation pass so that
        the neighbor_loader state is fresh and reproducible.

        Parameters
        ----------
        model      : BaseModel  — any model implementing the BaseModel interface
        model_name : str        — used in log output

        Returns
        -------
        dict with keys:
            'model'        : str
            'dataset'      : str
            'standard_mrr' : float
            'recency_mrr'  : float
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} on {self.dataset_name}")
        print(f"{'='*60}")

        # ── 1. Standard MRR (TGB hist_rnd negatives) ─────────────────────────
        # TGB's negatives: 50% historical destinations + 50% random pages.
        # Load them from TGB's pre-generated test file.
        print("\n[1/2] Standard MRR (TGB hist_rnd negatives)...")
        self.dataset.load_test_ns()
        # Cap TGB negatives to num_neg so Standard MRR and Recency MRR are
        # evaluated on the same pool size and are directly comparable.
        # TGB tgbl-wiki test provides 999 negatives per edge in py-tgb 2.2+.
        tgb_neg_sampler = _CappedNegSampler(
            self.dataset.negative_sampler, self.num_neg, self.seed
        )
        print(f"  (capped TGB negatives to {self.num_neg} per edge)")

        # Warmup before this eval pass to ensure a clean neighbor_loader state
        model.warmup(self.train_data, self.val_data)
        standard_mrr = model.evaluate(
            self.test_data, tgb_neg_sampler, "test", self.tgb_evaluator
        )
        print(f"  Standard MRR = {standard_mrr:.4f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── 2. Recency MRR (negatives from source's recent history) ──────────
        # Generate negatives (or load from cache if already generated).
        # Adaptive window = per-source median inter-event time.
        print("\n[2/2] Recency MRR (negatives from source's recent history)...")
        recency_path = self.recency_gen.generate(
            historical_data = self.train_data,
            eval_data       = self.test_data,
            split_mode      = "test",
            save_dir        = self.neg_cache_dir,
            window          = None,  # adaptive per-source window
        )

        recency_sampler = NegativeSampler(strategy="recency")
        recency_sampler.load_eval_set(recency_path, split_mode="test")

        # Re-warmup: the neighbor_loader was consumed by the Standard MRR pass,
        # so we reset it to get a clean state for the Recency MRR pass.
        model.warmup(self.train_data, self.val_data)
        recency_mrr = model.evaluate(
            self.test_data, recency_sampler, "test", self.tgb_evaluator
        )
        print(f"  Recency MRR  = {recency_mrr:.4f}")

        # ── Compile results ───────────────────────────────────────────────────
        results = {
            "model":        model_name,
            "dataset":      self.dataset_name,
            "standard_mrr": round(standard_mrr, 4),
            "recency_mrr":  round(recency_mrr, 4),
        }

        print(f"\n{'='*60}")
        print("Results:")
        print(json.dumps(results, indent=2))
        print(f"{'='*60}\n")

        return results
