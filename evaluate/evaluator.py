"""
Evaluation orchestrator for DynGraphEval.

The Evaluator class runs two evaluation dimensions on any BaseModel:

    1. Standard MRR  — TGB's built-in hist_rnd negatives (all 999).
       Directly comparable with the TGB leaderboard.

    2. Recency MRR   — last-K temporally hard negatives (no random fill).
       For each positive edge (u, v, t), negatives are the most recent K
       unique pages u visited before t (reverse-chronological order).
       A model with no temporal memory cannot distinguish "visiting now"
       from "visited most recently." Not on the same scale as Standard MRR
       — what matters is the relative ordering across models.

Both dimensions are further decomposed along two axes:

    Return edges : (u, v) pair appeared in train or val  → tests recall
    Explore edges: (u, v) pair never seen before          → tests generalization

    Recency K-curve: Recency MRR at K ∈ [10, 20, 50, 100, 500, 999].
        Negatives are pre-ordered most-recent-first at K=999.
        Slicing neg_scores[:k] gives the k most recent negatives for any k,
        so the full K-curve is computed from a single evaluation pass.

Usage
-----
    from evaluate.evaluator import Evaluator
    from models.tgn.model import TPNetTGN

    model = TPNetTGN(checkpoint_path='...', ...)
    model.load_checkpoint()

    ev = Evaluator(dataset, train_data, val_data, test_data,
                   dataset_name='tgbl-wiki', neg_cache_dir='neg_cache')
    results = ev.run(model)
    # results dict contains: standard_mrr, recency_mrr, recency_k_curve,
    # recency_return, recency_explore, standard_mrr_return, standard_mrr_explore
"""

import json
import time
import numpy as np
import torch
from tgb.linkproppred.evaluate import Evaluator as TGBEvaluator
from torch_geometric.data import TemporalData

from models.base import BaseModel
from .negative_sampler import RecencyNegativeGenerator, NegativeSampler


# K values for the Recency MRR temporal reach curve.
# We score once at K=999 and slice neg_scores[:k] for each k in post-processing.
K_VALUES = [10, 20, 50, 100, 500, 999]


# ── Score collection wrappers ─────────────────────────────────────────────────

class _ScoreCollector:
    """
    Wraps a TGB evaluator, intercepting every eval() call to store raw scores.

    The model calls evaluator.eval(input_dict) once per scored positive edge
    (edges with 0 negatives are skipped by the model and never reach eval()).
    This wrapper stores (pos_score, neg_scores) for each call while returning
    the original result unchanged, so the model's evaluate() is unaffected.

    For recency passes, neg_scores[i] is the model score for the i-th most
    recently visited unique destination (most-recent-first order from
    RecencyNegativeGenerator). Slicing neg_scores[:k] gives the k most recent
    negatives, enabling K-curve computation without re-running the model.
    """

    def __init__(self, base_evaluator):
        self._base   = base_evaluator
        self.records = []  # list of {"pos_score": float, "neg_scores": np.ndarray}

    def eval(self, input_dict: dict) -> dict:
        result = self._base.eval(input_dict)
        self.records.append({
            "pos_score":  float(input_dict["y_pred_pos"][0]),
            "neg_scores": np.array(input_dict["y_pred_neg"], dtype=np.float32),
        })
        return result

    def reset(self):
        self.records = []


class _TrackingNegSampler:
    """
    Wraps any neg_sampler and records which (src, dst, t) edges were scored.

    Models call query_batch() once per batch, then iterate over the returned
    list and skip entries with empty neg lists (no evaluator.eval() call for
    those edges). By recording only the non-empty entries in the same iteration
    order, tracked_edges[i] corresponds exactly to _ScoreCollector.records[i].

    This alignment allows correlating per-edge score records with Return/Explore
    labels after the evaluation pass completes.
    """

    def __init__(self, base_sampler):
        self._base        = base_sampler
        self.scored_edges = []  # list of (src, dst, t) in PyG 0-indexed convention

    def query_batch(self, pos_src, pos_dst, pos_t, split_mode: str) -> list:
        neg_lists = self._base.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)
        # Convert tensors to numpy for dict-key lookups (0-indexed, PyG convention)
        src_np = pos_src.detach().cpu().numpy() if isinstance(pos_src, torch.Tensor) else np.asarray(pos_src)
        dst_np = pos_dst.detach().cpu().numpy() if isinstance(pos_dst, torch.Tensor) else np.asarray(pos_dst)
        t_np   = pos_t.detach().cpu().numpy()   if isinstance(pos_t,   torch.Tensor) else np.asarray(pos_t)
        for i, neg_list in enumerate(neg_lists):
            if len(neg_list) > 0:
                self.scored_edges.append((int(src_np[i]), int(dst_np[i]), int(t_np[i])))
        return neg_lists

    def reset(self):
        self.scored_edges = []


class _PassthroughNegSampler:
    """Passes TGB negatives through unchanged (all 999)."""

    def __init__(self, sampler):
        self._sampler = sampler

    def query_batch(self, pos_src, pos_dst, pos_t, split_mode):
        return self._sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_mrr(records: list, k: int = None) -> float:
    """
    Compute mean reciprocal rank from a list of per-edge score records.

    Uses the same tie-handling formula as TGB:
        optimistic_rank  = 1 + count(neg > pos)
        pessimistic_rank = 1 + count(neg >= pos)
        rank = 0.5 * (optimistic_rank + pessimistic_rank)
        mrr  = 1 / rank

    Parameters
    ----------
    records : list of {"pos_score": float, "neg_scores": np.ndarray}
              For recency passes, neg_scores are ordered most-recent-first.
    k       : int or None
              If given, use only the first k negatives per edge (the k most
              recent visits). Edges with fewer than k negatives contribute
              with their actual count (no padding). If None, use all.

    Returns
    -------
    float : mean reciprocal rank (0.0 if no scoreable records)
    """
    mrr_values = []
    for r in records:
        pos = r["pos_score"]
        neg = r["neg_scores"][:k] if k is not None else r["neg_scores"]
        if len(neg) == 0:
            continue
        opt_rank  = int((neg > pos).sum()) + 1
        pess_rank = int((neg >= pos).sum()) + 1
        rank      = 0.5 * (opt_rank + pess_rank)
        mrr_values.append(1.0 / rank)
    return float(np.mean(mrr_values)) if mrr_values else 0.0


def _build_historical_pairs(train_data: TemporalData, val_data: TemporalData) -> set:
    """
    Return the set of (src, dst) pairs that appear in train or val data.

    A test edge is labeled **Return** if its (src, dst) pair is in this set —
    meaning the source node has visited this destination at least once before.
    **Explore** edges are those where the source has never visited the destination.

    Node IDs use PyG 0-indexed convention.
    """
    pairs = set()
    for tdata in (train_data, val_data):
        src_np = tdata.src.cpu().numpy()
        dst_np = tdata.dst.cpu().numpy()
        for s, d in zip(src_np, dst_np):
            pairs.add((int(s), int(d)))
    return pairs


def _k_curve(records: list) -> dict:
    """Compute Recency MRR at each K in K_VALUES. Returns {"k10": ..., "k20": ..., ...}."""
    return {f"k{k}": round(_compute_mrr(records, k), 4) for k in K_VALUES}


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs Standard MRR and Recency MRR on any BaseModel, with Return/Explore
    split and Recency K-curve.

    Parameters
    ----------
    dataset       : PyGLinkPropPredDataset  — TGB dataset object (provides neg sampler)
    train_data    : TemporalData
    val_data      : TemporalData
    test_data     : TemporalData
    dataset_name  : str   used for logging and filenames (default 'tgbl-wiki')
    neg_cache_dir : str   directory to cache generated negatives (default 'neg_cache')
    num_neg       : int   negatives per positive edge for recency pass (default 999)
    seed          : int   random seed for negative generation (default 42)
    first_dst_id  : int   (unused, kept for backwards compat)
    last_dst_id   : int   (unused, kept for backwards compat)
    """

    def __init__(
        self,
        dataset,
        train_data,
        val_data,
        test_data,
        dataset_name: str = "tgbl-wiki",
        neg_cache_dir: str = "neg_cache",
        num_neg: int = 999,
        seed: int = 42,
        first_dst_id: int = None,
        last_dst_id: int = None,
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
            dataset_name=dataset_name,
            num_neg=num_neg,
            seed=seed,
        )

    def run(
        self,
        model: BaseModel,
        model_name: str  = "model",
        skip_standard: bool  = False,
        standard_mrr: float  = None,
    ) -> dict:
        """
        Run both evaluation dimensions and return the full results dict.

        The model must already have load_checkpoint() called before run().
        warmup() is called internally before each evaluation pass so that
        the internal state is fresh and reproducible.

        Parameters
        ----------
        model         : BaseModel  — any model implementing the BaseModel interface
        model_name    : str        — used in log output and results dict
        skip_standard : bool       — if True, skip Standard MRR pass (use for
                                     memory-free models whose training Test MRR
                                     equals Standard MRR)
        standard_mrr  : float      — pre-computed Standard MRR to record when
                                     skip_standard=True

        Returns
        -------
        dict with keys:
            standard_mrr          : float
            standard_mrr_return   : float or None (None when skip_standard=True)
            standard_mrr_explore  : float or None
            recency_mrr           : float  (all negatives, backward compat scalar)
            recency_k_curve       : dict   {"k10": ..., "k20": ..., ..., "k999": ...}
            recency_return        : dict   {"n": int, "mrr": float, "k10": ..., ...}
            recency_explore       : dict   {"n": int, "mrr": float, "k10": ..., ...}
            n_scored              : int    test edges with ≥1 recency negative
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} on {self.dataset_name}")
        print(f"{'='*60}")
        print(f"  test edges        : {len(self.test_data.src):,}")
        print(f"  recency neg/edge  : last-{self.num_neg} unique visits (most-recent-first)")
        print(f"  K-curve values    : {K_VALUES}")

        # ── Build Return/Explore labels ───────────────────────────────────────
        # Do this once before any pass since it only depends on train+val history.
        print("\n[0/2] Building Return/Explore labels from train+val history...")
        t_hist = time.time()
        hist_pairs = _build_historical_pairs(self.train_data, self.val_data)
        print(f"  {len(hist_pairs):,} unique (src, dst) pairs  ({time.time()-t_hist:.0f}s)")

        std_collector = None
        std_tracker   = None

        # ── 1. Standard MRR ───────────────────────────────────────────────────
        if skip_standard:
            print(f"\n[1/2] Standard MRR: skipped (using training Test MRR = {standard_mrr})")
        else:
            print("\n[1/2] Standard MRR (TGB hist_rnd negatives, 999/edge)...")
            self.dataset.load_test_ns()

            # Wrap both the neg sampler and TGB evaluator so we can capture per-edge
            # scores for the Return/Explore split without re-running the model.
            passthrough   = _PassthroughNegSampler(self.dataset.negative_sampler)
            std_tracker   = _TrackingNegSampler(passthrough)
            std_collector = _ScoreCollector(self.tgb_evaluator)

            t0 = time.time()
            print("  warming up memory bank (train + val replay)...")
            model.warmup(self.train_data, self.val_data)
            print(f"  warmup done in {time.time()-t0:.0f}s")

            t1 = time.time()
            standard_mrr = model.evaluate(
                self.test_data, std_tracker, "test", std_collector
            )
            print(f"  Standard MRR = {standard_mrr:.4f}  ({time.time()-t1:.0f}s)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── 2. Recency MRR (single pass at K=999) ─────────────────────────────
        # All K-curve values are derived from this single pass by slicing the
        # stored neg_scores arrays. No re-running the model needed.
        print(f"\n[2/2] Recency MRR (last-{self.num_neg} unique visits, K-curve + Return/Explore)...")
        t2 = time.time()
        recency_path = self.recency_gen.generate(
            historical_data=self.train_data,
            eval_data=self.test_data,
            split_mode="test",
            save_dir=self.neg_cache_dir,
        )
        print(f"  negatives ready in {time.time()-t2:.0f}s  ({recency_path})")

        recency_sampler = NegativeSampler(strategy="recency")
        recency_sampler.load_eval_set(recency_path, split_mode="test")

        rec_tracker   = _TrackingNegSampler(recency_sampler)
        rec_collector = _ScoreCollector(self.tgb_evaluator)

        print("  warming up memory bank (train + val replay)...")
        t3 = time.time()
        model.warmup(self.train_data, self.val_data)
        print(f"  warmup done in {time.time()-t3:.0f}s")

        t4 = time.time()
        recency_mrr = model.evaluate(
            self.test_data, rec_tracker, "test", rec_collector
        )
        print(f"  Recency MRR  = {recency_mrr:.4f}  ({time.time()-t4:.0f}s)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Post-process: K-curve × Return/Explore ────────────────────────────
        # rec_tracker.scored_edges[i] ↔ rec_collector.records[i] (same edge, same pass order)
        is_return = [(s, d) in hist_pairs for s, d, t in rec_tracker.scored_edges]
        ret_records = [r for r, flag in zip(rec_collector.records, is_return) if     flag]
        exp_records = [r for r, flag in zip(rec_collector.records, is_return) if not flag]

        n_return  = sum(is_return)
        n_explore = len(is_return) - n_return

        print(f"\n  Scored edges  : {len(rec_collector.records):,}  "
              f"(Return: {n_return:,}, Explore: {n_explore:,})")

        # K-curve for all / return / explore subsets
        rec_k_curve         = _k_curve(rec_collector.records)
        rec_k_curve_return  = _k_curve(ret_records)
        rec_k_curve_explore = _k_curve(exp_records)

        print(f"  Recency K-curve (all)    : {rec_k_curve}")
        print(f"  Recency K-curve (return) : {rec_k_curve_return}")
        print(f"  Recency K-curve (explore): {rec_k_curve_explore}")

        # Standard MRR Return/Explore split (available when skip_standard=False)
        if std_collector is not None:
            std_is_return = [(s, d) in hist_pairs for s, d, t in std_tracker.scored_edges]
            std_ret = [r for r, flag in zip(std_collector.records, std_is_return) if     flag]
            std_exp = [r for r, flag in zip(std_collector.records, std_is_return) if not flag]
            std_return_mrr  = round(_compute_mrr(std_ret), 4)
            std_explore_mrr = round(_compute_mrr(std_exp), 4)
            n_std_return    = sum(std_is_return)
            n_std_explore   = len(std_is_return) - n_std_return
            print(f"  Standard MRR Return ({n_std_return:,} edges): {std_return_mrr}")
            print(f"  Standard MRR Explore ({n_std_explore:,} edges): {std_explore_mrr}")
        else:
            std_return_mrr  = None
            std_explore_mrr = None

        # ── Compile results ───────────────────────────────────────────────────
        results = {
            "model":   model_name,
            "dataset": self.dataset_name,
            # Standard MRR (overall + Return/Explore split)
            "standard_mrr":         round(standard_mrr, 4),
            "standard_mrr_return":  std_return_mrr,
            "standard_mrr_explore": std_explore_mrr,
            # Recency MRR — overall scalar (backward-compatible key)
            "recency_mrr": round(recency_mrr, 4),
            # Recency K-curve over all scored test edges
            "recency_k_curve": rec_k_curve,
            # Recency Return edges (src visited dst before in train+val)
            "recency_return": {
                "n":   n_return,
                "mrr": round(_compute_mrr(ret_records), 4),
                **rec_k_curve_return,
            },
            # Recency Explore edges (src never visited dst in train+val)
            "recency_explore": {
                "n":   n_explore,
                "mrr": round(_compute_mrr(exp_records), 4),
                **rec_k_curve_explore,
            },
            # Edges with at least one recency negative (edges with 0 history are skipped)
            "n_scored": len(rec_collector.records),
        }

        print(f"\n{'='*60}")
        print("Results:")
        print(json.dumps(results, indent=2))
        print(f"{'='*60}\n")

        return results
