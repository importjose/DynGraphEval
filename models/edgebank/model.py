"""
EdgeBank: non-parametric baseline for temporal link prediction.

Poursafaei et al., "Towards Better Evaluation for Dynamic Link Prediction",
NeurIPS 2022 Datasets & Benchmarks Track.

EdgeBank∞ (infinite window):
    score(u, v, t) = 1.0 if (u, v) has been seen in train+val history before t
                   = 0.0 otherwise

Memory is initialized from train+val in warmup() and updated with each
positive test edge in order (rolling update). This means the model grows
stronger as it observes more test edges — matching the original paper's
definition of EdgeBank∞.

Why EdgeBank matters for Recency MRR
-------------------------------------
EdgeBank scores 1 for any (src, dst) pair it has seen before and 0 otherwise.
Under Standard MRR the recency negatives are sampled from the source's recent
history — meaning they are almost always in EdgeBank's memory too. So both
the positive and most negatives score 1, causing heavy ties.

TGB tie-handling:
    rank = 0.5 * (1 + count(neg > pos)) + 0.5 * (1 + count(neg >= pos))
         = 0.5 * (1 + 0) + 0.5 * (1 + n_ties)  when pos = 1 and n_ties negatives also = 1
         = (2 + n_ties) / 2

For a Return edge with all K recency negatives also in memory:
    MRR = 2 / (2 + K)   →   for K=999: MRR ≈ 0.002

For an Explore edge (pos = 0):
    Every recency negative seen before also scores 1 → pos always last.
    MRR = 1 / (1 + K)   →   for K=999: MRR ≈ 0.001

So we expect EdgeBank Recency MRR to be very close to 0, confirming that
Recency MRR specifically probes temporal reasoning that memorization alone
cannot provide.
"""

import numpy as np
from tqdm import tqdm
from torch_geometric.data import TemporalData

from ..base import BaseModel


class EdgeBankModel(BaseModel):
    """
    EdgeBank∞ — infinite-window non-parametric baseline.

    Parameters
    ----------
    batch_size : int
        Number of test edges processed per iteration (does not affect results,
        only controls tqdm granularity). Default 200.
    """

    def __init__(self, batch_size: int = 200):
        self.batch_size   = batch_size
        # (src, dst) pairs seen in train+val; updated during test evaluation
        self._seen_pairs: set = set()

    # ── BaseModel interface ────────────────────────────────────────────────────

    def load_checkpoint(self, path=None) -> None:
        """No-op: EdgeBank has no learnable parameters."""
        pass

    def warmup(self, train_data: TemporalData, val_data: TemporalData) -> None:
        """
        Initialize seen-pair memory from train+val history.

        Resets the memory completely each call so repeated evaluation passes
        (Standard pass then Recency pass) start from the same clean state.
        """
        self._seen_pairs = set()
        for tdata in (train_data, val_data):
            src_np = tdata.src.cpu().numpy()
            dst_np = tdata.dst.cpu().numpy()
            for s, d in zip(src_np, dst_np):
                self._seen_pairs.add((int(s), int(d)))

    def evaluate(
        self,
        eval_data: TemporalData,
        neg_sampler,
        split_mode: str,
        evaluator,
    ) -> float:
        """
        Score each test edge and return mean MRR.

        Scoring rule:
            score(u, v) = 1.0  if (u, v) in seen_pairs
                        = 0.0  otherwise

        After scoring each batch, the positive edges in that batch are added
        to seen_pairs (rolling update). This matches EdgeBank∞ semantics:
        the model observes the stream and remembers every edge it has seen.

        Parameters
        ----------
        eval_data   : TemporalData  — test edges
        neg_sampler : object with .query_batch(src, dst, t, split_mode) -> list
        split_mode  : 'val' or 'test'
        evaluator   : TGB evaluator with .eval(input_dict) -> {'mrr': float}

        Returns
        -------
        float : mean MRR across all positive edges that have ≥1 negative
        """
        src_np = eval_data.src.cpu().numpy()
        dst_np = eval_data.dst.cpu().numpy()
        n      = len(src_np)

        perf_list = []
        n_batches = (n + self.batch_size - 1) // self.batch_size

        pbar = tqdm(
            range(0, n, self.batch_size),
            total=n_batches,
            desc="  scoring",
            ncols=100,
            unit="batch",
        )

        for start in pbar:
            end = min(start + self.batch_size, n)

            # TGB neg_sampler expects tensors; slice from eval_data directly
            pos_src = eval_data.src[start:end]
            pos_dst = eval_data.dst[start:end]
            pos_t   = eval_data.t[start:end]

            neg_batch_list = neg_sampler.query_batch(
                pos_src, pos_dst, pos_t, split_mode=split_mode
            )

            for idx, neg_batch in enumerate(neg_batch_list):
                if len(neg_batch) == 0:
                    # Source has no recency history — skip (evaluator never called)
                    continue

                s = int(src_np[start + idx])
                d = int(dst_np[start + idx])

                # Binary scores: 1 if seen, 0 if not
                pos_score  = 1.0 if (s, d) in self._seen_pairs else 0.0
                neg_scores = np.array(
                    [1.0 if (s, int(nd)) in self._seen_pairs else 0.0
                     for nd in neg_batch],
                    dtype=np.float32,
                )

                input_dict = {
                    "y_pred_pos":  np.array([pos_score]),
                    "y_pred_neg":  neg_scores,
                    "eval_metric": ["mrr"],
                }
                perf_list.append(evaluator.eval(input_dict)["mrr"])

            # Rolling update: add positive edges in this batch to memory
            # (done after scoring to avoid leaking the current positive)
            for idx in range(end - start):
                self._seen_pairs.add((int(src_np[start + idx]), int(dst_np[start + idx])))

            running_mrr = float(np.mean(perf_list)) if perf_list else 0.0
            pbar.set_postfix(mrr=f"{running_mrr:.4f}")

        return float(np.mean(perf_list)) if perf_list else 0.0
