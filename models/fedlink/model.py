"""
FedLink model wrapper (4 clients, static GraphSAGE, no temporal reasoning).

FedLink is a static graph learning baseline. It:
    - Discards all timestamps — the model never sees time
    - Builds a static undirected edge_index from training edges
    - Learns separate embeddings for users and pages (heterogeneous graph)
    - Uses 2-layer GraphSAGE to aggregate neighborhood structure
    - Scores links via dot product of node embeddings

This serves as a lower-bound comparison for TGN: if TGN significantly
outperforms FedLink, temporal reasoning is contributing meaningful signal
beyond pure graph structure.

The same source-node partitioning is used as in FederatedTGN so that
data splits are directly comparable across models.

warmup() is a no-op because FedLink has no stateful memory to rebuild.
"""

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from ..base import BaseModel
from evaluate.partition import build_src_sets, partition_by_source
from ..components import FedLinkModel


def _build_undirected_edge_index(tdata: TemporalData, device: torch.device) -> Tensor:
    """
    Convert temporal edges to a static undirected edge_index.

    Timestamps are discarded — only the (src, dst) pairs are kept.
    Both directions are included (src→dst and dst→src) to allow
    GraphSAGE to aggregate from both sides of each edge.
    """
    src = tdata.src.to(device)
    dst = tdata.dst.to(device)
    return torch.stack([
        torch.cat([src, dst]),  # forward + backward sources
        torch.cat([dst, src]),  # forward + backward destinations
    ], dim=0)


class FedLink(BaseModel):
    """
    FedLink: static GraphSAGE baseline with federated source-node partitioning.

    Parameters
    ----------
    checkpoint_paths : list[str]  one per client; each is a flat state_dict
    num_users        : int   number of user (source) nodes
    num_pages        : int   number of page (destination) nodes
    train_data       : TemporalData
    val_data         : TemporalData
    test_data        : TemporalData
    min_dst_idx      : int   smallest destination node ID (for offset handling)
    device           : torch.device
    num_clients      : int   (default 4)
    hidden_dim       : int   embedding and GNN hidden dimension (default 64)
    batch_size       : int   (default 200)
    """

    def __init__(
        self,
        checkpoint_paths: list,
        num_users: int,
        num_pages: int,
        train_data: TemporalData,
        val_data: TemporalData,
        test_data: TemporalData,
        min_dst_idx: int,
        device: torch.device,
        num_clients: int = 4,
        hidden_dim: int = 64,
        batch_size: int = 200,
    ):
        assert len(checkpoint_paths) == num_clients, (
            f"Expected {num_clients} checkpoint paths, got {len(checkpoint_paths)}"
        )

        self.device      = device
        self.num_clients = num_clients
        self.ckpt_paths  = checkpoint_paths
        self.batch_size  = batch_size
        self.num_users   = num_users
        self.num_pages   = num_pages
        self.min_dst_idx = min_dst_idx  # offset to convert global dst ID → page emb index

        # ── Partition data by source node (same as FederatedTGN) ──────────────
        src_sets = build_src_sets(train_data, num_clients)

        self.client_train = partition_by_source(train_data, src_sets)
        self.client_val   = partition_by_source(val_data,   src_sets)
        self.client_test  = partition_by_source(test_data,  src_sets)

        # ── Build static edge indices per client ──────────────────────────────

        # Train-only edge_index: used as the "local graph" during training
        self.client_train_ei = [
            _build_undirected_edge_index(self.client_train[c], device)
            for c in range(num_clients)
        ]

        # Train+val edge_index: used for test evaluation (richer context)
        # At test time, all training history is available, so we include val edges.
        self.client_tv_ei = []
        for c in range(num_clients):
            tv_src = torch.cat([
                self.client_train[c].src, self.client_val[c].src
            ]).to(device)
            tv_dst = torch.cat([
                self.client_train[c].dst, self.client_val[c].dst
            ]).to(device)
            ei = torch.stack([
                torch.cat([tv_src, tv_dst]),
                torch.cat([tv_dst, tv_src]),
            ], dim=0)
            self.client_tv_ei.append(ei)

        # ── Instantiate one FedLinkModel per client ───────────────────────────
        self.models = [
            FedLinkModel(num_users, num_pages, hidden_dim).to(device)
            for _ in range(num_clients)
        ]

    # ── BaseModel interface ───────────────────────────────────────────────────

    def load_checkpoint(self, path=None) -> None:
        """Load one state_dict per client."""
        paths = path if path is not None else self.ckpt_paths
        for c, model in enumerate(self.models):
            sd = torch.load(paths[c], map_location=self.device)
            model.load_state_dict(sd)

    def warmup(self, train_data=None, val_data=None) -> None:
        """
        No-op — FedLink has no stateful memory to rebuild.

        This method exists only to satisfy the BaseModel interface.
        Static embeddings are recomputed fresh at every evaluate() call.
        """
        pass

    @torch.no_grad()
    def evaluate(
        self, eval_data: TemporalData, neg_sampler, split_mode: str, evaluator
    ) -> float:
        """
        Evaluate each client on its local test partition.

        For each client:
            1. Compute static embeddings from train+val edge_index (one forward pass)
            2. Score all (src, dst) candidate pairs via dot product
            3. Compute MRR using the TGB evaluator

        eval_data is ignored — clients use their own partitioned test data.
        Returns weighted-average MRR by edge count.
        """
        mrrs, sizes = [], []
        for c in range(self.num_clients):
            n_test = self.client_test[c].num_events
            if n_test == 0:
                continue  # skip empty partitions

            mrr = self._evaluate_client(c, split_mode, evaluator, neg_sampler)
            mrrs.append(mrr)
            sizes.append(n_test)
            print(f"  Client {c}: MRR={mrr:.4f}  (n={n_test})")

        total = sum(sizes)
        return sum(m * s / total for m, s in zip(mrrs, sizes))

    def _evaluate_client(
        self, c: int, split_mode: str, evaluator, neg_sampler
    ) -> float:
        """
        Run evaluation for a single client.

        Static embeddings are computed once per client by passing the
        train+val edge_index through the GNN. These same embeddings are
        then used to score all test edges for this client.
        """
        model = self.models[c]
        model.eval()

        # One forward pass to get static node embeddings for all nodes
        # (timestamps are never involved — this is purely structural)
        z = model(self.client_tv_ei[c])

        loader    = TemporalDataLoader(self.client_test[c], batch_size=self.batch_size)
        perf_list = []

        for pos_batch in loader:
            pos_src = pos_batch.src.to(self.device)
            pos_dst = pos_batch.dst.to(self.device)
            pos_t   = pos_batch.t.to(self.device)

            neg_batch_list = neg_sampler.query_batch(
                pos_src, pos_dst, pos_t, split_mode=split_mode
            )

            for idx, neg_batch in enumerate(neg_batch_list):
                src = torch.full(
                    (1 + len(neg_batch),), pos_src[idx].item(), device=self.device
                )
                dst = torch.tensor(
                    [pos_dst[idx].item()] + list(neg_batch), device=self.device
                )

                # Dot product scores for all candidate pairs, then sigmoid to [0,1]
                scores = model.predict(z, src, dst).sigmoid()

                input_dict = {
                    "y_pred_pos":  np.array([scores[0].cpu().item()]),
                    "y_pred_neg":  np.array(scores[1:].cpu().numpy()),
                    "eval_metric": ["mrr"],
                }
                perf_list.append(evaluator.eval(input_dict)["mrr"])

        return float(np.mean(perf_list)) if perf_list else 0.0
