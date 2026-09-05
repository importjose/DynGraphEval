"""
Federated TGN model wrapper (4 clients, source-node partition).

Federated setup:
    - Data is split by source node ID: each client owns edges where
      src ∈ their assigned partition (sorted node IDs split equally).
    - Each client has an independent TGN with its own memory and neighbor_loader.
    - The GNN and LinkPredictor weights are shared via FedAvg during training,
      but memory buffers remain local (they encode private interaction history).

Evaluation:
    - Each client evaluates on its local test partition independently.
    - Global MRR = weighted average across clients, weighted by edge count.
      (Clients with more test edges contribute proportionally more to the score.)

Memory fragmentation is the key limitation:
    - Page (destination) nodes appear in multiple clients' graphs.
    - Each client's page memory only reflects that client's interactions.
    - This causes information loss compared to the centralized model.
"""

import numpy as np
import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from ..base import BaseModel
from evaluate.partition import build_src_sets, partition_by_source
from ..components import (
    TGNMemory,
    GraphAttentionEmbedding,
    LinkPredictor,
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)


class _TGNClient:
    """
    One federated TGN client — evaluation-only (no optimizer, no training).

    Each client holds:
        - An independent TGN (memory + gnn + link_pred)
        - A LastNeighborLoader for its local neighborhood
        - Local edge tensors (local_all_t, local_all_msg) for e_id lookups

    The local_all_t/msg tensors concatenate train+val+test edge timestamps
    and features for this client. When the neighbor_loader returns e_ids,
    they index into these local tensors (not the global dataset tensors).

    This mirrors how e_ids are tracked during training: train edges get IDs
    [0, num_train), val edges get [num_train, num_train+num_val), etc.
    """

    def __init__(
        self,
        client_id: int,
        num_nodes: int,
        msg_dim: int,
        device: torch.device,
        train_t,
        train_msg,
        val_t,
        val_msg,
        test_t,
        test_msg,
        mem_dim: int = 100,
        time_dim: int = 100,
        emb_dim: int = 100,
        num_neighbors: int = 10,
        batch_size: int = 200,
    ):
        self.client_id  = client_id
        self.device     = device
        self.batch_size = batch_size

        # ── Build TGN components ──────────────────────────────────────────────

        msg_module = IdentityMessage(msg_dim, mem_dim, time_dim)

        self.memory = TGNMemory(
            num_nodes,
            msg_dim,
            mem_dim,
            time_dim,
            message_module    = msg_module,
            aggregator_module = LastAggregator(),
        ).to(device)
        self.memory.reset_state()

        self.gnn = GraphAttentionEmbedding(
            in_channels  = mem_dim,
            out_channels = emb_dim,
            msg_dim      = msg_dim,
            time_enc     = self.memory.time_enc,
        ).to(device)

        self.link_pred = LinkPredictor(in_channels=emb_dim).to(device)

        self.neighbor_loader = LastNeighborLoader(
            num_nodes, size=num_neighbors, device=device
        )
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

        # Local edge tensors: concatenate train+val+test so e_ids are consistent
        # with how they were assigned during training.
        self.local_all_t   = torch.cat([train_t,   val_t,   test_t]).to(device)
        self.local_all_msg = torch.cat([train_msg, val_msg, test_msg]).to(device)

    def load_checkpoint(self, path: str) -> None:
        """
        Load checkpoint for this client.

        Supports two checkpoint formats:
            - {'memory': state_dict, 'gnn': ..., 'link_pred': ...}
              (saved directly from the local model)
            - {'memory_params': state_dict, 'gnn': ..., 'link_pred': ...}
              (federated format: only learnable params, no memory buffers)
        """
        ckpt = torch.load(path, map_location=self.device)

        if "memory" in ckpt:
            self.memory.load_state_dict(ckpt["memory"])
        elif "memory_params" in ckpt:
            # Partial load: update only the params (GRU weights, time encoder)
            # but keep the current memory buffer values (local interaction history)
            current_sd = self.memory.state_dict()
            current_sd.update(ckpt["memory_params"])
            self.memory.load_state_dict(current_sd)
        else:
            raise ValueError(
                f"Unknown checkpoint format. Found keys: {list(ckpt.keys())}. "
                "Expected 'memory' or 'memory_params'."
            )

        self.gnn.load_state_dict(ckpt["gnn"])
        self.link_pred.load_state_dict(ckpt["link_pred"])
        # Reset only message stores — the memory buffer contains trained node
        # states from the checkpoint and must be preserved, not zeroed.
        self.memory._reset_message_store()
        # Switch to eval mode via the base class, bypassing TGNMemory.train()'s
        # override. That override runs GRU(zeros, memory) for all nodes when
        # message stores are empty, which would corrupt the loaded checkpoint values.
        torch.nn.Module.train(self.memory, False)

    def warmup_neighbors(
        self, train_data: TemporalData, val_data: TemporalData
    ) -> None:
        """Replay this client's train+val edges into its local neighbor_loader."""
        self.neighbor_loader.reset_state()
        for tdata in [train_data, val_data]:
            loader = TemporalDataLoader(tdata, batch_size=self.batch_size)
            for batch in loader:
                self.neighbor_loader.insert(
                    batch.src.to(self.device),
                    batch.dst.to(self.device),
                )

    @torch.no_grad()
    def evaluate(
        self, eval_data: TemporalData, neg_sampler, split_mode: str, evaluator
    ) -> float:
        """Run evaluation on this client's test partition. Returns MRR."""
        self.memory.eval()
        self.gnn.eval()
        self.link_pred.eval()

        loader    = TemporalDataLoader(eval_data, batch_size=self.batch_size)
        perf_list = []

        for pos_batch in loader:
            pos_src = pos_batch.src.to(self.device)
            pos_dst = pos_batch.dst.to(self.device)
            pos_t   = pos_batch.t.to(self.device)
            pos_msg = pos_batch.msg.to(self.device)

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

                n_id, edge_index, e_id = self.neighbor_loader(
                    torch.cat([src, dst]).unique()
                )
                self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

                z, last_update = self.memory(n_id)

                # Use local edge tensors (not global) — e_ids index into local concat
                z = self.gnn(
                    z, last_update, edge_index,
                    self.local_all_t[e_id],
                    self.local_all_msg[e_id],
                )

                y_pred = self.link_pred(z[self.assoc[src]], z[self.assoc[dst]])

                input_dict = {
                    "y_pred_pos":  np.array([y_pred[0].squeeze().cpu()]),
                    "y_pred_neg":  np.array(y_pred[1:].squeeze(-1).cpu()),
                    "eval_metric": ["mrr"],
                }
                perf_list.append(evaluator.eval(input_dict)["mrr"])

            self.memory.update_state(pos_src, pos_dst, pos_t, pos_msg)
            self.neighbor_loader.insert(pos_src, pos_dst)

        return float(np.mean(perf_list)) if perf_list else 0.0


class FederatedTGN(BaseModel):
    """
    Federated TGN: 4 independent TGN clients with source-node partitioning.

    Parameters
    ----------
    checkpoint_paths : list[str]  one .pt checkpoint per client (length = num_clients)
    num_nodes        : int
    msg_dim          : int   edge feature dimension
    train_data       : TemporalData  used for partitioning and neighbor warmup
    val_data         : TemporalData  used for partitioning and neighbor warmup
    test_data        : TemporalData  used for partitioning and local_all_t
    device           : torch.device
    num_clients      : int  (default 4)
    mem_dim          : int  (default 100)
    time_dim         : int  (default 100)
    emb_dim          : int  (default 100)
    num_neighbors    : int  (default 10)
    batch_size       : int  (default 200)
    """

    def __init__(
        self,
        checkpoint_paths: list,
        num_nodes: int,
        msg_dim: int,
        train_data: TemporalData,
        val_data: TemporalData,
        test_data: TemporalData,
        device: torch.device,
        num_clients: int = 4,
        mem_dim: int = 100,
        time_dim: int = 100,
        emb_dim: int = 100,
        num_neighbors: int = 10,
        batch_size: int = 200,
    ):
        assert len(checkpoint_paths) == num_clients, (
            f"Expected {num_clients} checkpoint paths, got {len(checkpoint_paths)}"
        )

        self.device      = device
        self.num_clients = num_clients
        self.ckpt_paths  = checkpoint_paths

        # ── Partition data by source node ─────────────────────────────────────
        # Source node IDs are sorted and split equally across clients.
        # This must match exactly what was done during training.
        src_sets = build_src_sets(train_data, num_clients)

        self.client_train = partition_by_source(train_data, src_sets)
        self.client_val   = partition_by_source(val_data,   src_sets)
        self.client_test  = partition_by_source(test_data,  src_sets)

        # ── Instantiate one TGN client per partition ──────────────────────────
        self.clients = []
        for c in range(num_clients):
            tr, va, te = self.client_train[c], self.client_val[c], self.client_test[c]
            client = _TGNClient(
                client_id     = c,
                num_nodes     = num_nodes,
                msg_dim       = msg_dim,
                device        = device,
                train_t       = tr.t,   train_msg = tr.msg,
                val_t         = va.t,   val_msg   = va.msg,
                test_t        = te.t,   test_msg  = te.msg,
                mem_dim       = mem_dim,
                time_dim      = time_dim,
                emb_dim       = emb_dim,
                num_neighbors = num_neighbors,
                batch_size    = batch_size,
            )
            self.clients.append(client)

    # ── BaseModel interface ───────────────────────────────────────────────────

    def load_checkpoint(self, path=None) -> None:
        """Load one checkpoint per client."""
        paths = path if path is not None else self.ckpt_paths
        for c, client in enumerate(self.clients):
            client.load_checkpoint(paths[c])

    def warmup(self, train_data=None, val_data=None) -> None:
        """
        Warmup each client's neighbor_loader with its local train+val edges.

        train_data/val_data arguments are ignored here — each client uses
        its own pre-partitioned data stored at initialization time.
        """
        for c, client in enumerate(self.clients):
            client.warmup_neighbors(self.client_train[c], self.client_val[c])

    @torch.no_grad()
    def evaluate(
        self, eval_data: TemporalData, neg_sampler, split_mode: str, evaluator
    ) -> float:
        """
        Evaluate each client on its local test partition.

        eval_data is intentionally ignored — each client uses its own
        pre-partitioned test data that was built at initialization time.

        Returns weighted-average MRR, where each client's contribution
        is proportional to the number of test edges it evaluates on.
        """
        mrrs, sizes = [], []
        for c, client in enumerate(self.clients):
            n_test = self.client_test[c].num_events
            if n_test == 0:
                continue  # skip clients with no test edges

            mrr = client.evaluate(
                self.client_test[c], neg_sampler, split_mode, evaluator
            )
            mrrs.append(mrr)
            sizes.append(n_test)
            print(f"  Client {c}: MRR={mrr:.4f}  (n={n_test})")

        # Weighted average: clients with more edges influence the global MRR more
        total = sum(sizes)
        return sum(m * s / total for m, s in zip(mrrs, sizes))
