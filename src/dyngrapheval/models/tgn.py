"""
Centralized TGN model wrapper.

The Temporal Graph Network (TGN) maintains a per-node memory vector that is
updated via GRU after every interaction. At evaluation time, the model:
    1. Queries the LastNeighborLoader for a local temporal subgraph
    2. Reads memory states for involved nodes
    3. Runs TransformerConv with temporal edge attributes
    4. Scores (src, dst) pairs with a link predictor MLP

Key design choices:
    - neighbor_loader must be warmed up (via warmup()) before evaluation.
      Without this, the loader is empty and the GNN has no neighborhood context.
    - all_t and all_msg are global edge tensors indexed by e_id (edge ID).
      The loader returns e_ids for the subgraph; we use them to look up
      the corresponding timestamps and features for the GNN.
"""

import numpy as np
import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from ..base_model import BaseModel
from .components import (
    TGNMemory,
    GraphAttentionEmbedding,
    LinkPredictor,
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)


class TGN(BaseModel):
    """
    Centralized Temporal Graph Network for link prediction evaluation.

    Architecture:
        TGNMemory (GRU-based per-node state)
        → GraphAttentionEmbedding (TransformerConv with temporal edges)
        → LinkPredictor (MLP scorer)

    Parameters
    ----------
    checkpoint_path : str   path to .pt file with keys 'memory', 'gnn', 'link_pred'
    num_nodes       : int   total number of nodes in the graph
    msg_dim         : int   edge feature dimension (e.g., 172 for tgbl-wiki)
    all_t           : Tensor  all edge timestamps in the dataset (shape: num_edges)
    all_msg         : Tensor  all edge features in the dataset (shape: num_edges × msg_dim)
    device          : torch.device
    mem_dim         : int   per-node memory dimension (default 100)
    time_dim        : int   time encoding dimension (default 100)
    emb_dim         : int   GNN output embedding dimension (default 100)
    num_neighbors   : int   number of temporal neighbors to store per node (default 10)
    batch_size      : int   edges per evaluation batch (default 200)
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_nodes: int,
        msg_dim: int,
        all_t: torch.Tensor,
        all_msg: torch.Tensor,
        device: torch.device,
        mem_dim: int = 100,
        time_dim: int = 100,
        emb_dim: int = 100,
        num_neighbors: int = 10,
        batch_size: int = 200,
    ):
        self.device       = device
        self.batch_size   = batch_size
        self.ckpt_path    = checkpoint_path

        # Global edge tensors: indexed by e_id returned from neighbor_loader.
        # The neighbor loader stores edge IDs, and we use them here to retrieve
        # the actual timestamps and features needed for the GNN.
        self.all_t   = all_t.to(device)
        self.all_msg = all_msg.to(device)

        # ── Build model components ────────────────────────────────────────────

        # Message function: concat(src_mem, dst_mem, edge_feat, time_enc)
        msg_module = IdentityMessage(msg_dim, mem_dim, time_dim)

        # Memory module: GRU-based per-node state
        self.memory = TGNMemory(
            num_nodes,
            msg_dim,
            mem_dim,
            time_dim,
            message_module    = msg_module,
            aggregator_module = LastAggregator(),
        ).to(device)
        self.memory.reset_state()

        # GNN: TransformerConv with temporal edge attributes (shares time_enc with memory)
        self.gnn = GraphAttentionEmbedding(
            in_channels  = mem_dim,
            out_channels = emb_dim,
            msg_dim      = msg_dim,
            time_enc     = self.memory.time_enc,  # shared time encoder
        ).to(device)

        # Link scorer: MLP that takes (src_emb, dst_emb) and outputs a score
        self.link_pred = LinkPredictor(in_channels=emb_dim).to(device)

        # Neighbor loader: ring buffer of the 10 most recent neighbors per node.
        # Populated during warmup(), updated after each evaluation batch.
        self.neighbor_loader = LastNeighborLoader(
            num_nodes, size=num_neighbors, device=device
        )

        # assoc[n_id[i]] = i  — maps global node ID to local index in current subgraph
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    # ── BaseModel interface ───────────────────────────────────────────────────

    def load_checkpoint(self, path: str = None) -> None:
        """
        Load model weights from a checkpoint.

        Checkpoint must be a dict with keys:
            'memory'    — TGNMemory state_dict
            'gnn'       — GraphAttentionEmbedding state_dict
            'link_pred' — LinkPredictor state_dict

        After loading, memory state is reset so the message stores are
        properly initialized on the current device.
        """
        path = path or self.ckpt_path
        ckpt = torch.load(path, map_location=self.device)
        self.memory.load_state_dict(ckpt["memory"])
        self.gnn.load_state_dict(ckpt["gnn"])
        self.link_pred.load_state_dict(ckpt["link_pred"])
        # Reset only message stores — the memory buffer contains trained node
        # states from the checkpoint and must be preserved, not zeroed.
        self.memory._reset_message_store()
        # Switch to eval mode via the base class, bypassing TGNMemory.train()'s
        # override. That override runs GRU(zeros, memory) for all nodes when
        # message stores are empty, which would corrupt the loaded checkpoint values.
        torch.nn.Module.train(self.memory, False)

    def warmup(self, train_data: TemporalData, val_data: TemporalData) -> None:
        """
        Replay train+val edges into the neighbor_loader.

        This gives the GNN access to a rich temporal neighborhood at test time.
        Without warmup, the loader is empty and the GNN has no context.

        We only insert edges (not update memory) because the memory state
        is already populated in the checkpoint from training.
        """
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
        """
        Evaluate on eval_data and return mean MRR.

        For each positive edge (u, v, t):
            1. Query neg_sampler for a list of negative destinations
            2. Build candidate set: [v] + negatives
            3. Look up the temporal subgraph for all candidate nodes
            4. Run memory → GNN → link predictor
            5. Rank v against negatives using TGB evaluator

        After scoring each batch, memory and neighbor_loader are updated
        so future batches see the correct state (continuous evaluation).

        Returns
        -------
        float : mean MRR across all positive edges
        """
        # Set all components to eval mode
        # (TGNMemory.train() -> .eval() finalizes memory from message store)
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

            # Query negatives for this batch (one list of neg_dsts per positive edge)
            neg_batch_list = neg_sampler.query_batch(
                pos_src, pos_dst, pos_t, split_mode=split_mode
            )

            # Score each positive edge against its negatives
            for idx, neg_batch in enumerate(neg_batch_list):
                # Candidate source: repeated for (1 positive + N negative) dsts
                src = torch.full(
                    (1 + len(neg_batch),), pos_src[idx].item(), device=self.device
                )
                # Candidate destinations: [true_dst, neg_dst_1, neg_dst_2, ...]
                dst = torch.tensor(
                    [pos_dst[idx].item()] + list(neg_batch), device=self.device
                )

                # Get temporal subgraph: local edge indices and edge IDs
                n_id, edge_index, e_id = self.neighbor_loader(
                    torch.cat([src, dst]).unique()
                )
                # Map global node IDs to local indices in this subgraph
                self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

                # Get memory and last_update for subgraph nodes
                z, last_update = self.memory(n_id)

                # Run GNN with temporal edge attributes (timestamps + features)
                z = self.gnn(
                    z, last_update, edge_index,
                    self.all_t[e_id],   # edge timestamps from global tensor
                    self.all_msg[e_id], # edge features from global tensor
                )

                # Score: sigmoid(MLP(z_src, z_dst)) for each candidate pair
                y_pred = self.link_pred(z[self.assoc[src]], z[self.assoc[dst]])

                # TGB evaluator expects {y_pred_pos, y_pred_neg, eval_metric}
                input_dict = {
                    "y_pred_pos":  np.array([y_pred[0].squeeze().cpu()]),
                    "y_pred_neg":  np.array(y_pred[1:].squeeze(-1).cpu()),
                    "eval_metric": ["mrr"],
                }
                perf_list.append(evaluator.eval(input_dict)["mrr"])

            # Update memory and neighbor cache with this batch's edges
            # (continuous evaluation: state carries over to the next batch)
            self.memory.update_state(pos_src, pos_dst, pos_t, pos_msg)
            self.neighbor_loader.insert(pos_src, pos_dst)

        return float(np.mean(perf_list)) if perf_list else 0.0
