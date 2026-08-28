"""
Shared neural network modules used by TGN and FedLink models.

TGN Components (used by TGN and FederatedTGN):
    TimeEncoder            — maps scalar time deltas to cosine embeddings
    IdentityMessage        — concatenates node memories + edge feature + time encoding
    LastAggregator         — selects the most recent message per node
    TGNMemory              — GRU-based per-node memory module (the core of TGN)
    GraphAttentionEmbedding — TransformerConv with temporal edge attributes
    LinkPredictor          — MLP that scores (src_emb, dst_emb) pairs

FedLink Components (used by FedLink):
    GNNBase                — 2-layer GraphSAGE
    FedLinkModel           — user/page embeddings + GNNBase + dot-product scoring

These are unchanged from the original implementation and match the training code
in tgn_tgbl_wiki.ipynb and fl_fedlink_tgbl_wiki.ipynb.
"""

import copy
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear
from torch_geometric.nn import SAGEConv, TransformerConv
from torch_scatter import scatter


# ─────────────────────────────────────────────────────────────────────────────
# TGN Components
# ─────────────────────────────────────────────────────────────────────────────

class TimeEncoder(nn.Module):
    """
    Encode a scalar time delta into a fixed-size vector using a learnable
    cosine basis: enc(t) = cos(W * t + b).

    This is the standard TGN time encoder (Rossi et al., 2020).
    Output dimension = out_channels.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        # t: scalar or 1-D tensor of time deltas
        # unsqueeze adds the feature dimension, then cos maps to [-1, 1]
        return self.lin(t.unsqueeze(-1)).cos()


class IdentityMessage(nn.Module):
    """
    Message function: concatenate source memory, destination memory,
    raw edge feature, and time encoding into a single message vector.

    No learned projection — just concatenation (hence "Identity").
    Output dimension = raw_msg_dim + 2 * memory_dim + time_dim.
    """

    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(
        self,
        z_src: Tensor,   # source memory at time of interaction
        z_dst: Tensor,   # destination memory at time of interaction
        raw_msg: Tensor, # raw edge feature (e.g., edit vector)
        t_enc: Tensor,   # time encoding of relative timestamp
    ) -> Tensor:
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class LastAggregator(nn.Module):
    """
    When a node receives multiple messages in a batch, keep only the
    most recent one (highest timestamp) — discard older ones.

    Uses scatter_argmax so that gradients only flow through the selected
    message (not through messages that were overwritten).
    """

    def forward(
        self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int
    ) -> Tensor:
        from torch_geometric.utils._scatter import scatter_argmax

        # Find the index of the latest message per destination node
        argmax = scatter_argmax(t, index, dim=0, dim_size=dim_size)

        out  = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)   # nodes that received at least one message
        out[mask] = msg[argmax[mask]]
        return out


# Type alias for the per-node message store used inside TGNMemory
MsgStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]


class TGNMemory(nn.Module):
    """
    Per-node memory module at the heart of TGN.

    Each node maintains a fixed-size memory vector that gets updated
    via a GRU every time the node is involved in an interaction.

    Memory update pipeline (per batch):
        1. For each node in the batch, retrieve its message store
           (all past interactions stored since last update)
        2. Compute messages via msg_s_module (for src role) and
           msg_d_module (for dst role)
        3. Aggregate messages with LastAggregator (keep most recent)
        4. Update memory: new_mem = GRU(aggregated_msg, old_mem)

    Training vs. eval difference:
        - Training: memory is computed on-the-fly from message store
          (avoids gradient leakage across batches)
        - Eval: memory is read directly from the buffer
          (populated by checkpoint or update_state calls)
    """

    def __init__(
        self,
        num_nodes,
        raw_msg_dim,
        memory_dim,
        time_dim,
        message_module,
        aggregator_module,
    ):
        super().__init__()
        self.num_nodes   = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim  = memory_dim
        self.time_dim    = time_dim

        # Separate message modules for source role and destination role
        # (both have the same architecture but independent weights)
        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module  = aggregator_module

        self.time_enc       = TimeEncoder(time_dim)
        self.memory_updater = GRUCell(message_module.out_channels, memory_dim)

        # Persistent buffers (saved in checkpoints)
        self.register_buffer("memory",      torch.empty(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.empty(num_nodes, dtype=torch.long))
        self.register_buffer("_assoc",      torch.empty(num_nodes, dtype=torch.long))

        # Message stores: dict[node_id] -> (src, dst, t, raw_msg)
        self.msg_s_store: MsgStoreType = {}
        self.msg_d_store: MsgStoreType = {}

        self.reset_parameters()

    @property
    def device(self):
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Zero all memory vectors and clear message stores."""
        self.memory.fill_(0)
        self.last_update.fill_(0)
        self._reset_message_store()

    def detach(self):
        """Detach memory from computation graph (used for truncated BPTT)."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Return memory and last_update for a set of nodes.

        In training mode: recomputes memory on-the-fly from message store
        so that gradients flow correctly without leaking across batches.
        In eval mode: reads directly from the memory buffer.
        """
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]
        return memory, last_update

    def update_state(self, src, dst, t, raw_msg):
        """
        Ingest a batch of new interactions and update memory.

        Called after each batch during training and evaluation.
        The order of store/update is swapped between train and eval to
        correctly handle gradient flow.
        """
        n_id = torch.cat([src, dst]).unique()
        if self.training:
            # Training: update memory first (using OLD store), then store new msgs
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            # Eval: store new msgs first, then update memory to include them
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        """Initialize empty message stores for every node."""
        i   = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id]      = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id):
        """Compute updated memory for n_id by running the full message pipeline."""
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        msg_s, t_s, src_s, _ = self._compute_msg(n_id, self.msg_s_store, self.msg_s_module)
        msg_d, t_d, src_d, _ = self._compute_msg(n_id, self.msg_d_store, self.msg_d_module)

        idx  = torch.cat([src_s, src_d])
        msg  = torch.cat([msg_s, msg_d])
        t    = torch.cat([t_s,   t_d])

        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))
        memory = self.memory_updater(aggr, self.memory[n_id])
        last_update = scatter(
            t, idx, dim=0, dim_size=self.last_update.size(0), reduce="max"
        )[n_id]
        return memory, last_update

    def _update_msg_store(self, src, dst, t, raw_msg, msg_store):
        """Store the latest message per source node (overwrite older ones)."""
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(self, n_id, msg_store, msg_module):
        """Compute messages for all nodes in n_id from their message store."""
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src     = torch.cat(src).to(self.device)
        dst     = torch.cat(dst).to(self.device)
        t       = torch.cat(t).to(self.device)
        raw_msg = [m for i, m in enumerate(raw_msg) if m.numel() > 0 or i == 0]
        raw_msg = torch.cat(raw_msg).to(self.device)
        t_rel   = t - self.last_update[src]
        t_enc   = self.time_enc(t_rel.to(raw_msg.dtype))
        msg     = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return msg, t, src, dst

    def train(self, mode: bool = True):
        """On switching from train to eval: finalize memory and clear store."""
        if self.training and not mode:
            self._update_memory(
                torch.arange(self.num_nodes, device=self.memory.device)
            )
            self._reset_message_store()
        super().train(mode)


class GraphAttentionEmbedding(nn.Module):
    """
    Temporal graph transformer: TransformerConv with time-aware edge attributes.

    For each edge in the neighbor subgraph, computes:
        rel_t     = last_update[src] - t_edge   (how long ago did src interact?)
        rel_t_enc = TimeEncoder(rel_t)
        edge_attr = concat(rel_t_enc, raw_edge_msg)

    Then runs TransformerConv (2 heads × emb_dim/2) to produce node embeddings.
    """

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(self, x, last_update, edge_index, t, msg):
        # Relative time: how recently did the source node last interact?
        rel_t     = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(nn.Module):
    """
    MLP-based link scorer for TGN.

    Score(src, dst) = sigmoid(W3 * relu(W1 * z_src + W2 * z_dst))

    Returns a value in [0, 1] for each (src, dst) pair.
    Note: training uses BCEWithLogitsLoss so sigmoid is applied again there,
    but for evaluation the sigmoid output is used directly for ranking.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.lin_src   = Linear(in_channels, in_channels)
        self.lin_dst   = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()


# ─────────────────────────────────────────────────────────────────────────────
# FedLink Components
# ─────────────────────────────────────────────────────────────────────────────

class GNNBase(nn.Module):
    """
    2-layer GraphSAGE backbone for FedLink.

    Input: node feature matrix x (shape: num_nodes × hidden_dim)
    Output: updated node embeddings (same shape)

    No temporal reasoning — timestamps are discarded before this runs.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class FedLinkModel(nn.Module):
    """
    Full FedLink model: separate user/page embeddings + GraphSAGE + dot-product scoring.

    Architecture:
        x = concat(user_emb.weight, page_emb.weight)   # shape: (num_users+num_pages, hidden)
        z = GNNBase(x, static_edge_index)               # node embeddings
        score(src, dst) = (z[src] * z[dst]).sum(-1)    # dot product

    Timestamps are completely ignored — this is a static structural model.
    The separation of user and page embeddings matches FedLink's original design
    (heterogeneous user-item graph).
    """

    def __init__(self, num_users: int, num_pages: int, hidden_channels: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, hidden_channels)
        self.page_emb = nn.Embedding(num_pages, hidden_channels)
        self.gnn      = GNNBase(hidden_channels)

    def forward(self, edge_index: Tensor) -> Tensor:
        """Compute node embeddings from static graph structure."""
        x = torch.cat([self.user_emb.weight, self.page_emb.weight], dim=0)
        return self.gnn(x, edge_index)

    def predict(self, z: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        """Score (src, dst) pairs using dot product of embeddings."""
        return (z[src] * z[dst]).sum(dim=-1)
