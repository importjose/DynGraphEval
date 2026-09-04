"""
TPNetTGN: wrapper around the TGB_TPNet MemoryModel (TGN variant).

Compared to our TGN:
  - NeighborSampler stores ALL historical neighbors (not just last 10 via ring buffer)
  - 2-layer GraphAttentionEmbedding (vs our 1-layer TransformerConv)
  - 1-indexed node and edge IDs (index 0 is padding)
  - LinkPredictor is a 2-layer MLP on concat(src_emb, dst_emb) rather than a dot-product

Checkpoint format:
  Saved by their train_link_prediction.py as a state_dict of nn.Sequential(backbone, link_pred).
  Keys are prefixed '0.' (MemoryModel) and '1.' (LinkPredictor).
  The file is a .pkl saved with torch.save(model.state_dict(), path).
"""

import numpy as np
import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from ..base_model import BaseModel
from .tpnet_components import MemoryModel, NeighborSampler, LinkPredictor


class TPNetTGN(BaseModel):
    """
    Wrapper for the TPNet-style TGN model implementing the DynGraphEval BaseModel interface.

    Parameters
    ----------
    checkpoint_path : str      path to .pkl checkpoint from their training code
    num_nodes       : int      total number of nodes (0-indexed in PyG; +1 for their 1-indexed format)
    msg_dim         : int      edge feature dimension
    train_data      : TemporalData
    val_data        : TemporalData
    test_data       : TemporalData
    device          : torch.device
    output_dim      : int      node embedding dimension (default 100)
    time_feat_dim   : int      time encoding dimension (default 100)
    num_layers      : int      GraphAttention layers (default 2)
    num_heads       : int      attention heads (default 2)
    num_neighbors   : int      temporal neighbors per node (default 20)
    batch_size      : int      edges per evaluation batch (default 20)
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_nodes: int,
        msg_dim: int,
        train_data: TemporalData,
        val_data: TemporalData,
        test_data: TemporalData,
        device: torch.device,
        output_dim: int = 100,
        time_feat_dim: int = 100,
        num_layers: int = 2,
        num_heads: int = 2,
        num_neighbors: int = 20,
        batch_size: int = 20,
    ):
        self.device        = device
        self.batch_size    = batch_size
        self.num_neighbors = num_neighbors
        self.ckpt_path     = checkpoint_path

        self.train_data = train_data
        self.val_data   = val_data
        self.test_data  = test_data

        num_train = train_data.num_events
        num_val   = val_data.num_events
        num_test  = test_data.num_events
        num_edges = num_train + num_val + num_test

        # ── Build edge_raw_features (1-indexed, index 0 = zero padding) ───────
        # Concatenate all edge features in chronological order: train, val, test
        all_msgs = torch.cat([train_data.msg, val_data.msg, test_data.msg], dim=0).numpy()
        edge_raw_features = np.zeros((num_edges + 1, msg_dim), dtype=np.float32)
        edge_raw_features[1:] = all_msgs  # 1-indexed

        # ── Node features: tgbl-wiki has none → zeros (num_nodes+1, 1) ────────
        # index 0 = padding; the +1 accounts for 1-indexed node IDs
        node_raw_features = np.zeros((num_nodes + 1, 1), dtype=np.float32)

        # ── Precompute 1-indexed edge ID ranges for each split ─────────────────
        self._train_eids = np.arange(1,                       num_train + 1)
        self._val_eids   = np.arange(num_train + 1,           num_train + num_val + 1)
        self._test_eids  = np.arange(num_train + num_val + 1, num_edges + 1)

        # ── Build a minimal NeighborSampler for model construction ─────────────
        # Will be replaced with the full sampler in warmup().
        init_sampler = self._build_neighbor_sampler(
            [train_data], [self._train_eids], num_nodes)

        # ── Model components ───────────────────────────────────────────────────
        self.backbone = MemoryModel(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=init_sampler,
            output_dim=output_dim,
            time_feat_dim=time_feat_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            device=str(device),
        ).to(device)

        self.link_predictor = LinkPredictor(
            input_dim=output_dim,
            hidden_dim=output_dim,
        ).to(device)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_neighbor_sampler(self, tdata_list, eid_list, num_nodes):
        """Build NeighborSampler from one or more TemporalData splits."""
        adj_list = [[] for _ in range(num_nodes + 1)]  # 1-indexed: index 0 unused
        for tdata, eids in zip(tdata_list, eid_list):
            srcs  = tdata.src.numpy() + 1  # 0→1 indexed
            dsts  = tdata.dst.numpy() + 1
            times = tdata.t.numpy().astype(np.float64)
            for src, dst, eid, t in zip(srcs, dsts, eids, times):
                adj_list[src].append((dst, int(eid), t))
                adj_list[dst].append((src, int(eid), t))
        return NeighborSampler(adj_list=adj_list, sample_neighbor_strategy='recent', seed=0)

    def _to_numpy(self, tdata, eids):
        """Convert a TemporalData split to 1-indexed numpy arrays."""
        return (tdata.src.numpy() + 1,          # src (1-indexed)
                tdata.dst.numpy() + 1,           # dst (1-indexed)
                tdata.t.numpy().astype(np.float64),
                eids)

    # ── BaseModel interface ───────────────────────────────────────────────────

    def load_checkpoint(self, path: str = None) -> None:
        """
        Load checkpoint saved by their train_link_prediction.py.

        Expected format: state_dict of nn.Sequential(backbone, link_pred).
        Keys: '0.*' for MemoryModel, '1.*' for LinkPredictor.
        """
        path = path or self.ckpt_path
        ckpt = torch.load(path, map_location=self.device)

        backbone_sd = {k[2:]: v for k, v in ckpt.items() if k.startswith('0.')}
        lp_sd       = {k[2:]: v for k, v in ckpt.items() if k.startswith('1.')}

        self.backbone.load_state_dict(backbone_sd)
        self.link_predictor.load_state_dict(lp_sd)

        # Reset memory after loading weights
        self.backbone.memory_bank.__init_memory_bank__()

    def warmup(self, train_data=None, val_data=None) -> None:
        """
        Build the full NeighborSampler (train+val+test adjacency) and replay
        train+val edges through memory to populate node states.

        The NeighborSampler includes test edges so temporal neighborhood
        queries during test evaluation have access to the full graph.
        Crucially, memory replay only uses train+val — test edges are never
        seen during warmup to avoid label leakage.
        """
        num_nodes = self.backbone.num_nodes - 1  # subtract padding index

        # NeighborSampler: train + val + test (full temporal graph)
        full_sampler = self._build_neighbor_sampler(
            [self.train_data, self.val_data, self.test_data],
            [self._train_eids, self._val_eids, self._test_eids],
            num_nodes,
        )
        self.backbone.set_neighbor_sampler(full_sampler)

        # Reset memory then replay train+val
        self.backbone.memory_bank.__init_memory_bank__()

        with torch.no_grad():
            for tdata, eids in [(self.train_data, self._train_eids),
                                 (self.val_data,   self._val_eids)]:
                srcs, dsts, times, edge_ids = self._to_numpy(tdata, eids)
                n = len(srcs)
                for start in range(0, n, self.batch_size):
                    end = min(start + self.batch_size, n)
                    self.backbone.compute_src_dst_node_temporal_embeddings(
                        src_node_ids       = srcs[start:end],
                        dst_node_ids       = dsts[start:end],
                        node_interact_times= times[start:end],
                        edge_ids           = edge_ids[start:end],
                        edges_are_positive = True,
                        num_neighbors      = self.num_neighbors,
                    )
                    self.backbone.memory_bank.detach_memory_bank()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(
        self, eval_data: TemporalData, neg_sampler, split_mode: str, evaluator
    ) -> float:
        """
        Evaluate on eval_data and return mean MRR.

        Protocol (matches their evaluation):
          For each batch of positive edges:
            1. Score each positive edge + its negatives WITHOUT memory update.
            2. Update memory with the positive batch (edges_are_positive=True).
        """
        self.backbone.eval()
        self.link_predictor.eval()

        srcs, dsts, times, edge_ids = self._to_numpy(eval_data, self._test_eids)
        n         = len(srcs)
        perf_list = []

        # We iterate in batches but score per-edge (TGB has per-edge neg lists)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)

            batch_src   = srcs[start:end]
            batch_dst   = dsts[start:end]
            batch_times = times[start:end]
            batch_eids  = edge_ids[start:end]

            # PyG tensors for the neg sampler (it expects 0-indexed torch tensors)
            pos_src_torch = eval_data.src[start:end]
            pos_dst_torch = eval_data.dst[start:end]
            pos_t_torch   = eval_data.t[start:end]

            neg_batch_list = neg_sampler.query_batch(
                pos_src_torch, pos_dst_torch, pos_t_torch, split_mode=split_mode)

            # ── Score each edge against its negatives (no memory update) ──────
            for idx, neg_batch in enumerate(neg_batch_list):
                # 1-indexed negative destinations
                neg_dsts = np.array([int(d) + 1 for d in neg_batch], dtype=np.int64)
                if len(neg_dsts) == 0:
                    continue

                src_1  = np.array([batch_src[idx]])
                dst_1  = np.array([batch_dst[idx]])
                t_1    = np.array([batch_times[idx]])

                # Score positive
                src_emb_p, dst_emb_p = self.backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src_1, dst_node_ids=dst_1, node_interact_times=t_1,
                    edge_ids=None, edges_are_positive=False, num_neighbors=self.num_neighbors)
                pos_score = self.link_predictor(src_emb_p, dst_emb_p).squeeze(-1).item()
                del src_emb_p, dst_emb_p

                # Score negatives (same src, same time, different dsts)
                neg_srcs = np.full(len(neg_dsts), batch_src[idx], dtype=np.int64)
                neg_ts   = np.full(len(neg_dsts), batch_times[idx])
                src_emb_n, dst_emb_n = self.backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=neg_srcs, dst_node_ids=neg_dsts, node_interact_times=neg_ts,
                    edge_ids=None, edges_are_positive=False, num_neighbors=self.num_neighbors)
                neg_scores = self.link_predictor(src_emb_n, dst_emb_n).squeeze(-1).cpu().numpy()
                del src_emb_n, dst_emb_n

                input_dict = {
                    "y_pred_pos":  np.array([pos_score]),
                    "y_pred_neg":  neg_scores,
                    "eval_metric": ["mrr"],
                }
                perf_list.append(evaluator.eval(input_dict)["mrr"])

            # ── Update memory with positive batch ──────────────────────────────
            self.backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids       = batch_src,
                dst_node_ids       = batch_dst,
                node_interact_times= batch_times,
                edge_ids           = batch_eids,
                edges_are_positive = True,
                num_neighbors      = self.num_neighbors,
            )
            self.backbone.memory_bank.detach_memory_bank()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return float(np.mean(perf_list)) if perf_list else 0.0
