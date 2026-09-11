"""
TGATModel: wrapper around the TPNet TGAT implementing the DynGraphEval BaseModel interface.

TGAT (Temporal Graph Attention Network) has NO memory bank — node state is not
carried between edges. Each query attends over the K most recent neighbors using
a TRAINABLE time encoder (Fourier features with learnable weights), so it can
learn that recent neighbors matter more than distant ones.

Differences from GraphMixerModel:
  - TimeEncoder is trainable (parameter_requires_grad=True)
  - Aggregation uses MultiHeadAttention instead of MLP-Mixer
  - Needs num_heads parameter

Differences from TPNetTGN:
  - warmup() only builds NeighborSampler; no memory replay needed
  - evaluate() has no memory-update step after each batch
  - compute_src_dst_node_temporal_embeddings() has no edges_are_positive / edge_ids args
"""

import time
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import TemporalData

from ..base import BaseModel
from ..tgn.tpnet_components import NeighborSampler, LinkPredictor
from .tgat_components import TGAT


class TGATModel(BaseModel):
    """
    Wrapper for the TPNet-style TGAT implementing the DynGraphEval BaseModel interface.

    Parameters
    ----------
    checkpoint_path : str      path to .pkl checkpoint from TPNet training
    num_nodes       : int      total number of nodes (0-indexed in PyG; +1 for 1-indexed format)
    msg_dim         : int      edge feature dimension
    train_data      : TemporalData
    val_data        : TemporalData
    test_data       : TemporalData
    device          : torch.device
    output_dim      : int      node embedding dimension (default 100)
    time_feat_dim   : int      time encoding dimension (default 100)
    num_layers      : int      attention layers (default 2)
    num_heads       : int      attention heads per layer (default 2)
    num_neighbors   : int      temporal neighbors per node (default 20)
    batch_size      : int      edges per evaluation batch (default 20)
    dropout         : float    dropout rate (default 0.1)
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
        dropout: float = 0.1,
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

        # ── Edge features: 1-indexed, index 0 = zero padding ──────────────────
        all_msgs = torch.cat([train_data.msg, val_data.msg, test_data.msg], dim=0).numpy()
        edge_raw_features = np.zeros((num_edges + 1, msg_dim), dtype=np.float32)
        edge_raw_features[1:] = all_msgs

        # ── Node features: tgbl-wiki has none → zeros ──────────────────────────
        node_raw_features = np.zeros((num_nodes + 1, 1), dtype=np.float32)

        # ── 1-indexed edge ID ranges per split ────────────────────────────────
        self._train_eids = np.arange(1,                       num_train + 1)
        self._val_eids   = np.arange(num_train + 1,           num_train + num_val + 1)
        self._test_eids  = np.arange(num_train + num_val + 1, num_edges + 1)

        # ── Build initial NeighborSampler (train only; replaced in warmup) ─────
        init_sampler = self._build_neighbor_sampler(
            [train_data], [self._train_eids], num_nodes
        )

        # ── Model components ───────────────────────────────────────────────────
        self.backbone = TGAT(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=init_sampler,
            time_feat_dim=time_feat_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            device=str(device),
        ).to(device)

        self.link_predictor = LinkPredictor(
            input_dim=output_dim,
            hidden_dim=output_dim,
        ).to(device)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_neighbor_sampler(self, tdata_list, eid_list, num_nodes):
        adj_list = [[] for _ in range(num_nodes + 1)]
        for tdata, eids in zip(tdata_list, eid_list):
            srcs  = tdata.src.numpy() + 1
            dsts  = tdata.dst.numpy() + 1
            times = tdata.t.numpy().astype(np.float64)
            for src, dst, eid, t in zip(srcs, dsts, eids, times):
                adj_list[src].append((dst, int(eid), t))
                adj_list[dst].append((src, int(eid), t))
        return NeighborSampler(adj_list=adj_list, sample_neighbor_strategy="recent", seed=0)

    def _to_numpy(self, tdata, eids):
        return (
            tdata.src.numpy() + 1,
            tdata.dst.numpy() + 1,
            tdata.t.numpy().astype(np.float64),
            eids,
        )

    # ── BaseModel interface ───────────────────────────────────────────────────

    def load_checkpoint(self, path: str = None) -> None:
        path = path or self.ckpt_path
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # TPNet saves {'model': state_dict, 'args': ..., 'message': ...}
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        backbone_sd = {k[2:]: v for k, v in state_dict.items() if k.startswith("0.")}
        lp_sd       = {k[2:]: v for k, v in state_dict.items() if k.startswith("1.")}

        self.backbone.load_state_dict(backbone_sd)
        self.link_predictor.load_state_dict(lp_sd)

    def warmup(self, train_data=None, val_data=None) -> None:
        """
        Build the full NeighborSampler over train+val+test.

        TGAT has no memory bank so no replay is needed — just ensure the
        neighbor sampler has access to the complete temporal graph so
        historical neighbor queries during test evaluation are accurate.
        """
        num_nodes = self.backbone.node_raw_features.shape[0] - 1

        full_sampler = self._build_neighbor_sampler(
            [self.train_data, self.val_data, self.test_data],
            [self._train_eids, self._val_eids, self._test_eids],
            num_nodes,
        )
        self.backbone.set_neighbor_sampler(full_sampler)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(
        self, eval_data: TemporalData, neg_sampler, split_mode: str, evaluator
    ) -> float:
        """
        Evaluate on eval_data and return mean MRR.

        TGAT has no memory so there is no memory-update step after each
        batch — we simply score every (positive, negatives) tuple independently.
        """
        self.backbone.eval()
        self.link_predictor.eval()

        srcs, dsts, times, _ = self._to_numpy(eval_data, self._test_eids)
        n         = len(srcs)
        perf_list = []
        n_batches = (n + self.batch_size - 1) // self.batch_size
        t0        = time.time()

        pbar = tqdm(
            range(0, n, self.batch_size),
            total=n_batches,
            desc="  scoring",
            ncols=100,
            unit="batch",
        )

        for start in pbar:
            end = min(start + self.batch_size, n)

            batch_src   = srcs[start:end]
            batch_dst   = dsts[start:end]
            batch_times = times[start:end]

            pos_src_torch = eval_data.src[start:end]
            pos_dst_torch = eval_data.dst[start:end]
            pos_t_torch   = eval_data.t[start:end]

            neg_batch_list = neg_sampler.query_batch(
                pos_src_torch, pos_dst_torch, pos_t_torch, split_mode=split_mode
            )

            for idx, neg_batch in enumerate(neg_batch_list):
                neg_dsts = np.array([int(d) + 1 for d in neg_batch], dtype=np.int64)
                if len(neg_dsts) == 0:
                    continue

                src_1 = np.array([batch_src[idx]])
                dst_1 = np.array([batch_dst[idx]])
                t_1   = np.array([batch_times[idx]])

                # Score positive
                src_emb_p, dst_emb_p = self.backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src_1, dst_node_ids=dst_1,
                    node_interact_times=t_1, num_neighbors=self.num_neighbors,
                )
                pos_score = self.link_predictor(src_emb_p, dst_emb_p).squeeze(-1).item()
                del src_emb_p, dst_emb_p

                # Score negatives
                neg_srcs = np.full(len(neg_dsts), batch_src[idx], dtype=np.int64)
                neg_ts   = np.full(len(neg_dsts), batch_times[idx])
                src_emb_n, dst_emb_n = self.backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=neg_srcs, dst_node_ids=neg_dsts,
                    node_interact_times=neg_ts, num_neighbors=self.num_neighbors,
                )
                neg_scores = self.link_predictor(src_emb_n, dst_emb_n).squeeze(-1).cpu().numpy()
                del src_emb_n, dst_emb_n

                input_dict = {
                    "y_pred_pos":  np.array([pos_score]),
                    "y_pred_neg":  neg_scores,
                    "eval_metric": ["mrr"],
                }
                perf_list.append(evaluator.eval(input_dict)["mrr"])

            running_mrr = float(np.mean(perf_list)) if perf_list else 0.0
            pbar.set_postfix(mrr=f"{running_mrr:.4f}", elapsed=f"{time.time()-t0:.0f}s")

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return float(np.mean(perf_list)) if perf_list else 0.0
