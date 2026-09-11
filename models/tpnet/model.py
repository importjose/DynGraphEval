"""
TPNetModel: wrapper around TPNet implementing the DynGraphEval BaseModel interface.

TPNet (Time-decay Projection Network) is the SOTA model on the tgbl-wiki leaderboard.
It combines:
  - MLP-Mixer over temporal neighbors (like GraphMixer)
  - RandomProjectionModule: stateful temporal walk matrices that accumulate history

The RandomProjectionModule makes TPNet behave like a memory-bank model:
  - warmup() must replay all train+val interactions through .update() to restore
    the correct projection state before test evaluation
  - evaluate() must call .update() after each batch (causal: score first, then update)

Fixed (non-trainable) time encoder — same as GraphMixer.
No recurrent memory bank — state lives entirely in the random projections.
"""

import time
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import TemporalData

from ..base import BaseModel
from ..tgn.tpnet_components import NeighborSampler
from .tpnet_components import RandomProjectionModule, TPNet, LinkPredictor_v1


class TPNetModel(BaseModel):
    """
    Wrapper for TPNet implementing the DynGraphEval BaseModel interface.

    Parameters
    ----------
    checkpoint_path    : str    path to .pkl checkpoint from TPNet training
    num_nodes          : int    total number of nodes (0-indexed in PyG; +1 for 1-indexed)
    msg_dim            : int    edge feature dimension
    train_data         : TemporalData
    val_data           : TemporalData
    test_data          : TemporalData
    device             : torch.device
    output_dim         : int    node embedding dimension (default 100)
    time_feat_dim      : int    time encoding dimension (default 100)
    num_layers         : int    MLP-Mixer layers (default 2)
    num_neighbors      : int    temporal neighbors per node (default 20)
    batch_size         : int    edges per evaluation batch (default 200)
    dropout            : float  dropout rate (default 0.1)
    rp_dim_factor      : int    controls random projection dimension (default 10)
    rp_num_layer       : int    max hop of temporal walk matrices (default 2)
    rp_time_decay      : float  time decay weight lambda (default 1e-6)
    warmup_batch_size  : int    batch size for warmup replay (default 200)
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
        num_neighbors: int = 20,
        batch_size: int = 200,
        dropout: float = 0.1,
        rp_dim_factor: int = 10,
        rp_num_layer: int = 2,
        rp_time_decay: float = 1e-6,
        warmup_batch_size: int = 200,
    ):
        self.device            = device
        self.batch_size        = batch_size
        self.num_neighbors     = num_neighbors
        self.ckpt_path         = checkpoint_path
        self.warmup_batch_size = warmup_batch_size

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

        # ── Beginning time for RandomProjectionModule ──────────────────────────
        beginning_time = float(train_data.t.min().item())

        # ── Build initial NeighborSampler (train only; replaced in warmup) ─────
        init_sampler = self._build_neighbor_sampler(
            [train_data], [self._train_eids], num_nodes
        )

        # ── RandomProjectionModule (shared by backbone and link predictor) ─────
        self.random_projections = RandomProjectionModule(
            node_num=num_nodes + 1,
            edge_num=num_edges,
            dim_factor=rp_dim_factor,
            num_layer=rp_num_layer,
            time_decay_weight=rp_time_decay,
            device=str(device),
            use_matrix=False,
            beginning_time=np.float64(beginning_time),
            not_scale=False,
            enforce_dim=-1,
        ).to(device)

        # ── Model components ───────────────────────────────────────────────────
        self.backbone = TPNet(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=init_sampler,
            time_feat_dim=time_feat_dim,
            output_dim=output_dim,
            dropout=dropout,
            random_projections=self.random_projections,
            num_layers=num_layers,
            num_neighbors=num_neighbors,
            device=str(device),
        ).to(device)

        self.link_predictor = LinkPredictor_v1(
            input_dim=output_dim,
            hidden_dim=output_dim,
            output_dim=1,
            random_projections=self.random_projections,
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

    def _replay_through_projections(self, tdata, desc: str):
        """Replay interactions through RandomProjectionModule.update() in batches."""
        srcs  = tdata.src.numpy() + 1
        dsts  = tdata.dst.numpy() + 1
        times = tdata.t.numpy().astype(np.float64)
        n = len(srcs)
        for start in tqdm(range(0, n, self.warmup_batch_size), desc=desc, ncols=100):
            end = min(start + self.warmup_batch_size, n)
            self.random_projections.update(
                src_node_ids=srcs[start:end],
                dst_node_ids=dsts[start:end],
                node_interact_times=times[start:end],
            )

    # ── BaseModel interface ───────────────────────────────────────────────────

    def load_checkpoint(self, path: str = None) -> None:
        path = path or self.ckpt_path
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        backbone_sd = {k[2:]: v for k, v in state_dict.items() if k.startswith("0.")}
        lp_sd       = {k[2:]: v for k, v in state_dict.items() if k.startswith("1.")}

        self.backbone.load_state_dict(backbone_sd)
        self.link_predictor.load_state_dict(lp_sd)
        # random_projections state is now loaded via backbone (shared object)

    def warmup(self, train_data=None, val_data=None) -> None:
        """
        Restore RandomProjectionModule to end-of-val state, then rebuild NeighborSampler.

        TPNet's random projections must reflect all prior interactions before test
        evaluation. We:
          1. Keep random_projections[0] (trained basis) — reset_zero=False
          2. Clear accumulated projections[1:] and reset now_time
          3. Replay train interactions through .update()
          4. Replay val interactions through .update()
        """
        num_nodes = self.backbone.node_raw_features.shape[0] - 1

        # Reset accumulated projections, keep trained basis
        self.random_projections.reset_random_projections(reset_zero=False)

        # Replay to rebuild projection state
        self._replay_through_projections(self.train_data, "  warmup train")
        self._replay_through_projections(self.val_data,   "  warmup val  ")

        # Rebuild full NeighborSampler (train + val + test)
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

        Unlike GraphMixer/TGAT, TPNet must call random_projections.update() after
        each batch (score first, then update) so the projection state stays causally
        consistent with the interaction history seen so far.
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
                    src_node_ids=src_1, dst_node_ids=dst_1, node_interact_times=t_1,
                )
                pos_score = self.link_predictor(
                    src_node_ids=src_1, dst_node_ids=dst_1,
                    src_emb=src_emb_p, dst_emb=dst_emb_p,
                ).squeeze(-1).item()
                del src_emb_p, dst_emb_p

                # Score negatives
                neg_srcs = np.full(len(neg_dsts), batch_src[idx], dtype=np.int64)
                neg_ts   = np.full(len(neg_dsts), batch_times[idx])
                src_emb_n, dst_emb_n = self.backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=neg_srcs, dst_node_ids=neg_dsts, node_interact_times=neg_ts,
                )
                neg_scores = self.link_predictor(
                    src_node_ids=neg_srcs, dst_node_ids=neg_dsts,
                    src_emb=src_emb_n, dst_emb=dst_emb_n,
                ).squeeze(-1).cpu().numpy()
                del src_emb_n, dst_emb_n

                input_dict = {
                    "y_pred_pos":  np.array([pos_score]),
                    "y_pred_neg":  neg_scores,
                    "eval_metric": ["mrr"],
                }
                perf_list.append(evaluator.eval(input_dict)["mrr"])

            # ── Causal update: update projections with this batch AFTER scoring ─
            self.random_projections.update(
                src_node_ids=batch_src,
                dst_node_ids=batch_dst,
                node_interact_times=batch_times,
            )

            running_mrr = float(np.mean(perf_list)) if perf_list else 0.0
            pbar.set_postfix(mrr=f"{running_mrr:.4f}", elapsed=f"{time.time()-t0:.0f}s")

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return float(np.mean(perf_list)) if perf_list else 0.0
