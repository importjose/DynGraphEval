"""
TPNet model components from lxd99/TGB_TPNet, copied verbatim for version stability.

Source: https://github.com/lxd99/TGB_TPNet
Files:  models/TPNet.py, models/modules.py (LinkPredictor_v1)

TPNet differs from GraphMixer/TGAT in one critical way: the RandomProjectionModule
maintains STATEFUL temporal walk matrices that accumulate interaction history.
These must be updated (via .update()) after every batch of interactions, during
both warmup and evaluation — making TPNet behave like a memory-bank model despite
having no explicit recurrent memory.

TimeEncoder and NeighborSampler are imported from tgn.tpnet_components to avoid
duplication.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from ..tgn.tpnet_components import NeighborSampler, TimeEncoder


# ── RandomProjectionModule ──────────────────────────────────────────────────────

class RandomProjectionModule(nn.Module):
    """
    Maintains temporal walk matrices A^(0)(t), ..., A^(k)(t) via random
    feature propagation.  Provides pairwise node features for link scoring.

    Must call .update() after every observed interaction batch so the matrices
    reflect current history.  Must call .reset_random_projections(reset_zero=False)
    at the start of any replay pass to clear accumulated higher-order projections
    while keeping the trained basis (random_projections[0]).
    """

    def __init__(
        self,
        node_num: int,
        edge_num: int,
        dim_factor: int,
        num_layer: int,
        time_decay_weight: float,
        device: str,
        use_matrix: bool,
        beginning_time: np.float64,
        not_scale: bool,
        enforce_dim: int,
    ):
        super(RandomProjectionModule, self).__init__()
        self.node_num = node_num
        self.edge_num = edge_num
        if enforce_dim != -1:
            self.dim = enforce_dim
        else:
            self.dim = min(int(math.log(self.edge_num * 2)) * dim_factor, node_num)
        self.num_layer = num_layer
        self.time_decay_weight = time_decay_weight
        self.begging_time = nn.Parameter(torch.tensor(beginning_time), requires_grad=False)
        self.now_time     = nn.Parameter(torch.tensor(beginning_time), requires_grad=False)
        self.device = device
        self.random_projections = nn.ParameterList()
        self.use_matrix = use_matrix
        self.not_scale  = not_scale

        if self.use_matrix:
            self.dim = self.node_num
            for i in range(self.num_layer + 1):
                if i == 0:
                    self.random_projections.append(
                        nn.Parameter(torch.eye(self.node_num), requires_grad=False))
                else:
                    self.random_projections.append(
                        nn.Parameter(torch.zeros_like(self.random_projections[i - 1]), requires_grad=False))
        else:
            for i in range(self.num_layer + 1):
                if i == 0:
                    self.random_projections.append(
                        nn.Parameter(
                            torch.normal(0, 1 / math.sqrt(self.dim), (self.node_num, self.dim)),
                            requires_grad=False))
                else:
                    self.random_projections.append(
                        nn.Parameter(torch.zeros_like(self.random_projections[i - 1]), requires_grad=False))

        self.pair_wise_feature_dim = (2 * self.num_layer + 2) ** 2
        self.mlp = nn.Sequential(
            nn.Linear(self.pair_wise_feature_dim, self.pair_wise_feature_dim * 4),
            nn.ReLU(),
            nn.Linear(self.pair_wise_feature_dim * 4, self.pair_wise_feature_dim),
        )

    def update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
               node_interact_times: np.ndarray):
        src_node_ids = torch.from_numpy(src_node_ids).to(self.device)
        dst_node_ids = torch.from_numpy(dst_node_ids).to(self.device)
        next_time = node_interact_times[-1]
        node_interact_times = torch.from_numpy(node_interact_times).to(
            dtype=torch.float, device=self.device)
        time_weight = torch.exp(
            -self.time_decay_weight * (next_time - node_interact_times))[:, None]

        for i in range(1, self.num_layer + 1):
            self.random_projections[i].data = self.random_projections[i].data * np.power(
                np.exp(-self.time_decay_weight * (next_time - self.now_time.cpu().numpy())), i)

        for i in range(self.num_layer, 0, -1):
            src_update = self.random_projections[i - 1][dst_node_ids] * time_weight
            dst_update = self.random_projections[i - 1][src_node_ids] * time_weight
            self.random_projections[i].scatter_add_(
                dim=0, index=src_node_ids[:, None].expand(-1, self.dim), src=src_update)
            self.random_projections[i].scatter_add_(
                dim=0, index=dst_node_ids[:, None].expand(-1, self.dim), src=dst_update)

        self.now_time.data = torch.tensor(next_time, device=self.device)

    def get_random_projections(self, node_ids: np.ndarray):
        return [self.random_projections[i][node_ids] for i in range(self.num_layer + 1)]

    def get_pair_wise_feature(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray):
        src_rp = torch.stack(self.get_random_projections(src_node_ids), dim=1)
        dst_rp = torch.stack(self.get_random_projections(dst_node_ids), dim=1)
        rp = torch.cat([src_rp, dst_rp], dim=1)
        feat = torch.matmul(rp, rp.transpose(1, 2)).reshape(len(src_node_ids), -1)
        if self.not_scale:
            return self.mlp(feat)
        else:
            feat[feat < 0] = 0
            feat = torch.log(feat + 1.0)
            return self.mlp(feat)

    def reset_random_projections(self, reset_zero: bool = True):
        for i in range(1, self.num_layer + 1):
            nn.init.zeros_(self.random_projections[i])
        self.now_time.data = self.begging_time.clone()
        if not self.use_matrix and reset_zero:
            nn.init.normal_(self.random_projections[0], mean=0, std=1 / math.sqrt(self.dim))

    def backup_random_projections(self):
        return (self.now_time.clone(),
                [self.random_projections[i].clone() for i in range(1, self.num_layer + 1)])

    def reload_random_projections(self, backup):
        now_time, rp_list = backup
        self.now_time.data = now_time.clone()
        for i in range(1, self.num_layer + 1):
            self.random_projections[i].data = rp_list[i - 1].clone()


# ── FeedForwardNet ──────────────────────────────────────────────────────────────

class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        super(FeedForwardNet, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, int(dim_expansion_factor * input_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim_expansion_factor * input_dim), input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


# ── MLPMixer ────────────────────────────────────────────────────────────────────

class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int,
                 token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0,
                 dropout: float = 0.0):
        super(MLPMixer, self).__init__()
        self.token_norm       = nn.LayerNorm(num_tokens)
        self.token_feedforward   = FeedForwardNet(num_tokens,   token_dim_expansion_factor,   dropout)
        self.channel_norm     = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(num_channels, channel_dim_expansion_factor, dropout)

    def forward(self, x: torch.Tensor):
        # mix tokens
        h = self.token_feedforward(self.token_norm(x.permute(0, 2, 1))).permute(0, 2, 1)
        x = h + x
        # mix channels
        h = self.channel_feedforward(self.channel_norm(x))
        return h + x


# ── TPNetEmbedding ──────────────────────────────────────────────────────────────

class TPNetEmbedding(nn.Module):

    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler,
                 time_encoder, node_feat_dim, edge_feat_dim, output_dim,
                 time_feat_dim, num_layers, num_neighbors, dropout, random_projections):
        super(TPNetEmbedding, self).__init__()
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler  = neighbor_sampler
        self.time_encoder      = time_encoder
        self.node_feat_dim     = node_feat_dim
        self.edge_feat_dim     = edge_feat_dim
        self.output_dim        = output_dim
        self.time_feat_dim     = time_feat_dim
        self.num_layers        = num_layers
        self.num_neighbors     = num_neighbors
        self.dropout           = dropout
        self.random_projections = random_projections

        rp_dim = random_projections.pair_wise_feature_dim * 2 if random_projections is not None else 0
        self.projection_layer = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim + time_feat_dim + rp_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=num_neighbors, num_channels=output_dim,
                     token_dim_expansion_factor=0.5, channel_dim_expansion_factor=4.0,
                     dropout=dropout)
            for _ in range(num_layers)
        ])

    def compute_node_temporal_embeddings(self, node_ids, src_node_ids, dst_node_ids,
                                         node_interact_times):
        device = self.node_raw_features.device

        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=self.num_neighbors,
            )

        neighbor_node_features = self.node_raw_features[torch.from_numpy(neighbor_node_ids)]
        neighbor_delta_times   = torch.from_numpy(
            node_interact_times[:, np.newaxis] - neighbor_times).float().to(device)
        neighbor_delta_times   = torch.log(neighbor_delta_times + 1.0)
        neighbor_time_features = self.time_encoder(neighbor_delta_times)
        neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]

        if self.random_projections is not None:
            concat_rp = self.random_projections.get_pair_wise_feature(
                src_node_ids=np.tile(neighbor_node_ids.reshape(-1), 2),
                dst_node_ids=np.concatenate([
                    np.repeat(src_node_ids, self.num_neighbors),
                    np.repeat(dst_node_ids, self.num_neighbors),
                ]),
            )
            neighbor_rp = torch.cat([
                concat_rp[:len(node_ids) * self.num_neighbors],
                concat_rp[len(node_ids) * self.num_neighbors:],
            ], dim=1).reshape(len(node_ids), self.num_neighbors, -1)
            combined = torch.cat([neighbor_node_features, neighbor_time_features,
                                  neighbor_edge_features, neighbor_rp], dim=2)
        else:
            combined = torch.cat([neighbor_node_features, neighbor_time_features,
                                  neighbor_edge_features], dim=2)

        embeddings = self.projection_layer(combined)
        embeddings.masked_fill(
            torch.from_numpy(neighbor_node_ids == 0)[:, :, None].to(device), 0)
        for mixer in self.mlp_mixers:
            embeddings = mixer(embeddings)
        return torch.mean(embeddings, dim=1)


# ── TPNet ───────────────────────────────────────────────────────────────────────

class TPNet(nn.Module):

    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler,
                 time_feat_dim, output_dim, dropout, random_projections,
                 num_layers, num_neighbors, device, not_embedding=False):
        super(TPNet, self).__init__()

        self.node_raw_features = torch.from_numpy(
            node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(
            edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim     = self.node_raw_features.shape[1]
        self.edge_feat_dim     = self.edge_raw_features.shape[1]
        self.time_feat_dim     = time_feat_dim
        self.output_dim        = output_dim
        self.device            = device
        self.not_embedding     = not_embedding
        self.num_nodes         = self.node_raw_features.shape[0]
        self.random_projections = random_projections

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)

        if not self.not_embedding:
            self.embedding_module = TPNetEmbedding(
                node_raw_features=self.node_raw_features,
                edge_raw_features=self.edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                time_encoder=self.time_encoder,
                node_feat_dim=self.node_feat_dim,
                edge_feat_dim=self.edge_feat_dim,
                time_feat_dim=time_feat_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                num_neighbors=num_neighbors,
                dropout=dropout,
                random_projections=random_projections,
            )

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids,
                                                 node_interact_times):
        if self.not_embedding:
            n = len(src_node_ids)
            zeros = torch.zeros((2 * n, self.output_dim), device=self.device)
            return zeros[:n], zeros[n:]

        embeddings = self.embedding_module.compute_node_temporal_embeddings(
            node_ids=np.concatenate([src_node_ids, dst_node_ids]),
            src_node_ids=np.tile(src_node_ids, 2),
            dst_node_ids=np.tile(dst_node_ids, 2),
            node_interact_times=np.tile(node_interact_times, 2),
        )
        n = len(src_node_ids)
        return embeddings[:n], embeddings[n:]

    def set_neighbor_sampler(self, neighbor_sampler):
        if not self.not_embedding:
            self.embedding_module.neighbor_sampler = neighbor_sampler
            ns = self.embedding_module.neighbor_sampler
            if ns.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
                assert ns.seed is not None
                ns.reset_random_state()


# ── LinkPredictor_v1 ────────────────────────────────────────────────────────────

class LinkPredictor_v1(nn.Module):
    """
    Link predictor that optionally appends RandomProjectionModule pairwise features.
    Used by TPNet (and only TPNet) — other models use the simpler LinkPredictor.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 random_projections=None, not_encode: bool = False):
        super().__init__()
        self.random_projections = random_projections
        self.not_encode = not_encode
        rp_dim = random_projections.pair_wise_feature_dim if random_projections is not None else 0
        self.fc1 = nn.Linear(input_dim * 2 + rp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                src_emb: torch.Tensor, dst_emb: torch.Tensor):
        if self.not_encode:
            src_emb = torch.zeros_like(src_emb)
            dst_emb = torch.zeros_like(dst_emb)
        if self.random_projections is not None:
            rp = self.random_projections.get_pair_wise_feature(src_node_ids, dst_node_ids)
            x = torch.cat([src_emb, dst_emb, rp], dim=1)
        else:
            x = torch.cat([src_emb, dst_emb], dim=1)
        return self.fc2(self.act(self.fc1(x)))
