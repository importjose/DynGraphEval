"""
GraphMixer model components from lxd99/TGB_TPNet, copied verbatim for version stability.

Source: https://github.com/lxd99/TGB_TPNet
Files:  models/GraphMixer.py, models/modules.py (TimeEncoder), utils/utils.py (NeighborSampler)

Only the components needed for GraphMixer are included.
NeighborSampler is imported from tpnet_components to avoid duplication.
"""

import numpy as np
import torch
import torch.nn as nn

from ..tgn.tpnet_components import NeighborSampler, TimeEncoder, LinkPredictor


# ── FeedForwardNet ─────────────────────────────────────────────────────────────

class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        super(FeedForwardNet, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


# ── MLPMixer ───────────────────────────────────────────────────────────────────

class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int,
                 token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0,
                 dropout: float = 0.0):
        super(MLPMixer, self).__init__()
        self.token_norm       = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens,
                                                dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)
        self.channel_norm       = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels,
                                                  dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)

    def forward(self, input_tensor: torch.Tensor):
        # mix tokens
        hidden = self.token_norm(input_tensor.permute(0, 2, 1))
        hidden = self.token_feedforward(hidden).permute(0, 2, 1)
        output = hidden + input_tensor
        # mix channels
        hidden = self.channel_norm(output)
        hidden = self.channel_feedforward(hidden)
        output = hidden + output
        return output


# ── GraphMixer ─────────────────────────────────────────────────────────────────

class GraphMixer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                 neighbor_sampler: NeighborSampler, time_feat_dim: int, output_dim: int,
                 num_tokens: int, num_layers: int = 2,
                 token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0,
                 dropout: float = 0.1, device: str = 'cpu'):
        super(GraphMixer, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler            = neighbor_sampler
        self.node_feat_dim               = self.node_raw_features.shape[1]
        self.edge_feat_dim               = self.edge_raw_features.shape[1]
        self.time_feat_dim               = time_feat_dim
        self.output_dim                  = output_dim
        self.num_tokens                  = num_tokens
        self.num_layers                  = num_layers
        self.token_dim_expansion_factor  = token_dim_expansion_factor
        self.channel_dim_expansion_factor = channel_dim_expansion_factor
        self.dropout                     = dropout
        self.device                      = device

        self.num_channels = self.output_dim
        # Non-trainable time encoder (key design choice of GraphMixer)
        self.time_encoder  = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(self.edge_feat_dim + time_feat_dim, self.num_channels)

        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_tokens, num_channels=self.num_channels,
                     token_dim_expansion_factor=self.token_dim_expansion_factor,
                     channel_dim_expansion_factor=self.channel_dim_expansion_factor,
                     dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(
            in_features=self.num_channels + self.node_feat_dim,
            out_features=self.output_dim,
            bias=True,
        )

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 num_neighbors: int = 20,
                                                 time_gap: int = 2000):
        src_emb = self.compute_node_temporal_embeddings(
            node_ids=src_node_ids, node_interact_times=node_interact_times,
            num_neighbors=num_neighbors, time_gap=time_gap)
        dst_emb = self.compute_node_temporal_embeddings(
            node_ids=dst_node_ids, node_interact_times=node_interact_times,
            num_neighbors=num_neighbors, time_gap=time_gap)
        return src_emb, dst_emb

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray,
                                         node_interact_times: np.ndarray,
                                         num_neighbors: int = 20,
                                         time_gap: int = 2000):
        # ── Link encoder ──────────────────────────────────────────────────────
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=num_neighbors)

        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
        nodes_neighbor_time_features = self.time_encoder(
            timestamps=torch.from_numpy(
                node_interact_times[:, np.newaxis] - neighbor_times
            ).float().to(self.device))

        nodes_neighbor_time_features.masked_fill_(
            torch.from_numpy(neighbor_node_ids == 0)[:, :, None].to(self.device), 0.0)

        combined_features = torch.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim=-1)
        combined_features = self.projection_layer(combined_features)

        for mlp_mixer in self.mlp_mixers:
            combined_features = mlp_mixer(input_tensor=combined_features)

        combined_features = torch.mean(combined_features, dim=1)

        # ── Node encoder ──────────────────────────────────────────────────────
        time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(
            node_ids=node_ids,
            node_interact_times=node_interact_times,
            num_neighbors=time_gap)

        nodes_time_gap_neighbor_node_raw_features = self.node_raw_features[
            torch.from_numpy(time_gap_neighbor_node_ids)]

        valid_mask = torch.from_numpy((time_gap_neighbor_node_ids > 0).astype(np.float32))
        valid_mask[valid_mask == 0] = -1e10
        scores = torch.softmax(valid_mask, dim=1).to(self.device)

        nodes_time_gap_agg = torch.mean(
            nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1), dim=1)

        output_node_features = nodes_time_gap_agg + self.node_raw_features[torch.from_numpy(node_ids)]

        node_embeddings = self.output_layer(
            torch.cat([combined_features, output_node_features], dim=1))

        return node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
