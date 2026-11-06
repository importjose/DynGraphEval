"""
MultiScaleGraphMixer components for evaluation.
"""
import numpy as np
import torch
import torch.nn as nn

from ..tgn.tpnet_components import NeighborSampler, TimeEncoder, LinkPredictor
from ..graphmixer.graphmixer_components import FeedForwardNet, MLPMixer


class MultiScaleGraphMixer(nn.Module):
    """
    Multi-scale GraphMixer: runs K=5, K=20, K=100 streams in parallel,
    combines with a fusion layer. No memory bank, fixed time encoder.
    """

    SCALES = [5, 20, 100]

    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler,
                 time_feat_dim, output_dim, num_layers=2,
                 token_dim_expansion_factor=0.5, channel_dim_expansion_factor=4.0,
                 dropout=0.1, device='cpu', time_gap=2000):
        super().__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = node_raw_features.shape[1]
        self.edge_feat_dim = edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.output_dim = output_dim
        self.num_channels = output_dim
        self.num_layers = num_layers
        self.device = device
        self.time_gap = time_gap

        # Fixed time encoder (not trainable — same as GraphMixer)
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)

        # Shared projection: (edge_feat_dim + time_feat_dim) → num_channels
        self.projection_layer = nn.Linear(self.edge_feat_dim + time_feat_dim, self.num_channels)

        # Per-scale MLP-Mixer stacks (each scale has different num_tokens=K)
        self.scale_mixers = nn.ModuleList([
            nn.ModuleList([
                MLPMixer(
                    num_tokens=K,
                    num_channels=self.num_channels,
                    token_dim_expansion_factor=token_dim_expansion_factor,
                    channel_dim_expansion_factor=channel_dim_expansion_factor,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
            for K in self.SCALES
        ])

        # Fusion: (num_scales * num_channels + node_feat_dim) → output_dim
        fusion_in_dim = self.num_channels * len(self.SCALES) + self.node_feat_dim
        self.fusion_layer = nn.Linear(fusion_in_dim, output_dim)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids,
                                                  node_interact_times, num_neighbors=20,
                                                  time_gap=2000):
        src_emb = self.compute_node_temporal_embeddings(src_node_ids, node_interact_times)
        dst_emb = self.compute_node_temporal_embeddings(dst_node_ids, node_interact_times)
        return src_emb, dst_emb

    def compute_node_temporal_embeddings(self, node_ids, node_interact_times):
        scale_embeddings = []

        for K, mixers in zip(self.SCALES, self.scale_mixers):
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(
                    node_ids=node_ids,
                    node_interact_times=node_interact_times,
                    num_neighbors=K,
                )

            # (batch, K, edge_feat_dim)
            edge_feats = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
            # (batch, K, time_feat_dim)
            time_deltas = torch.from_numpy(
                node_interact_times[:, np.newaxis] - neighbor_times
            ).float().to(self.device)
            time_feats = self.time_encoder(timestamps=time_deltas)
            # zero out padding
            pad_mask = torch.from_numpy(neighbor_node_ids == 0)[:, :, None].to(self.device)
            time_feats.masked_fill_(pad_mask, 0.0)

            # (batch, K, num_channels)
            x = self.projection_layer(torch.cat([edge_feats, time_feats], dim=-1))
            for mixer in mixers:
                x = mixer(x)

            # mean pool → (batch, num_channels)
            scale_embeddings.append(x.mean(dim=1))

        # Node encoder (same as GraphMixer)
        tg_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(
            node_ids=node_ids, node_interact_times=node_interact_times,
            num_neighbors=self.time_gap)
        tg_node_feats = self.node_raw_features[torch.from_numpy(tg_node_ids)]
        valid_mask = torch.from_numpy((tg_node_ids > 0).astype(np.float32))
        valid_mask[valid_mask == 0] = -1e10
        scores = torch.softmax(valid_mask, dim=1).to(self.device)
        node_agg = torch.mean(tg_node_feats * scores.unsqueeze(-1), dim=1)
        node_feats = node_agg + self.node_raw_features[torch.from_numpy(node_ids)]

        # Fuse all scales + node features
        all_feats = torch.cat(scale_embeddings + [node_feats], dim=-1)
        return self.fusion_layer(all_feats)

    def set_neighbor_sampler(self, neighbor_sampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
