"""
TGAT model from lxd99/TGB_TPNet, copied verbatim for version stability.

Source: https://github.com/lxd99/TGB_TPNet
File:   models/TGAT.py

TimeEncoder, MergeLayer, MultiHeadAttention, NeighborSampler, and LinkPredictor
are imported from tpnet_components to avoid duplication. TGAT differs from
GraphMixer in two ways:
  - TimeEncoder is trainable (parameter_requires_grad=True, the default)
  - Aggregation uses multi-head temporal attention instead of MLP-Mixer
"""

import numpy as np
import torch
import torch.nn as nn

from ..tgn.tpnet_components import (
    NeighborSampler,
    TimeEncoder,
    MergeLayer,
    MultiHeadAttention,
    LinkPredictor,
)


class TGAT(nn.Module):

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super(TGAT, self).__init__()

        self.node_raw_features = torch.from_numpy(
            node_raw_features.astype(np.float32)
        ).to(device)
        self.edge_raw_features = torch.from_numpy(
            edge_raw_features.astype(np.float32)
        ).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim  = self.node_raw_features.shape[1]
        self.edge_feat_dim  = self.edge_raw_features.shape[1]
        self.time_feat_dim  = time_feat_dim
        self.output_dim     = output_dim
        self.num_layers     = num_layers
        self.num_heads      = num_heads
        self.dropout        = dropout

        # Trainable time encoder — distinguishes TGAT from GraphMixer
        self.time_encoder = TimeEncoder(
            time_dim=time_feat_dim, parameter_requires_grad=True
        )

        self.temporal_conv_layers = nn.ModuleList([
            MultiHeadAttention(
                node_feat_dim=self.node_feat_dim,
                edge_feat_dim=self.edge_feat_dim,
                time_feat_dim=self.time_feat_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )
        ])
        self.merge_layers = nn.ModuleList([
            MergeLayer(
                input_dim1=self.temporal_conv_layers[-1].query_dim,
                input_dim2=self.node_feat_dim,
                hidden_dim=self.output_dim,
                output_dim=self.output_dim,
            )
        ])

        for _ in range(num_layers - 1):
            self.temporal_conv_layers.append(
                MultiHeadAttention(
                    node_feat_dim=self.output_dim,
                    edge_feat_dim=self.edge_feat_dim,
                    time_feat_dim=self.time_feat_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
            )
            self.merge_layers.append(
                MergeLayer(
                    input_dim1=self.temporal_conv_layers[-1].query_dim,
                    input_dim2=self.node_feat_dim,
                    hidden_dim=self.output_dim,
                    output_dim=self.output_dim,
                )
            )

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
    ):
        src_emb = self.compute_node_temporal_embeddings(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            current_layer_num=self.num_layers,
            num_neighbors=num_neighbors,
        )
        dst_emb = self.compute_node_temporal_embeddings(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            current_layer_num=self.num_layers,
            num_neighbors=num_neighbors,
        )
        return src_emb, dst_emb

    def compute_node_temporal_embeddings(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        current_layer_num: int,
        num_neighbors: int = 20,
    ):
        assert current_layer_num >= 0
        device = self.node_raw_features.device

        # Query node time feature: time interval == 0 for the query itself
        node_time_features = self.time_encoder(
            timestamps=torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device)
        )
        node_raw_features = self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_raw_features

        # Recurse to get embeddings from previous layer
        node_conv_features = self.compute_node_temporal_embeddings(
            node_ids=node_ids,
            node_interact_times=node_interact_times,
            current_layer_num=current_layer_num - 1,
            num_neighbors=num_neighbors,
        )

        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=num_neighbors,
            )

        neighbor_node_conv_features = self.compute_node_temporal_embeddings(
            node_ids=neighbor_node_ids.flatten(),
            node_interact_times=neighbor_times.flatten(),
            current_layer_num=current_layer_num - 1,
            num_neighbors=num_neighbors,
        )
        neighbor_node_conv_features = neighbor_node_conv_features.reshape(
            node_ids.shape[0], num_neighbors, -1
        )

        neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times
        neighbor_time_features = self.time_encoder(
            timestamps=torch.from_numpy(neighbor_delta_times).float().to(device)
        )

        neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]

        output, _ = self.temporal_conv_layers[current_layer_num - 1](
            node_features=node_conv_features,
            node_time_features=node_time_features,
            neighbor_node_features=neighbor_node_conv_features,
            neighbor_node_time_features=neighbor_time_features,
            neighbor_node_edge_features=neighbor_edge_features,
            neighbor_masks=neighbor_node_ids,
        )

        output = self.merge_layers[current_layer_num - 1](
            input_1=output, input_2=node_raw_features
        )
        return output

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
