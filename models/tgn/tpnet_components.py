"""
Model components from lxd99/TGB_TPNet, copied verbatim for version stability.

Source: https://github.com/lxd99/TGB_TPNet
Files:  models/MemoryModel.py, models/modules.py, utils/utils.py (NeighborSampler)

Only the components needed for MemoryModel with model_name='TGN' are included.
PINT, NAT, FreqMerge, and numba-dependent code are excluded.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Callable


# ── Time Encoder ──────────────────────────────────────────────────────────────

class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))
        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        # timestamps: (batch, seq_len)
        timestamps = timestamps.unsqueeze(dim=2)
        return torch.cos(self.w(timestamps))


# ── MergeLayer ────────────────────────────────────────────────────────────────

class MergeLayer(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        x = torch.cat([input_1, input_2], dim=1)
        return self.fc2(self.act(self.fc1(x)))


# ── MultiHeadAttention ────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads
        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim
        self.head_dim = self.query_dim // num_heads
        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.scaling_factor = self.head_dim ** -0.5
        self.layer_norm = nn.LayerNorm(self.query_dim)
        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, node_time_features, neighbor_node_features,
                neighbor_node_time_features, neighbor_node_edge_features, neighbor_masks):
        node_features = torch.unsqueeze(node_features, dim=1)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attention = torch.einsum('bhld,bhnd->bhln', query, key) * self.scaling_factor
        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)
        attention = attention.masked_fill(attention_mask, -1e10)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)
        output = self.dropout(self.residual_fc(attention_output))
        output = self.layer_norm(output + residual)
        return output.squeeze(dim=1), attention_scores.squeeze(dim=2)


# ── LinkPredictor ─────────────────────────────────────────────────────────────

class LinkPredictor(nn.Module):
    """
    Their LinkPredictor_v1 with random_projections=None and not_encode=False.
    MLP: concat(src_emb, dst_emb) → hidden → 1.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, src_node_embeddings: torch.Tensor, dst_node_embeddings: torch.Tensor):
        x = torch.cat([src_node_embeddings, dst_node_embeddings], dim=1)
        return self.fc2(self.act(self.fc1(x)))


# ── NeighborSampler ───────────────────────────────────────────────────────────

class NeighborSampler:
    """
    Stores the full sorted interaction history for each node.
    Uses binary search to return neighbors before a given timestamp.
    """

    def __init__(self, adj_list: list, sample_neighbor_strategy: str = 'recent', seed: int = None):
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        for per_node_neighbors in adj_list:
            sorted_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_neighbors]))

        if seed is not None:
            self.random_state = np.random.RandomState(seed)

    def find_neighbors_before(self, node_id: int, interact_time: float):
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)
        return (self.nodes_neighbor_ids[node_id][:i],
                self.nodes_edge_ids[node_id][:i],
                self.nodes_neighbor_times[node_id][:i])

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                  num_neighbors: int = 20):
        nodes_neighbor_ids   = np.zeros((len(node_ids), num_neighbors), dtype=np.int64)
        nodes_edge_ids       = np.zeros((len(node_ids), num_neighbors), dtype=np.int64)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors), dtype=np.float64)

        for idx, (node_id, t) in enumerate(zip(node_ids, node_interact_times)):
            nbr_ids, nbr_eids, nbr_times = self.find_neighbors_before(node_id, t)
            if len(nbr_ids) == 0:
                continue

            if self.sample_neighbor_strategy == 'recent':
                nbr_ids   = nbr_ids[-num_neighbors:]
                nbr_eids  = nbr_eids[-num_neighbors:]
                nbr_times = nbr_times[-num_neighbors:]
                nodes_neighbor_ids[idx,   num_neighbors - len(nbr_ids):]   = nbr_ids
                nodes_edge_ids[idx,       num_neighbors - len(nbr_eids):]  = nbr_eids
                nodes_neighbor_times[idx, num_neighbors - len(nbr_times):] = nbr_times
            else:
                # uniform sampling
                if self.seed is None:
                    sampled = np.random.choice(len(nbr_ids), num_neighbors)
                else:
                    sampled = self.random_state.choice(len(nbr_ids), num_neighbors)
                sampled = sampled[nbr_times[sampled].argsort()]
                nodes_neighbor_ids[idx]   = nbr_ids[sampled]
                nodes_edge_ids[idx]       = nbr_eids[sampled]
                nodes_neighbor_times[idx] = nbr_times[sampled]

        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


# ── Memory Modules ────────────────────────────────────────────────────────────

class MessageAggregator(nn.Module):
    def __init__(self):
        super(MessageAggregator, self).__init__()

    def aggregate_messages(self, node_ids: np.ndarray, node_raw_messages: dict):
        unique_node_messages, unique_node_timestamps, to_update_node_ids, node_indexes = [], [], [], []
        for node_index, node_id in enumerate(node_ids):
            if len(node_raw_messages[node_id]) > 0:
                node_indexes.append(node_index)
                to_update_node_ids.append(node_id)
                unique_node_messages.append(node_raw_messages[node_id][-1][0])
                unique_node_timestamps.append(node_raw_messages[node_id][-1][1])
        to_update_node_ids   = np.array(to_update_node_ids)
        unique_node_messages = torch.stack(unique_node_messages, dim=0) if unique_node_messages else torch.Tensor([])
        unique_node_timestamps = np.array(unique_node_timestamps)
        node_indexes = np.array(node_indexes)
        return to_update_node_ids, unique_node_messages, unique_node_timestamps, node_indexes


class MemoryBank(nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int):
        super(MemoryBank, self).__init__()
        self.num_nodes  = num_nodes
        self.memory_dim = memory_dim
        self.node_memories           = nn.Parameter(torch.zeros((num_nodes, memory_dim)), requires_grad=False)
        self.node_last_updated_times = nn.Parameter(torch.zeros(num_nodes), requires_grad=False)
        self.node_raw_messages  = defaultdict(list)
        self.dirty_message_nodes = set()
        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        self.node_memories.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.node_raw_messages  = defaultdict(list)
        self.dirty_message_nodes = set()

    def get_memories(self, node_ids: np.ndarray):
        return self.node_memories[torch.from_numpy(node_ids)].clone()

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        self.node_memories[torch.from_numpy(node_ids)] = updated_node_memories

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):
        return self.node_last_updated_times[torch.from_numpy(unique_node_ids)].clone()

    def store_node_raw_messages(self, node_ids: np.ndarray, new_node_raw_messages: dict):
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])
        self.dirty_message_nodes |= set(node_ids.tolist())

    def clear_node_raw_messages(self, node_ids: np.ndarray):
        for node_id in node_ids:
            self.node_raw_messages[node_id] = []

    def backup_memory_bank(self):
        cloned = {nid: [(m[0].clone(), m[1].copy()) for m in msgs]
                  for nid, msgs in self.node_raw_messages.items()}
        return (self.node_memories.data.clone(),
                self.node_last_updated_times.data.clone(),
                cloned,
                self.dirty_message_nodes.copy())

    def reload_memory_bank(self, backup):
        self.node_memories.data, self.node_last_updated_times.data = backup[0].clone(), backup[1].clone()
        self.dirty_message_nodes = backup[3].copy()
        self.node_raw_messages = defaultdict(list)
        for nid, msgs in backup[2].items():
            self.node_raw_messages[nid] = [(m[0].clone(), m[1].copy()) for m in msgs]

    def detach_memory_bank(self):
        self.node_memories.detach_()
        for node_id in self.dirty_message_nodes:
            self.node_raw_messages[node_id] = [
                (m[0].detach(), m[1]) for m in self.node_raw_messages[node_id]
            ]
        self.dirty_message_nodes = set()


class MemoryUpdater(nn.Module):
    def __init__(self, memory_bank: MemoryBank):
        super(MemoryUpdater, self).__init__()
        self.memory_bank = memory_bank

    def update_memories(self, unique_node_ids, unique_node_messages, unique_node_timestamps):
        if len(unique_node_ids) <= 0:
            return
        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item()
        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)
        updated = self.memory_updater(unique_node_messages, node_memories)
        self.memory_bank.set_memories(node_ids=unique_node_ids, updated_node_memories=updated)
        self.memory_bank.node_last_updated_times[torch.from_numpy(unique_node_ids)] = \
            torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

    def get_updated_memories(self, unique_node_ids, unique_node_messages, unique_node_timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory_bank.node_memories.new_empty((0, self.memory_bank.memory_dim))
        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item()
        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)
        return self.memory_updater(unique_node_messages, node_memories)


class GRUMemoryUpdater(MemoryUpdater):
    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        super().__init__(memory_bank)
        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)


# ── Graph Attention Embedding ─────────────────────────────────────────────────

class GraphAttentionEmbedding(nn.Module):
    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler,
                 time_encoder, node_feat_dim, edge_feat_dim, time_feat_dim, output_dim,
                 num_layers=2, num_heads=2, dropout=0.1):
        super().__init__()
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler  = neighbor_sampler
        self.time_encoder      = time_encoder
        self.node_feat_dim     = node_feat_dim
        self.edge_feat_dim     = edge_feat_dim
        self.time_feat_dim     = time_feat_dim
        self.output_dim        = output_dim
        self.num_layers        = num_layers
        self.num_heads         = num_heads
        self.dropout           = dropout

        self.temporal_conv_layers = nn.ModuleList([
            MultiHeadAttention(node_feat_dim=output_dim, edge_feat_dim=edge_feat_dim,
                               time_feat_dim=time_feat_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.merge_layers = nn.ModuleList([
            MergeLayer(input_dim1=self.temporal_conv_layers[-1].query_dim,
                       input_dim2=output_dim, hidden_dim=output_dim, output_dim=output_dim)
            for _ in range(num_layers)
        ])

        assert node_feat_dim <= output_dim
        self.projection_layer = nn.Linear(node_feat_dim, output_dim, bias=True) if node_feat_dim < output_dim else None

    def compute_node_temporal_embeddings(self, get_updated_memories: Callable, node_ids: np.ndarray,
                                         node_interact_times: np.ndarray, current_layer_num: int,
                                         num_neighbors: int = 20):
        assert current_layer_num >= 0
        device = self.node_raw_features.device

        node_time_features = self.time_encoder(
            torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device))

        node_features, _ = get_updated_memories(node_ids)
        if self.projection_layer is None:
            node_features = node_features + self.node_raw_features[torch.from_numpy(node_ids)]
        else:
            node_features = node_features + self.projection_layer(self.node_raw_features[torch.from_numpy(node_ids)])

        if current_layer_num == 0:
            return node_features

        node_conv_features = self.compute_node_temporal_embeddings(
            get_updated_memories, node_ids, node_interact_times, current_layer_num - 1, num_neighbors)

        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(
                node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors)

        neighbor_conv_features = self.compute_node_temporal_embeddings(
            get_updated_memories, neighbor_node_ids.flatten(), neighbor_times.flatten(),
            current_layer_num - 1, num_neighbors)
        neighbor_conv_features = neighbor_conv_features.reshape(len(node_ids), num_neighbors, self.output_dim)

        neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times
        neighbor_time_features = self.time_encoder(
            torch.from_numpy(neighbor_delta_times).float().to(device))
        neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]

        output, _ = self.temporal_conv_layers[current_layer_num - 1](
            node_features=node_conv_features,
            node_time_features=node_time_features,
            neighbor_node_features=neighbor_conv_features,
            neighbor_node_time_features=neighbor_time_features,
            neighbor_node_edge_features=neighbor_edge_features,
            neighbor_masks=neighbor_node_ids,
        )
        return self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_features)


# ── MemoryModel ───────────────────────────────────────────────────────────────

class MemoryModel(torch.nn.Module):
    """
    TGN memory model from lxd99/TGB_TPNet.
    Only supports model_name='TGN' (GRU memory + graph attention embedding).
    """

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                 neighbor_sampler: NeighborSampler, output_dim: int, time_feat_dim: int,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu'):
        super(MemoryModel, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.output_dim    = output_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers    = num_layers
        self.num_heads     = num_heads
        self.dropout       = dropout
        self.device        = device

        self.num_nodes  = self.node_raw_features.shape[0]
        self.memory_dim = output_dim
        # message dim = src_mem + dst_mem + time_enc + edge_feat
        self.message_dim = self.memory_dim + self.memory_dim + time_feat_dim + self.edge_feat_dim

        self.time_encoder       = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=True)
        self.message_aggregator = MessageAggregator()
        self.memory_bank        = MemoryBank(num_nodes=self.num_nodes, memory_dim=self.memory_dim)
        self.memory_updater     = GRUMemoryUpdater(memory_bank=self.memory_bank,
                                                   message_dim=self.message_dim,
                                                   memory_dim=self.memory_dim)
        self.embedding_module   = GraphAttentionEmbedding(
            node_raw_features=self.node_raw_features,
            edge_raw_features=self.edge_raw_features,
            neighbor_sampler=neighbor_sampler,
            time_encoder=self.time_encoder,
            node_feat_dim=self.node_feat_dim,
            edge_feat_dim=self.edge_feat_dim,
            time_feat_dim=time_feat_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, edge_ids: np.ndarray = None,
                                                 edges_are_positive: bool = True, num_neighbors: int = 20):
        node_ids = np.concatenate([src_node_ids, dst_node_ids])

        updated_node_memories, updated_node_last_updated_times = self.get_updated_memories(node_ids)

        node_embeddings = self.embedding_module.compute_node_temporal_embeddings(
            get_updated_memories=self.get_updated_memories,
            node_ids=node_ids,
            node_interact_times=np.concatenate([node_interact_times, node_interact_times]),
            current_layer_num=self.num_layers,
            num_neighbors=num_neighbors,
        )

        src_embs = node_embeddings[:len(src_node_ids)]
        dst_embs = node_embeddings[len(src_node_ids):]

        if edges_are_positive:
            assert edge_ids is not None
            self.update_memories(node_ids)
            self.memory_bank.clear_node_raw_messages(node_ids)

            unique_src_ids, new_src_msgs = self.compute_new_node_raw_messages(
                src_node_ids, dst_node_ids, dst_embs, node_interact_times, edge_ids)
            unique_dst_ids, new_dst_msgs = self.compute_new_node_raw_messages(
                dst_node_ids, src_node_ids, src_embs, node_interact_times, edge_ids)

            self.memory_bank.store_node_raw_messages(unique_src_ids, new_src_msgs)
            self.memory_bank.store_node_raw_messages(unique_dst_ids, new_dst_msgs)

        return src_embs, dst_embs

    def get_updated_memories(self, node_ids: np.ndarray):
        unique_node_ids, inverse_node_ids = np.unique(node_ids, return_inverse=True)
        node_memories           = self.memory_bank.get_memories(unique_node_ids)
        node_last_updated_times = self.memory_bank.get_node_last_updated_times(unique_node_ids)

        to_update_ids, unique_msgs, unique_ts, node_indexes = self.message_aggregator.aggregate_messages(
            unique_node_ids, self.memory_bank.node_raw_messages)

        node_memories[node_indexes] = self.memory_updater.get_updated_memories(
            to_update_ids, unique_msgs, unique_ts)
        node_last_updated_times[node_indexes] = torch.from_numpy(unique_ts).float().to(self.device)

        return node_memories[inverse_node_ids], node_last_updated_times[inverse_node_ids]

    def update_memories(self, node_ids: np.ndarray):
        unique_node_ids = np.unique(node_ids)
        to_update_ids, unique_msgs, unique_ts, _ = self.message_aggregator.aggregate_messages(
            unique_node_ids, self.memory_bank.node_raw_messages)
        self.memory_updater.update_memories(to_update_ids, unique_msgs, unique_ts)

    def compute_new_node_raw_messages(self, src_node_ids, dst_node_ids, dst_node_embeddings,
                                      node_interact_times, edge_ids):
        src_memories   = self.memory_bank.get_memories(src_node_ids)
        dst_memories   = self.memory_bank.get_memories(dst_node_ids)
        delta_times    = torch.from_numpy(node_interact_times).float().to(self.device) - \
                         self.memory_bank.node_last_updated_times[torch.from_numpy(src_node_ids)]
        delta_features = self.time_encoder(delta_times.unsqueeze(1)).reshape(len(src_node_ids), -1)
        edge_features  = self.edge_raw_features[torch.from_numpy(edge_ids)]

        raw_messages   = torch.cat([src_memories, dst_memories, delta_features, edge_features], dim=1)

        new_node_raw_messages = defaultdict(list)
        for i in range(len(src_node_ids)):
            new_node_raw_messages[src_node_ids[i]].append(
                (raw_messages[i].clone(), node_interact_times[i].copy()))

        return np.unique(src_node_ids), new_node_raw_messages

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.embedding_module.neighbor_sampler = neighbor_sampler


def compute_src_dst_node_time_shifts(src_node_ids, dst_node_ids, node_interact_times):
    """Compute mean/std of time shifts for src and dst nodes (used for JODIE, not needed for TGN)."""
    src_last, dst_last = {}, {}
    src_shifts, dst_shifts = [], []
    for src, dst, t in zip(src_node_ids, dst_node_ids, node_interact_times):
        src_shifts.append(t - src_last.get(src, 0))
        dst_shifts.append(t - dst_last.get(dst, 0))
        src_last[src] = t
        dst_last[dst] = t
    return (np.mean(src_shifts), np.std(src_shifts),
            np.mean(dst_shifts), np.std(dst_shifts))
