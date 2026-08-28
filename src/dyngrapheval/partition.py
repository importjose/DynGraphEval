"""
Source-node partitioning for federated models.

Both FederatedTGN and FedLink use the same data partitioning strategy:
  - Collect all unique source nodes from training data
  - Sort them by node ID (deterministic)
  - Split into equal-sized groups with numpy.array_split
  - Each client owns all edges where src belongs to its group

This mirrors the partitioning used during federated training, so evaluation
sees the same client-data assignments as training.
"""

import numpy as np
import torch
from torch_geometric.data import TemporalData


def partition_by_source(
    data: TemporalData,
    src_sets: list[set],
) -> list[TemporalData]:
    """
    Filter a TemporalData object into per-client subsets based on source node.

    Parameters
    ----------
    data     : TemporalData  — the full dataset to split
    src_sets : list[set]    — one set of source node IDs per client

    Returns
    -------
    list[TemporalData] : one TemporalData per client containing only their edges
    """
    src_cpu = data.src.cpu().numpy()
    partitions = []
    for src_set in src_sets:
        # Create a boolean mask: True where this edge's src belongs to this client
        mask = torch.tensor(
            [int(s) in src_set for s in src_cpu], dtype=torch.bool
        )
        partitions.append(data[mask])
    return partitions


def build_src_sets(train_data: TemporalData, num_clients: int) -> list[set]:
    """
    Build per-client source-node assignment from the training data.

    Sorts unique source node IDs and splits them equally across clients.
    Using sorted IDs (not random) makes the assignment deterministic and
    reproducible without a fixed seed.

    Parameters
    ----------
    train_data  : TemporalData  — training edges (determines node ownership)
    num_clients : int

    Returns
    -------
    list[set] : src_sets[c] is the set of source node IDs belonging to client c
    """
    # Get all unique source nodes seen during training, sorted by ID
    all_src = torch.sort(train_data.src.unique()).values.cpu().numpy()

    # Split into equal-sized chunks (last chunk may be slightly larger)
    splits = np.array_split(all_src, num_clients)

    return [set(chunk.tolist()) for chunk in splits]
