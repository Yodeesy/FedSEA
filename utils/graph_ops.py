# utils/graph_ops.py
import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Optional, Union


def softmax_weights(entropy_list: Union[List[float], np.ndarray], tau: float = 1.0) -> np.ndarray:
    """
    Computes Softmax weights based on structural entropy.
    Lower entropy indicates higher structural reliability, resulting in higher weights.

    Formula: w_i = exp(-S_i / tau) / sum(exp(-S_j / tau))

    Args:
        entropy_list: List or array of entropy values.
        tau: Temperature parameter. Higher tau -> softer distribution (closer to uniform).

    Returns:
        np.ndarray: Normalized weights summing to 1.0.
    """
    entropies = np.array(entropy_list)
    # Use negative entropy because lower entropy = better structure = higher weight
    e = np.exp(-entropies / float(tau))
    w = e / (e.sum() + 1e-12)  # Add epsilon to avoid division by zero
    return w.astype(float)


def fuse_graphs(pseudo_graphs: List[Data],
                alphas: List[float],
                device: Optional[torch.device] = None) -> Optional[Data]:
    """
    [Unified Fusion Strategy: Disjoint Concatenation]

    Aggregates multiple generated subgraphs into a single large disjoint graph.
    This effectively reconstructs the global graph distribution from local client views.

    For example:
    - Arxiv: Aggregates client subgraphs to restore global information (e.g., 20k -> 170k nodes).
    - Cora: Acts as data augmentation to improve generalization.

    Args:
        pseudo_graphs: List of PyG Data objects (generated subgraphs).
        alphas: List of importance weights for each subgraph (sum=1.0).
        device: Target device for the fused graph.

    Returns:
        Data: A single PyG Data object containing the fused global graph.
    """
    if not pseudo_graphs:
        return None

    if device is None:
        device = pseudo_graphs[0].x.device

    all_x = []
    all_edge_index = []
    all_edge_attr = []

    # Offset is used to stack graphs disjointly (preventing index collision)
    # Graph 2's indices will start where Graph 1's indices ended.
    current_offset = 0

    # Pre-calculate scaling factor to normalize edge weights around 1.0
    # Since sum(alphas)=1, avg(alpha)=1/N. We scale by N so avg(weight)=1.
    num_graphs = len(pseudo_graphs)

    for i, g in enumerate(pseudo_graphs):
        # 1. Pruning: Skip graphs with negligible weights to save memory
        if alphas[i] < 1e-4:
            continue

        # 2. Features
        x_curr = g.x.to(device)
        all_x.append(x_curr)

        # 3. Edge Index (with Offset Shift)
        if g.edge_index is not None and g.edge_index.numel() > 0:
            edge_index = g.edge_index.to(device)
            edge_index_shifted = edge_index + current_offset
            all_edge_index.append(edge_index_shifted)

            # 4. Edge Weights (Scaled by Client Importance)
            num_edges = edge_index.size(1)

            # Logic: Alpha reflects sample importance.
            scale_factor = float(alphas[i] * num_graphs)

            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                # If generator produced weights, scale them
                weight = g.edge_attr.view(-1).to(device) * scale_factor
            else:
                # Otherwise, assign uniform importance based on alpha
                weight = torch.full((num_edges,), scale_factor, device=device)

            all_edge_attr.append(weight)

        # 5. Update Offset for the next graph
        current_offset += x_curr.size(0)

    # 6. Physical Concatenation
    if len(all_x) > 0:
        global_x = torch.cat(all_x, dim=0)
    else:
        return None

    if len(all_edge_index) > 0:
        global_edge_index = torch.cat(all_edge_index, dim=1)
        global_edge_attr = torch.cat(all_edge_attr, dim=0)
    else:
        # Handle edge case: No edges in any subgraph
        global_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        global_edge_attr = torch.empty((0,), device=device)

    return Data(x=global_x, edge_index=global_edge_index, edge_attr=global_edge_attr)