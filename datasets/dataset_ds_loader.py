# datasets/dataset_ds_loader.py
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def partition_by_node(data, clients_nodes):
    """
    Partitions the global graph data into local subgraphs for each client.

    Args:
        data (Data): The global PyG Data object.
        clients_nodes (list of list): A list where each element is a list of node IDs for a client.

    Returns:
        list[Data]: A list of subgraphs, one for each client.
    """
    clients_data = []

    for client_id, nodes in enumerate(clients_nodes):
        # Convert nodes to tensor for PyG utils
        nodes_tensor = torch.tensor(nodes, dtype=torch.long)

        # 1. Extract Subgraph (Vectorized & Fast)
        # relabel_nodes=True ensures the new node indices start from 0 to len(nodes)-1
        # This replaces the slow python loop
        edge_index, _ = subgraph(
            nodes_tensor,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )

        # 2. Slice Features & Labels
        sub_x = data.x[nodes]
        sub_y = data.y[nodes]

        sub_data = Data(x=sub_x, edge_index=edge_index, y=sub_y)

        # 3. Slice Masks (Train/Val/Test) if they exist
        if hasattr(data, "train_mask") and data.train_mask is not None:
            sub_data.train_mask = data.train_mask[nodes]

        if hasattr(data, "val_mask") and data.val_mask is not None:
            sub_data.val_mask = data.val_mask[nodes]

        if hasattr(data, "test_mask") and data.test_mask is not None:
            sub_data.test_mask = data.test_mask[nodes]

        # Optional: Add metadata
        sub_data.num_nodes = len(nodes)

        clients_data.append(sub_data)

    return clients_data