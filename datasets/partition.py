# datasets/partition.py
import numpy as np
import torch


def dirichlet_partitioner(data, num_clients, alpha, least_samples=10, max_attempts=100):
    """
    Partitions graph nodes among clients using a Dirichlet distribution on labels.
    This simulates Non-IID label skew.

    Args:
        data: PyG Data object (must contain data.y)
        num_clients: Number of clients
        alpha: Concentration parameter (smaller alpha = more Non-IID)
        least_samples: Minimum number of samples per client
        max_attempts: Max retries to satisfy least_samples constraint

    Returns:
        list[list[int]]: A list of node indices for each client.
    """
    graph_labels = data.y.numpy()
    num_nodes = graph_labels.shape[0]
    num_classes = len(np.unique(graph_labels))

    # Print global stats
    unique_labels, label_counts = np.unique(graph_labels, return_counts=True)
    print(f"Num Classes: {len(unique_labels)}")
    print(f"Global Label Dist: {label_counts}")

    client_node_indices = [[] for _ in range(num_clients)]
    min_size = 0
    attempt = 0

    # Try to partition until minimum size constraint is met
    while min_size < least_samples:
        if attempt >= max_attempts:
            print(
                f"[Warning] Dirichlet partition failed after {max_attempts} attempts. Alpha={alpha} might be too small.")
            break

        client_node_indices = [[] for _ in range(num_clients)]

        # Partition each class separately
        for k in range(num_classes):
            # Get all node indices for class k
            idx_k = np.where(graph_labels == k)[0]
            np.random.shuffle(idx_k)

            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

            # Balance check: Adjust proportions to avoid over-assigning to full clients
            # (Simple heuristic to prevent extreme imbalance)
            proportions = np.array([
                p * (len(client_node_indices[i]) < num_nodes / num_clients)
                for i, p in enumerate(proportions)
            ])

            # Normalize
            proportions = proportions / proportions.sum()

            # Calculate split points
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # Split and assign
            idx_split = np.split(idx_k, split_points)
            for i in range(num_clients):
                client_node_indices[i].extend(idx_split[i].tolist())

        # Check minimum client size
        min_size = min([len(c) for c in client_node_indices])
        attempt += 1

    # Sort indices for consistency
    for i in range(num_clients):
        client_node_indices[i].sort()

    return client_node_indices