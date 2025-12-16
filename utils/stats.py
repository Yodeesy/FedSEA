# utils/stats.py
import numpy as np


def graph_structural_entropy(adj):
    """
    Compute 1D Structural Entropy (SE) of a graph.
    Reference: "SE-GSL: Structure Learning via Structural Entropy", AAAI 2023.

    Formula:
        H(G) = - Σ (d_v / Vol(G)) * log(d_v / Vol(G))
        where d_v is the degree of node v, and Vol(G) is the sum of degrees (2 * num_edges).

    We return the normalized entropy: S_norm = H(G) / log(N).
    High entropy => Random/Complex topology.
    Low entropy => Structured/Regular topology.

    Args:
        adj: Adjacency matrix (numpy.ndarray or scipy.sparse, N x N).

    Returns:
        normalized_entropy (float): Value in range [0, 1].
    """
    # Convert to dense if sparse
    if not isinstance(adj, np.ndarray):
        try:
            adj = adj.toarray()
        except:
            # Fallback for other sparse formats
            pass

    # Degree vector: d_v = sum of row
    deg = np.sum(adj, axis=1)

    # Vol(G) = sum(d_v)
    vol = np.sum(deg)

    # Handle edge case: Empty graph
    if vol == 0:
        return 0.0

    # Probability distribution: p_v = d_v / Vol(G)
    p = deg / vol

    # Filter non-zero probabilities to avoid log(0)
    p = p[p > 0]

    # Compute Shannon Entropy
    entropy = -np.sum(p * np.log(p))

    # Normalize by log(N) to make it scale-invariant
    N = adj.shape[0]
    if N <= 1:
        return 0.0

    norm_entropy = entropy / np.log(N)

    # Numerical safety clip
    return float(np.clip(norm_entropy, 0.0, 1.0))


def compute_class_prototypes(features, labels, num_classes):
    """
    Compute class-wise prototypes (mean embeddings) for FedSEA.
    Used for:
    1. Conditioning the generator (Generative FL).
    2. Regularizing local training (FedProto style).

    Args:
        features: Node features (N, d) numpy array.
        labels: Node labels (N,) numpy array.
        num_classes: Total number of classes.

    Returns:
        prototypes: (num_classes, d) numpy array.
                    Missing classes are filled with zeros.
    """
    d = features.shape[1]
    prototypes = np.zeros((num_classes, d), dtype=np.float32)

    for c in range(num_classes):
        # Extract features for class c
        idx = np.where(labels == c)[0]

        if len(idx) == 0:
            # Handle missing class (Non-IID scenario)
            # Prototype remains 0 vector
            continue

        # Compute mean
        mu = features[idx].mean(axis=0)

        # Optional: L2 Normalize prototypes
        # This helps in cosine similarity calculations later
        norm = np.linalg.norm(mu) + 1e-12
        prototypes[c] = mu / norm

    return prototypes