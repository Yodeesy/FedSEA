# utils/ot_align.py
import numpy as np
import ot


def compute_structure_matrix(P):
    """
    Computes the structural relationship matrix (Intra-class relationship) for prototypes.
    Used as the 'Structure Cost' (C) in Gromov-Wasserstein.

    Formula: C[i,j] = 1 - CosineSimilarity(P_i, P_j).
    Range: [0, 2] (0 means identical direction, 2 means opposite).

    Args:
        P: Prototype matrix (num_classes, feature_dim).

    Returns:
        C: Structure matrix (num_classes, num_classes).
    """
    # Normalize rows for cosine similarity
    # Add epsilon to avoid division by zero
    P_norm = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)

    # Cosine Similarity = P_norm @ P_norm.T
    sim_matrix = np.dot(P_norm, P_norm.T)

    # Cosine Distance = 1 - Similarity
    return 1 - sim_matrix


def fgw_prototype_alignment(client_protos, alphas, num_classes, feature_dim, max_iter=5, beta=0.5):
    """
    Computes the Global Prototype Barycenter using Fused Gromov-Wasserstein (FGW).

    This function aligns client prototypes into a unified global space by minimizing
    both feature distance (Wasserstein) and structural distortion (Gromov-Wasserstein).

    Args:
        client_protos (list[np.ndarray]): List of client prototype matrices.
        alphas (list[float]): Weights for each client (sum=1).
        num_classes (int): Number of classes.
        feature_dim (int): Dimension of features.
        max_iter (int): Number of barycenter update iterations.
        beta (float): Trade-off parameter for FGW (alpha in OT papers).
                      beta=0   -> Pure Wasserstein (Euclidean alignment only).
                      beta=1   -> Pure Gromov-Wasserstein (Structure alignment only).
                      beta=0.5 -> Fused (Balanced).

    Returns:
        P_global (np.ndarray): The aligned global prototypes (num_classes, feature_dim).
    """
    # 1. Filter invalid data (e.g., from clients that failed to compute protos)
    valid_data = []
    for w, p in zip(alphas, client_protos):
        if p is not None:
            valid_data.append((w, p))

    if not valid_data:
        return np.zeros((num_classes, feature_dim), dtype=np.float32)

    valid_alphas_raw, valid_protos = zip(*valid_data)

    # Normalize weights
    w_sum = sum(valid_alphas_raw)
    valid_alphas = [w / w_sum for w in valid_alphas_raw]

    # Edge case: Single client -> No alignment needed
    if len(valid_protos) == 1:
        return valid_protos[0]

    # Initialize P_global with Simple Weighted Average (SWA) for warm start
    # This usually converges faster than random initialization
    P_global = np.zeros((num_classes, feature_dim), dtype=np.float32)
    for w, p in zip(valid_alphas, valid_protos):
        P_global += w * p

    print(f"  [FGW-OT] Aligning {len(valid_protos)} clients (beta={beta}, iter={max_iter})...")

    # 2. Iterative Barycenter Optimization
    # We assume uniform distribution over classes (class balance prior in latent space)
    dist_p = ot.unif(num_classes)  # Source distribution
    dist_q = ot.unif(num_classes)  # Target distribution

    for it in range(max_iter):
        P_new = np.zeros_like(P_global)
        C_global = compute_structure_matrix(P_global)

        for idx, P_k in enumerate(valid_protos):
            w_k = valid_alphas[idx]

            # A. Compute Cost Matrices
            # M: Feature cost (Euclidean distance between P_k and P_global)
            M = ot.dist(P_k, P_global, metric='euclidean')
            # C_k: Structure cost (Intra-class geometry of Client k)
            C_k = compute_structure_matrix(P_k)

            # B. Solve FGW Transport Plan (T_k)
            T_k = None
            try:
                # ot.gromov.fused_gromov_wasserstein solves:
                # min (1-beta)*<T,M> + beta*<T,C_k,C_global>
                T_k = ot.gromov.fused_gromov_wasserstein(
                    M, C_k, C_global, dist_p, dist_q,
                    loss_fun='square_loss', alpha=beta, verbose=False
                )
            except Exception:
                # Fallback if solver fails (numerical instability)
                T_k = None

            # [Defense] Numerical Stability Check
            if T_k is None or np.isnan(T_k).any():
                # Fallback to Identity (Assume perfectly aligned)
                T_k = np.eye(num_classes) * (1.0 / num_classes)

            # C. Barycentric Mapping (Project P_k to P_global space)
            # P_aligned = (num_classes * T.T) @ P_source
            # Note: num_classes factor compensates for the uniform weight (1/C)
            P_k_aligned = num_classes * np.matmul(T_k.T, P_k)

            # Accumulate weighted contribution
            P_new += w_k * P_k_aligned

        # Update global target
        P_global = P_new

    return P_global