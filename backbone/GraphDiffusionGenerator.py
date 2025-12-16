# backbone/GraphDiffusionGenerator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard Sinusoidal Positional Encodings for Time t.
    Based on "Attention Is All You Need".
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaLN(nn.Module):
    """
    [Core Component] Adaptive Layer Normalization (AdaLN)
    Modulates features based on condition (c) via scale (gamma) and shift (beta).
    Formula: AdaLN(x, c) = (1 + gamma(c)) * Norm(x) + beta(c)
    """

    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim, elementwise_affine=False, eps=1e-6)
        # Project condition to scale & shift parameters
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, in_dim * 2)
        )
        # Zero-init ensures identity mapping at the start of training
        with torch.no_grad():
            self.cond_proj[1].weight.zero_()
            self.cond_proj[1].bias.zero_()

    def forward(self, x, cond):
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        return self.layernorm(x) * (1 + scale) + shift


class ResDenoiseBlock(nn.Module):
    """
    [Core Component] Deep Residual Denoising Block
    Structure: Input -> AdaLN -> GELU -> Linear -> AdaLN -> GELU -> Linear -> Output + Input
    """

    def __init__(self, in_dim, cond_dim, hidden_dim, time_dim, dropout=0.1):
        super().__init__()

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim)
        )

        # Layer 1
        self.norm1 = AdaLN(in_dim, cond_dim)
        self.act1 = nn.GELU()
        self.conv1 = nn.Linear(in_dim, hidden_dim)

        # Layer 2
        self.norm2 = AdaLN(hidden_dim, cond_dim)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Linear(hidden_dim, in_dim)

        # Skip connection alignment
        self.skip = nn.Identity() if in_dim == in_dim else nn.Linear(in_dim, in_dim)

    def forward(self, x, t_emb, cond):
        h = x

        # 1. First transformation
        h = self.norm1(h, cond)
        h = self.act1(h)
        h = self.conv1(h)

        # 2. Inject Time Embedding (Additive)
        t_feat = self.time_mlp(t_emb)
        h = h + t_feat

        # 3. Second transformation
        h = self.norm2(h, cond)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip(x) + h


class GraphDiffusionGenerator(nn.Module):
    """
    [Backbone] Graph Diffusion Generator
    Generates node features conditioned on Prototypes and Structural Entropy.
    Reconstructs graph topology via deterministic KNN decoding.
    """

    def __init__(self, device, noise_dim, feature_dim, num_classes,
                 num_timesteps=100,
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.1,
                 args=None):
        super().__init__()
        self.device = device
        self.args = args
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # Determine Condition Dimension
        # Current conditions: Prototypes (feature_dim) + Entropy (1)
        self.cond_dim = feature_dim + 1

        # 1. Input Projection (Noise -> Hidden)
        self.init_proj = nn.Linear(noise_dim, hidden_dim)

        # 2. Time Embeddings
        time_dim = hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # 3. Deep Residual Stack
        self.layers = nn.ModuleList([
            ResDenoiseBlock(
                in_dim=hidden_dim,
                cond_dim=self.cond_dim,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # 4. Output Projection (Hidden -> Feature)
        self.final_norm = AdaLN(hidden_dim, self.cond_dim)
        self.final_proj = nn.Linear(hidden_dim, feature_dim)

    def build_cond_tensor(self, stats, device=None, assigned_labels=None):
        """
        Constructs the condition vector 'c' for the generator.
        Components:
        1. Class Prototypes (Semantic Guidance)
        2. Structural Entropy (Complexity Guidance)
        (Removed: Degree Histogram, Phi)
        """
        device = device or self.device

        prototypes = stats.get("prototypes", None)
        S_k = float(stats.get("S_k", 0.0))
        num_nodes = int(stats.get("num_nodes", 20))

        cond_parts = []

        # A. Prototypes (Instance-level)
        if prototypes is not None:
            p = np.array(prototypes)
            if assigned_labels is not None:
                p_tensor = torch.tensor(p, dtype=torch.float32, device=device)
                cond_proto = p_tensor[assigned_labels]
            else:
                # Fallback: Mean prototype repeated
                p_mean = torch.tensor(p.mean(axis=0), dtype=torch.float32, device=device)
                cond_proto = p_mean.unsqueeze(0).repeat(num_nodes, 1)
        else:
            # Fallback: Zero tensor
            cond_proto = torch.zeros((num_nodes, self.feature_dim), dtype=torch.float32, device=device)
        cond_parts.append(cond_proto)

        # B. Structure Entropy (Global-level)
        sk_t = torch.tensor([S_k], dtype=torch.float32, device=device).unsqueeze(0).repeat(num_nodes, 1)
        cond_parts.append(sk_t)

        return torch.cat(cond_parts, dim=1)

    def forward(self, z, cond):
        """
        Executes the denoising process (Reverse Diffusion).
        x_T (z) -> ... -> x_0 (Features)
        """
        # 1. Initial Projection
        x = self.init_proj(z)
        batch_size = x.size(0)

        # 2. Iterative Denoising
        for i in range(self.num_timesteps, 0, -1):
            t = torch.tensor([i] * batch_size, device=self.device).float()
            t_emb = self.time_embed(t)

            # Pass through ResBlocks
            for layer in self.layers:
                x = layer(x, t_emb, cond)

        # 3. Final Projection
        x = self.final_norm(x, cond)
        x = self.final_proj(x)

        return x

    def generate(self, stats):
        """
        End-to-End Generation Pipeline:
        1. Sample Noise z
        2. Generate Features X via Diffusion
        3. Reconstruct Topology A via KNN
        4. Return Graph Data object
        """
        num_nodes = int(stats.get("num_nodes", 20))
        y_gen = torch.randint(0, self.num_classes, (num_nodes,), device=self.device)

        # Build conditions
        cond = self.build_cond_tensor(stats, self.device, assigned_labels=y_gen)

        # Sample Gaussian Noise
        z = torch.randn((num_nodes, self.noise_dim), device=self.device)

        # 1. Generate Features (GPU)
        X_hat = self.forward(z, cond)

        # 2. Decode Structure (KNN)
        # Determine k (default 5, can be tuned via args)
        k_knn = getattr(self.args, "gen_knn", 5)
        if k_knn < 2: k_knn = 5

        # OOM Protection: Large graphs switch to CPU for KNN
        if num_nodes > 10000:
            x_cpu = X_hat.detach().cpu()
            edge_index = knn_graph(x_cpu, k=k_knn, loop=False).to(self.device)
        else:
            edge_index = knn_graph(X_hat, k=k_knn, loop=False)

        # 3. Wrap into Data object
        data = Data(x=X_hat, edge_index=edge_index)
        data.y = y_gen

        # Assign uniform edge weights (1.0)
        data.edge_attr = torch.ones(edge_index.size(1), device=self.device)

        return data