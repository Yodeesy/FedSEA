# backbone/GCN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Standard Graph Convolutional Network (GCN) implementation.

    Structure:
    - Input Layer
    - (Optional) Hidden Layers
    - Output Layer

    Returns:
        tuple: (penultimate_embeddings, logits)
        This dual return is crucial for methods that need access to features 
        before the final classifier (e.g., for prototype calculation).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

        # Case 1: Multi-layer GCN
        if num_layers > 1:
            # Input Layer: Input -> Hidden
            self.layers.append(GCNConv(input_dim, hidden_dim))

            # Hidden Layers: Hidden -> Hidden
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

            # Output Layer: Hidden -> Output
            self.layers.append(GCNConv(hidden_dim, output_dim))

        # Case 2: Single-layer GCN (Linear Probe on Graph)
        else:
            self.layers.append(GCNConv(input_dim, output_dim))

    def forward(self, data):
        # Unpack data
        x, edge_index = data.x, data.edge_index
        # Support edge_weight if present in data (Good practice)
        edge_weight = getattr(data, 'edge_weight', None)

        # 1. Forward pass through all layers except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. Save the penultimate features (embedding)
        embedding = x

        # 3. Final classification layer
        # Note: We do NOT apply Softmax here, as CrossEntropyLoss expects raw logits
        out = self.layers[-1](x, edge_index, edge_weight)

        # Return tuple: (Features used for Proto/Alignment, Logits for Classification)
        return embedding, out