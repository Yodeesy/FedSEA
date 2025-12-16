import torch
import numpy as np
from torch_geometric.data import Data

def softmax_weights(entropy_list, tau=1.0):
    e = np.exp(-np.array(entropy_list) / float(tau))
    w = e / (e.sum() + 1e-12)
    return w.astype(float)


def fuse_graphs(pseudo_graphs, alphas, device=None):
    """
    [Unified Fusion Strategy: Concatenation]
    逻辑：将所有生成的子图拼接成一个超级大图。
    - Arxiv: 节点数从 2w -> 17w (恢复全图信息，关键！)
    - Cora:  节点数从 2k -> 2w  (数据增强，提升泛化)
    """
    print(f"🔥🔥🔥 [DEBUG] 执行拼接融合！输入子图数量: {len(pseudo_graphs)}")
    if not pseudo_graphs:
        return None

    if device is None:
        device = pseudo_graphs[0].x.device

    all_x = []
    all_edge_index = []
    all_edge_attr = []

    # 偏移量：用来把图2接在图1后面，而不是叠在上面
    current_offset = 0

    for i, g in enumerate(pseudo_graphs):
        # 1. 忽略权重极小的图 (去噪)
        if alphas[i] < 1e-4:
            continue

        # 2. 特征 (Feature)
        x_curr = g.x.to(device)
        all_x.append(x_curr)

        # 3. 边 (Edge Index) - 必须加上偏移量！
        edge_index = g.edge_index.to(device)
        edge_index_shifted = edge_index + current_offset
        all_edge_index.append(edge_index_shifted)

        # 4. 边权重 (Edge Weight)
        num_edges = edge_index.size(1)

        # 逻辑：我们将 alpha 视为样本重要性。
        # 乘以 len(pseudo_graphs) 是为了保持权重的平均量级在 1.0 左右
        scale_factor = float(alphas[i] * len(pseudo_graphs))

        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            # 如果生成器输出了权重，保留并缩放
            weight = g.edge_attr.view(-1).to(device) * scale_factor
        else:
            # 如果没有权重，默认为 1.0 并缩放
            weight = torch.full((num_edges,), scale_factor, device=device)

        all_edge_attr.append(weight)

        # 5. 更新偏移量 (为下一个图做准备)
        current_offset += x_curr.size(0)

    # 6. 物理拼接 (Concatenation)
    # 这步绝对不会爆显存，因为是稀疏操作
    if len(all_x) > 0:
        global_x = torch.cat(all_x, dim=0)
    else:
        return None

    if len(all_edge_index) > 0:
        global_edge_index = torch.cat(all_edge_index, dim=1)
        global_edge_attr = torch.cat(all_edge_attr, dim=0)
    else:
        # 极端情况：没有任何边
        global_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        global_edge_attr = torch.empty((0,), device=device)

    # 返回拼接后的大图
    return Data(x=global_x, edge_index=global_edge_index, edge_attr=global_edge_attr)