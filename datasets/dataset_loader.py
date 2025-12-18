# datasets/dataset_loader.py
import os
import os.path as osp
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from ogb.nodeproppred import PygNodePropPredDataset


def get_random_split_masks(num_nodes, labels, train_prop=0.6, valid_prop=0.2, seed=42):
    """
    Generates deterministic random splits based on a seed.
    """
    rs = np.random.RandomState(seed)
    perm = torch.as_tensor(rs.permutation(num_nodes))

    train_num = int(num_nodes * train_prop)
    valid_num = int(num_nodes * valid_prop)

    train_indices = perm[:train_num]
    val_indices = perm[train_num: train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def load_dataset(train_val_test_split, root_dir, dataset_name, seed=42):
    """
    Unified Data Loader enforcing Academic Standards.
    """

    # 1. Validate Dataset Support
    supported_datasets = {
        'cora', 'citeseer', 'pubmed',
        'cs', 'physics',
        'computers', 'photo',
        'ogbn-arxiv'
    }
    assert dataset_name in supported_datasets, f"Invalid dataset: {dataset_name}"

    if len(train_val_test_split) != 3:
        train_val_test_split = [0.6, 0.2, 0.2]

    print(f"[Dataset] Loading {dataset_name} from {root_dir}...")

    # ==============================================================================
    # TYPE 1: Planetoid (Cora, Citeseer, Pubmed)
    # STANDARD: Public Split (Fixed)
    # ==============================================================================
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root_dir, name=dataset_name, split='public', transform=T.NormalizeFeatures())
        data = dataset[0]
        return data

    # ==============================================================================
    # TYPE 2: OGB (ogbn-arxiv)
    # STANDARD: Time-based Split (Fixed)
    # ==============================================================================
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=dataset_name, root=root_dir, transform=T.ToUndirected())
        data = dataset[0]
        data.y = data.y.squeeze()

        # Use OGB Official Split
        split_idx = dataset.get_idx_split()

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True

        return data

    # ==============================================================================
    # TYPE 3: Amazon & Coauthor
    # STANDARD: Random Split (Deterministic)
    # ==============================================================================
    elif dataset_name in ['computers', 'photo']:
        dataset = Amazon(root=root_dir, name=dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name in ['cs', 'physics']:
        dataset = Coauthor(root=root_dir, name=dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]


    data.train_mask, data.val_mask, data.test_mask = get_random_split_masks(
        num_nodes=data.num_nodes,
        labels=data.y,
        train_prop=train_val_test_split[0],
        valid_prop=train_val_test_split[1],
        seed=seed
    )

    return data