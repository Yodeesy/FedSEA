# datasets/dataset_loader.py
import os
import os.path as osp
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from ogb.nodeproppred import PygNodePropPredDataset


def rand_train_test_idx(label, train_prop, valid_prop, test_prop):
    """
    Randomly splits indices into train/valid/test based on proportions.
    Returns a dictionary of indices.
    """
    # Identify labeled nodes (ignore -1 if any)
    labeled_nodes = torch.where(label != -1)[0]
    n = labeled_nodes.shape[0]

    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)
    test_num = int(n * test_prop)

    # Permute indices
    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num: train_num + valid_num]
    test_indices = perm[train_num + valid_num: train_num + valid_num + test_num]

    return {
        'train': train_indices,
        'valid': val_indices,
        'test': test_indices
    }


def index_to_mask(splits_dict, num_nodes):
    """
    Converts indices dictionary to boolean masks.
    """
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[splits_dict['train']] = True
    val_mask[splits_dict['valid']] = True
    test_mask[splits_dict['test']] = True

    return train_mask, val_mask, test_mask


def load_dataset(train_val_test_split, root_dir, dataset_name):
    """
    Unified Data Loader.
    Supported Datasets:
    - Planetoid: cora, citeseer, pubmed
    - Coauthor: cs, physics
    - Amazon: computers, photo
    - OGB: ogbn-arxiv
    """

    # Validate Dataset Name
    supported_datasets = {
        'cora', 'citeseer', 'pubmed',
        'cs', 'physics',
        'computers', 'photo',
        'ogbn-arxiv'
    }
    assert dataset_name in supported_datasets, f"Invalid dataset: {dataset_name}"

    # Default split if not provided correctly
    if len(train_val_test_split) != 3:
        print("Warning: args.train_val_test_split != 3, using default [0.6, 0.2, 0.2]")
        train_val_test_split = [0.6, 0.2, 0.2]

    train_prop, valid_prop, test_prop = train_val_test_split

    print(f"[Dataset] Loading {dataset_name} from {root_dir}...")

    # 1. Planetoid Datasets (Cora, Citeseer, Pubmed)
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root_dir, name=dataset_name)
        data = dataset[0]

    # 2. Amazon Datasets (Computers, Photo)
    elif dataset_name in ['computers', 'photo']:
        dataset = Amazon(root=root_dir, name=dataset_name)
        data = dataset[0]

    # 3. Coauthor Datasets (CS, Physics)
    elif dataset_name in ['cs', 'physics']:
        dataset = Coauthor(root=root_dir, name=dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]

    # 4. OGB Datasets (Arxiv)
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=dataset_name, root=root_dir)
        data = dataset[0]
        # OGB labels are (N, 1), need squeeze to (N,)
        data.y = data.y.squeeze()

        # OGB has its own fixed split, usually we respect it or create new masks
        # Here we overwrite with random split for FL simulation consistency
        # If you want OGB fixed split, use dataset.get_idx_split()

    # --- Common Post-Processing: Generate Random Splits ---
    # Many datasets (Amazon, Coauthor) don't have public splits.
    # We generate random splits based on proportions.

    # Check if masks already exist and if we want to overwrite them
    # For FedSEA simulation, random split ensures IID/Non-IID control later

    splits_dict = rand_train_test_idx(
        data.y,
        train_prop=train_prop,
        valid_prop=valid_prop,
        test_prop=test_prop
    )

    data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_dict, data.num_nodes)

    return data