# utils/set_seed.py
import random
import torch
import numpy as np
import os


def set_random_seed(seed: int = 0):
    """
    Sets random seeds for reproducibility across various libraries.
    Includes flags for PyTorch's CuDNN backend to ensure deterministic behavior.

    Args:
        seed (int): The seed value to set.
    """
    # 1. Python built-in random
    random.seed(seed)

    # 2. Numpy
    np.random.seed(seed)

    # 3. PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # 4. CuDNN Determinism (Critical for exact reproducibility)
    # Note: This might slightly impact performance but guarantees consistent results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5. OS Environment (Optional, for hash stability)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"[System] Random seed set to: {seed}")