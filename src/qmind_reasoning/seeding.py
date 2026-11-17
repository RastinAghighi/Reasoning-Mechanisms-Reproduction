"""Seed every RNG that is available, without requiring heavy dependencies."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (the latter two only if installed).

    ``deterministic=True`` additionally requests deterministic CUDA kernels,
    which can be slower and may error on ops without a deterministic variant.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
