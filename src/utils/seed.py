import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy, torch (cpu/cuda) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (may reduce speed).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
