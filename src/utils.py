import json
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(data, path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
