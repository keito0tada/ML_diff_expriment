import datetime
import random

import numpy as np
import torch


def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


NOW = now()
