import os
import random

import numpy as np


def set_random_seed(
    random_seed: int = 42,
):
    """
    Set a global random seed for reproducibility across various libraries.

    Args:
        random_seed (int): Seed value to be set.
    """
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
