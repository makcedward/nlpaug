try:
    import torch
except ImportError:
    # No installation required if not using this function
    pass
import numpy as np
import random


class Randomness:
    @staticmethod
    def seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(2021)
        except:
            pass
