import random

import numpy as np
import torch


class Randomer:

    @staticmethod
    def set_seed(seed: int = 9):
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


if __name__ == '__main__':
    Randomer.set_seed()

