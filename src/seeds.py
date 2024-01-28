import os
import torch
import numpy as np
import random

class DeterministicSeed:
    def __init__(self, seed_num=36):
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1' 
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed_num)
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        #torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        self.seed = seed_num
