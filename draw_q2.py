import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma

from q2 import ITERS

_M_ = 5

FIG_DIR = "./fig"
os.makedirs(FIG_DIR, exist_ok=True)
FIG_DIR = "./fig/q2"
os.makedirs(FIG_DIR, exist_ok=True)


if __name__ == '__main__':
    f = np.load(f"q2_{_M_}.npz", allow_pickle=True)
    data = f['arr_0'].item()
    f.close()

    print(data['0_0_10'].shape)
    print(data['0_0_10'])
    print()
    print(data['1_1_2'].shape)
    print(data['1_1_2'])
