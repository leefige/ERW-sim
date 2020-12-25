import os
import shutil
import sys
from multiprocessing import Process

import numpy as np

from erw import ERW

MAX_STEP = 1e4
N_SAMPLES = 2048
TMP_DIR = "tmp_q3"
RES_DIR = "res_q3"

def sim(p, q):
    elephant = ERW(p, q)
    for _ in range(int(MAX_STEP)):
        elephant.walk()
    return elephant.where()

def worker_func(pid, num, p_, q_):
    print(p_, q_, num, pid)
    res = []
    for i in range(num):
        res.append(sim(p_/100, q_/100))
    res = np.array(res)
    np.save(f"{TMP_DIR}/{p_}_{q_}_{pid}.npy", res)

if __name__ == '__main__':
    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.mkdir(TMP_DIR)
    os.makedirs(RES_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        n_proc = int(sys.argv[1])
    else:
        n_proc = 8

    print(f"Num process: {n_proc}")

    for p_ in [20, 30, 40, 50, 60, 75, 80, 90]:
        for q_ in [25, 50, 75]:
            # divide
            batch = N_SAMPLES // n_proc
            procs = []
            for pid in range(n_proc):
                p = Process(target=worker_func, args=(pid, batch, p_, q_))
                procs.append(p)
                p.start()
            for p in procs:
                p.join()
    
            # merge
            points = []
            for pid in range(n_proc):
                points.append(np.load(f"{TMP_DIR}/{p_}_{q_}_{pid}.npy"))
            res = np.concatenate(points)
            np.save(f"{RES_DIR}/{p_}_{q_}.npy", res)

    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)
