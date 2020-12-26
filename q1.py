import os
import shutil
import sys
from multiprocessing import Process

import numpy as np

from erw import ERW

ITERS = 100
BOUND = 1e4
TMP_DIR = "tmp_q1"

class TooDeepError(RuntimeError):
    pass

def sim(p, q, A):
    elephant = ERW(p, q)
    # walk at least one step
    elephant.walk()
    # go on
    while elephant.where() != A:
        if elephant.steps() > BOUND:
            raise TooDeepError()
        elephant.walk()
    return elephant.steps()

def A_func(A):
    # with open(f"{TMP_DIR}/{A}.csv", 'w', encoding='utf-8') as fout:
    output = {}
    for p_ in range(0, 11):
        p = p_ / 10
        for q_ in range(0, 11):
            q = q_ / 10
            print(f"{p:.2f}, {q:.2f}, {A:d}")
            res = []
            for i in (range(ITERS)):
                try:
                    res.append(sim(p, q, A))
                except TooDeepError:
                    pass
            output[f'{p_}_{q_}_{A}'] = np.array(res)
    np.savez(f"{TMP_DIR}/{A}.npz", **output)

def worker_func(As):
    for A in As:
        A_func(A)

if __name__ == '__main__':
    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.mkdir(TMP_DIR)

    if len(sys.argv) > 1:
        n_proc = int(sys.argv[1])
    else:
        n_proc = 8

    print(f"Num process: {n_proc}")

    # divide
    tasks = []
    for i in range(n_proc):
        tasks.append([])
    for A in range(-10, 11):
        tasks[A % n_proc].append(A)

    procs = []
    for ip in range(n_proc):
        p = Process(target=worker_func, args=(tasks[ip],))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

    # merge    
    all_output = {}
    for A in range(-10, 11, 2):
        f = np.load(f"{TMP_DIR}/{A}.npz")
        for k in f.files:
            print(f"{k} loaded")
            all_output[k] = f[k]
        f.close()

    np.savez_compressed("q1.npz", all_output)

    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)
