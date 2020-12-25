import os
import shutil
import sys
from multiprocessing import Process

import numpy as np

from erw import ERW

ITERS = 30
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

def p_func(p_):
    with open(f"{TMP_DIR}/{p_}.csv", 'w', encoding='utf-8') as fout:
        p = p_ / 10
        for q_ in range(1, 10):
            for A in range(-10, 11, 2):
                q = q_ / 10
                print(f"{p:.2f}, {q:.2f}, {A:d}")
                res = []
                for i in (range(ITERS)):
                    try:
                        res.append(sim(p, q, A))
                    except TooDeepError:
                        pass
                fout.write(f"{p:.1f},{q:.1f},{A:d},{np.mean(res):.2f},{np.std(res):.2f},{len(res)/ITERS*100:.2f}\n")
                fout.flush()

def worker_func(ps):
    for p_ in ps:
        p_func(p_)

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
    for p_ in range(1, 10):
        tasks[p_ % n_proc].append(p_)

    procs = []
    for ip in range(n_proc):
        p = Process(target=worker_func, args=(tasks[ip],))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

    # merge
    with open("q1.csv", 'w') as fout:
        fout.write("p,q,A,mean,std,succ\n")
        fout.flush()
        for p_ in range(1, 10):
            with open(f"{TMP_DIR}/{p_}.csv", 'r', encoding='utf-8') as fin:
                lines = fin.readlines()
                fout.writelines(lines)
                fout.flush()

    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)
