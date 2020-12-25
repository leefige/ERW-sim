from erw import ERW
import numpy as np
from tqdm import tqdm

BOUND = 2e4

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

if __name__ == '__main__':
    iters = 30
    with open("q1.csv", 'w') as fout:
        fout.write("p,q,A,mean,std,succ\n")
        fout.flush()
        for p_ in range(1, 10):
            for q_ in range(1, 10):
                for A in range(-10, 11, 2):
                    p = p_ / 10
                    q = q_ / 10
                    res = []
                    for i in tqdm(range(iters)):
                        try:
                            res.append(sim(p, q, A))
                        except TooDeepError:
                            pass
                    fout.write(f"{p:.1f},{q:.1f},{A:d},{np.mean(res):.2f},{np.std(res):.2f},{len(res)/iters*100:.2f}\n")
                    fout.flush()
