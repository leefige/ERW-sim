from scipy.special import gamma
import numpy as np


def inner_func(T, alpha):
    res = 1
    while T > 1:
        res *= (T - 1) / (T - 1 + alpha)
        T -= 1
    assert T == 1
    return res / gamma(1 + alpha)

def outer_func(T, alpha):
    return gamma(T) / gamma(T + alpha)

xs = np.random.randint(2, 100, 50)
alpha = 2 * 0.75 - 1
outer = outer_func(xs, alpha)
inner = np.array([inner_func(x, alpha) for x in xs])
print(np.sum(outer - inner))