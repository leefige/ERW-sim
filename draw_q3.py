import os

from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

from q3 import MAX_STEP, p_s, q_s

FIG_DIR = "./fig"
os.makedirs(FIG_DIR, exist_ok=True)
FIG_DIR = "./fig/q3"
os.makedirs(FIG_DIR, exist_ok=True)

# A=8,q=0.7,check p
def plot_p_q(p_, q_):
    points = np.load(f"res_q3/{p_}_{q_}.npy")
    print(p_, q_, len(points), points[:10])
    points = points / np.sqrt(MAX_STEP)

    plt.title("Asymptotic distribution of $\\frac{S_n}{\\sqrt{n}}$: $p=%.1f, q=%.1f$" % (p_/100, q_/100))
    plt.grid()
    plt.xlabel(r"$\frac{S_n}{\sqrt{n}}$")
    plt.ylabel("density")
    plt.hist(points, bins=15, align='right', density=True)

    if p_ < 75:
        var = 1 / (3 - 4 * p_ / 100)
        print(np.mean(points), np.sqrt(var), np.std(points))
        minv = np.min(points)
        maxv = np.max(points)
        xs = np.linspace(minv, maxv)
        ys = stats.norm.pdf(xs, 0, np.sqrt(var))
        plt.plot(xs, ys, color='coral', label=r"pdf of $\mathcal{N}\left(0, \frac{1}{3-4p}\right)$")
        plt.legend()

    plt.savefig(f"{FIG_DIR}/{p_}_{q_}.png")
    plt.cla()

if __name__ == '__main__':
    for p_ in p_s:
        for q_ in q_s:
            plot_p_q(p_, q_)
