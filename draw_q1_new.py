import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma

from q1 import ITERS

FIG_DIR = "./fig"
os.makedirs(FIG_DIR, exist_ok=True)
FIG_DIR = "./fig/q1_new"
os.makedirs(FIG_DIR, exist_ok=True)

# A=8,q=0.7,check p
# def plot_p(df, A, q):
#     res = df.query(f'A=={A}').query(f'q=={q}')
#     print(res)
#     fig, ax1 = plt.subplots()
#     ax1.plot(res['p'].to_numpy(), res['mean'].to_numpy())
#     ax2 = ax1.twinx()
#     ax2.plot(res['p'].to_numpy(), res['succ'].to_numpy(), 'r')
#     plt.savefig(f"{FIG_DIR}/plot_p_{A}_{q}.png")

def plot_q(data, A, p_):
    p = p_ / 10
    points = []
    succ = []
    qs = []
    for q_ in range(0, 11):
        qs.append(q_ / 10)
        points.append(np.mean(data[f'{p_}_{q_}_{A}']))
        succ.append(len(data[f'{p_}_{q_}_{A}']) / ITERS)
    means = np.array(points)
    print(means)
    succ = np.array(succ)

    fig, ax1 = plt.subplots()
    ax1.set_title(f"E[T] vs q: p={p:.1f}, A={A:.1f}")
    ax1.set_xlabel("A")
    ax1.grid(True)
    ax1.scatter(qs, means, label="real mean")
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 1.1)
    ax2.plot(qs, succ, 'r', label="valid rate")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center left')
    plt.savefig(f"{FIG_DIR}/plot_q_{A}_{p}.png")

def plot_q_box(data, A, p_):
    p = p_ / 10
    points = []
    succ = []
    qs = []
    for q_ in range(0, 11):
        qs.append(q_ / 10)
        points.append(data[f'{p_}_{q_}_{A}'])
        succ.append(len(data[f'{p_}_{q_}_{A}']) / ITERS)
    succ = np.array(succ)

    fig, ax1 = plt.subplots()
    ax1.set_title(f"E[T] vs q: p={p:.1f}, A={A:.1f}")
    ax1.set_xlabel("E[T]")
    ax1.set_ylabel("q")
    ax1.grid(True)
    # ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.boxplot(points, labels=qs, vert=False, showmeans=True)
    plt.savefig(f"{FIG_DIR}/plot_q_box_{A}_{p}.png")

def exp_ta(ts, p):
    def inner_func(T, alpha):
        res = 1
        while T > 1:
            res *= (T - 1) / (T - 1 + alpha)
            T -= 1
        assert T == 1
        return res / gamma(1 + alpha)

    return np.mean([inner_func(t, 2 * p - 1) for t in ts])

def plot_reg(data, p_, q_):
    # log(exp_ta) = log(2q-1) - log(A) - log(gamma(2p))
    p = p_ / 10
    q = q_ / 10

    points = []
    As = list(range(-10, 11, 2))
    for A in As:
        points.append(data[f'{p_}_{q_}_{A}'])
    etas = [exp_ta(it, p) for it in points]
    etas = np.array(etas)
    log_etas = np.log(etas)

    succ = [len(it) / ITERS for it in points]
    succ = np.array(succ)

    if q > 0.5:
        xs = np.array(list(range(1, 11)))
        ys = np.log(2*q-1)-np.log(xs)-np.log(gamma(2*p))
    else:
        xs = np.array(list(range(-10, 0)))
        ys = np.log(1-2*q)-np.log(-xs)-np.log(gamma(2*p))
    
    fig, ax1 = plt.subplots()
    ax1.set_title(f"p={p:.1f}, q={q:.1f}")
    ax1.set_xlabel("A")
    ax1.grid(True)
    
    ax1.plot(xs, ys, label="predicted")
    ax1.scatter(As, log_etas, label="real mean")

    ax2 = ax1.twinx()
    print(log_etas)
    ax2.set_ylim(0, 1.1)
    ax2.plot(As, succ, 'r', label="valid rate")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower center')
    plt.savefig(f"{FIG_DIR}/plot_reg_{p_}_{q_}.png")
    plt.cla()

def plot_all_reg(data):
    for p_ in range(6, 10):
        plot_reg(data, p_, 0)
        plot_reg(data, p_, 1)
        plot_reg(data, p_, 2)
        plot_reg(data, p_, 3)
        plot_reg(data, p_, 4)
        plot_reg(data, p_, 6)
        plot_reg(data, p_, 7)
        plot_reg(data, p_, 8)
        plot_reg(data, p_, 9)
        plot_reg(data, p_, 10)

if __name__ == '__main__':
    f = np.load("q1.npz", allow_pickle=True)
    data = f['arr_0'].item()
    f.close()

    # plot_all_reg(data)

    for p_ in range(0, 5):
        plot_q_box(data, 10, p_)
        plot_q(data, 10, p_)

    # for q_ in range(0, 11):
    #     plot_reg(data, 10, q_)
