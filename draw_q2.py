import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma

from q2 import ITERS

_M_ = 10

FIG_DIR = "./fig"
os.makedirs(FIG_DIR, exist_ok=True)
FIG_DIR = "./fig/q2"
os.makedirs(FIG_DIR, exist_ok=True)
FIG_DIR = f"./fig/q2/box_{_M_}"
os.makedirs(FIG_DIR, exist_ok=True)

def gamma_ratio(n:int, alpha:float):
    """
    Gamma(n) / Gamma(n + alpha)
    """
    if alpha == 0:
        return 1
    elif alpha == 1:
        return 1 / n
    elif alpha == -1:
        return n - 1

    res = 1
    while n > 1:
        res *= (n - 1) / (n - 1 + alpha)
        n -= 1
    assert n == 1
    return res / gamma(1 + alpha)

def a_n_func(n:int, p:float):
    return gamma_ratio(n, 2 * p - 1) * gamma(2 * p)

def exp_ta(ts, p):
    return np.mean([gamma_ratio(t, 2 * p - 1) for t in ts])


def plot_reg(data, p_, q_):
    # log(exp_ta) = log(2q-1) - log(A) - log(gamma(2p))
    p = p_ / 10
    q = q_ / 10

    points = []
    As = list(range(-10, 11))
    for A in As:
        points.append(data[f'{p_}_{q_}_{A}'])

    succ = [len(it) / ITERS for it in points]
    succ = np.array(succ)

    small_ts = []
    for i in range(len(points)):
        t_arr = points[i]
        if len(t_arr.shape) > 1:
            small_ts.append(t_arr[np.isnan(t_arr[:, 1]), 0])
        else:
            small_ts.append(np.array([]))

    small_rate = np.array([len(it) / ITERS for it in small_ts])
    etas = np.array([exp_ta(it, p) for it in small_ts])
    log_etas = np.log(etas)

    if q > 0.5:
        xs = np.array(list(range(1, 11)))
        ys = np.log(2*q-1)-np.log(xs)-np.log(gamma(2*p))
    else:
        xs = np.array(list(range(-10, 0)))
        ys = np.log(1-2*q)-np.log(-xs)-np.log(gamma(2*p))
    
    fig, ax1 = plt.subplots()
    ax1.set_title(f"$p={p:.1f}, q={q:.1f}, M={_M_:d}, a_M={a_n_func(_M_, p):.2f}$")
    ax1.set_xlabel("A")
    ax1.grid(True)
    
    ax1.plot(xs, ys, label="predicted")
    ax1.scatter(As, log_etas, label="observed")

    ax2 = ax1.twinx()
    print(log_etas)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("rate")
    ax2.plot(As, succ, 'g', label="valid rate")
    ax2.plot(As, small_rate, 'r', label="smaller rate")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower center')
    plt.savefig(f"{FIG_DIR}/plot_reg_{_M_}_{p_}_{q_}.png")
    plt.cla()

def plot_large(data, p_, q_):
    # log(exp_ta) = log(2q-1) - log(A) - log(gamma(2p))
    p = p_ / 10
    q = q_ / 10

    ts = []
    As = list(range(-10, 11))
    for A in As:
        ts.append(data[f'{p_}_{q_}_{A}'])

    succ = [len(it) / ITERS for it in ts]
    succ = np.array(succ)

    large_ts = []
    for i in range(len(ts)):
        t_arr = ts[i]
        if len(t_arr.shape) > 1:
            large_ts.append(t_arr[np.logical_not(np.isnan(t_arr[:, 1]))])
        else:
            large_ts.append(np.array([[], []]).reshape(0, 2))

    large_rate = np.array([len(it) / ITERS for it in large_ts])
    exp_tsm = np.array([np.mean(it[:, 0] * it[:, 1]) for it in large_ts])
    # exp_sm = np.array([np.mean(it[:, 1]) for it in large_ts])

    xs = np.linspace(-10, 10)
    a_m = a_n_func(_M_, p)
    ys = xs / (2 * p - 1) - (2 * q - 1) * (2 - 2 * p) / (2 * p - 1) / a_m

    As = np.array(As)
    points = exp_tsm / _M_
    
    fig, ax1 = plt.subplots()
    ax1.set_title(f"$p={p:.1f}, q={q:.1f}, M={_M_:d}, a_M={a_m:.2f}$")
    ax1.set_xlabel("$A$")
    ax1.set_ylabel("$E[T_AS_M]/M$")
    ax1.grid(True)
    
    ax1.plot(xs, ys, label="predicted")
    ax1.scatter(As, points, label="observed")

    ax2 = ax1.twinx()
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("rate")
    ax2.plot(As, succ, 'g', label="valid rate")
    ax2.plot(As, large_rate, 'r', label="larger rate")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')
    plt.savefig(f"{FIG_DIR}/plot_large_{_M_}_{p_}_{q_}.png")
    plt.cla()


def plot_q_box(data, A, p_):
    p = p_ / 10
    points = []
    succ = []
    qs = []
    for q_ in range(0, 11):
        qs.append(q_ / 10)
        t_arr = data[f'{p_}_{q_}_{A}']
        if len(t_arr.shape) > 1:
            points.append(t_arr[:,0])
        else:
            points.append(np.array([]))
        succ.append(len(data[f'{p_}_{q_}_{A}']) / ITERS)
    succ = np.array(succ)

    fig, ax1 = plt.subplots()
    fig.subplots_adjust(top=0.85)
    ax1.set_title(f"E[T] vs q: p={p:.1f}, A={A:.1f}, M={_M_}")
    ax1.set_xlabel("E[T]")
    ax1.set_ylabel("q")
    ax1.grid(True)
    # ax1.set_yscale('log')
    ax1.set_xscale('log')
    # ax1.set_yticklabels([f"{i:.1f}" for i in qs]) 

    ax2 = ax1.twiny()
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("rate")
    ax2.plot(succ, range(len(points)), 'r', label="valid rate")

    ax1.boxplot(points, positions = range(len(points)), vert=False, showmeans=True)

    ticks_loc = ax2.get_yticks().tolist()
    ax2.set_yticks(ax1.get_yticks().tolist())
    ax2.set_yticklabels([f"{x/10:.1f}".format(x) for x in ticks_loc])

    plt.savefig(f"{FIG_DIR}/plot_q_box_{A}_{p}.png")


if __name__ == '__main__':
    f = np.load(f"q2_{_M_}.npz", allow_pickle=True)
    data = f['arr_0'].item()
    f.close()

    # for p_ in range(0, 5):
    #     for q_ in range(0, 5):
    #         plot_reg(data, p_, q_)
    #     for q_ in range(6, 11):
    #         plot_reg(data, p_, q_)
    # plot_large(data, 9, 5)

    for p_ in range(0, 5):
        plot_q_box(data, 0, p_)
        plot_q_box(data, 10, p_)
