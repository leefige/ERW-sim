import os

import numpy as np
import pandas
from matplotlib import pyplot as plt

FIG_DIR = "./fig"
os.makedirs(FIG_DIR, exist_ok=True)
FIG_DIR = "./fig/q1"
os.makedirs(FIG_DIR, exist_ok=True)

# A=8,q=0.7,check p
def plot_p(df, A, q):
    res = df.query(f'A=={A}').query(f'q=={q}')
    print(res)
    fig, ax1 = plt.subplots()
    ax1.plot(res['p'].to_numpy(), res['mean'].to_numpy())
    ax2 = ax1.twinx()
    ax2.plot(res['p'].to_numpy(), res['succ'].to_numpy(), 'r')
    plt.savefig(f"{FIG_DIR}/plot_p_{A}_{q}.png")

def plot_q(df, A, p):
    res = df.query(f'A=={A}').query(f'p=={p}')
    print(res)
    fig, ax1 = plt.subplots()
    ax1.plot(res['q'].to_numpy(), res['mean'].to_numpy())
    ax2 = ax1.twinx()
    ax2.plot(res['q'].to_numpy(), res['succ'].to_numpy(), 'r')
    plt.savefig(f"{FIG_DIR}/plot_q_{A}_{p}.png")


if __name__ == '__main__':
    df = pandas.read_csv("./q1.csv")
    plot_p(df, 0, 0.7)
    plot_p(df, 0, 0.5)
    plot_p(df, 0, 0.2)
    plot_q(df, 0, 0.1)
    plot_q(df, 0, 0.2)
    plot_q(df, 0, 0.3)
    plot_q(df, 0, 0.4)
    plot_q(df, 0, 0.5)
    plot_q(df, 0, 0.6)
    plot_q(df, 0, 0.7)
    plot_q(df, 0, 0.8)
