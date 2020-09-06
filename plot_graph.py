import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

experiments = ["PPO_PommeMultiAgent-v3_0_2020-08-29_02-59-08ixwpcwf4"]


# experiments = ["PPO_PommeMultiAgent-v3_0_2020-08-26_00-46-26msxifkca"]


def plot_graph():
    exp = pd.read_csv(
        '/home/lucius/ray_results/team_radio_2/PPO_PommeMultiAgent-v3_0_2020-08-31_01-58-50vgng3gfu/progress.csv')

    fig, ax = plt.subplots(figsize=(12, 4))

    for i, id in enumerate([6, 7, 8, 9]):
        ax.plot(exp["timesteps_total"]-3000000, exp[f"custom_metrics/policy_{id}/elo_rating_mean"], label=f'Policy {i}')
        # ax.set_xlim(0, 24000000)
        # plt.text(100000, 0.8, 'Against Agent 1')
        # plt.text(7000000, 0.8, 'Against Agent 2')
        # plt.text(16500000, 0.8, 'Against Agent 3')

        # plt.axvline(x=4800000, ls=':')
        # plt.axvline(x=13500000, ls=':')

    ax.set_xlabel('timesteps')
    ax.set_ylabel("Elo rating")
    ax.set_title("Elo rating during training process")
    ax.legend()

    plt.show()


def plot_win_rate():
    exp = pd.read_csv(
        '/home/lucius/ray_results/team_radio_1/PPO_PommeMultiAgent-v3_0_2020-08-13_03-33-55jymb7f2n/progress.csv')

    fig, ax = plt.subplots()
    ax.plot(exp["timesteps_total"][exp["timesteps_total"] < 24000000],
            exp[f"custom_metrics/policy_{0}/EnemyDeath_mean"][exp["timesteps_total"] < 24000000],
            label="Enemy Death mean")
    ax.set_xlim(0, 25000000)
    ax.set_ylim(0, 2)
    ax.set_title("Enemy Death mean")
    ax.set_xlabel("Timesteps")
    ax.text(50000, 0.1, 'Against Agent 1')
    plt.text(7000000, 0.1, 'Against Agent 2')
    plt.text(16500000, 0.1, 'Against Agent 3')
    plt.axvline(x=4800000, ls=':')
    plt.axvline(x=13500000, ls=':')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_win_rate()

    plot_graph()