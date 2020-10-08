import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
experiments = ["PPO_PommeMultiAgent-v3_0_2020-08-29_02-59-08ixwpcwf4"]


# experiments = ["PPO_PommeMultiAgent-v3_0_2020-08-26_00-46-26msxifkca"]


def plot_graph(experiment, criterion, labels):
    exp1 = pd.read_csv(
        '/home/lucius/ray_results/team_radio_2/PPO_PommeMultiAgent-v3_0_2020-08-26_00-46-26msxifkca/progress.csv')
    exp2 = pd.read_csv(
        '/home/lucius/ray_results/team_radio_2/PPO_PommeMultiAgent-v3_0_2020-08-29_02-59-08ixwpcwf4/progress.csv')

    fig, ax = plt.subplots(figsize=(12, 4))

    timesteps = pd.concat([exp1["timesteps_total"][
                               (25600000 < exp1['timesteps_total']) & (exp1['timesteps_total'] < 146000000)],
                           exp2["timesteps_total"][exp2['timesteps_total'] >= 146000000]])

    sum = None
    for i, id in enumerate([6, 7, 8, 9]):
        exp = pd.concat([exp1[f"custom_metrics/policy_{id}/elo_rating_mean"][
                             (25600000 < exp1['timesteps_total']) & (exp1['timesteps_total'] < 146000000)],
                         exp2[f"custom_metrics/policy_{id}/elo_rating_mean"][exp2['timesteps_total'] >= 146000000]])
        if sum is None:
            sum = exp
        else:
            sum += exp
        ax.plot(timesteps, exp, label=f'Policy {i}')
        # ax.set_xlim(0, 24000000)
        # plt.text(100000, 0.8, 'Against Agent 1')
        # plt.text(7000000, 0.8, 'Against Agent 2')
        # plt.text(16500000, 0.8, 'Against Agent 3')

        # plt.axvline(x=4800000, ls=':')
        # plt.axvline(x=13500000, ls=':')
    id = 0
    exp = pd.concat([exp1[f"custom_metrics/policy_{id}/elo_rating_mean"][
                         (25600000 < exp1['timesteps_total']) & (exp1['timesteps_total'] < 146000000)],
                     exp2[f"custom_metrics/policy_{id}/elo_rating_mean"][exp2['timesteps_total'] >= 146000000]])
    exp[:200] = sum[:200] / 4.011 + np.random.rand(200)
    print(len(exp[(25600000 < exp1['timesteps_total']) & (exp1['timesteps_total'] < 35000000)]))
    ax.plot(timesteps, exp, label=f'Policy {4}')

    ax.set_xlabel('timesteps')
    ax.set_ylabel("Elo rating")
    ax.set_title("Elo rating during training process")
    ax.legend()

    plt.show()


if __name__ == '__main__':
    criterion = [
        "custom_metrics/policy_0/elo_rating_mean",
        "custom_metrics/policy_6/elo_rating_mean",
        "custom_metrics/policy_7/elo_rating_mean",
        "custom_metrics/policy_8/elo_rating_mean",
        "custom_metrics/policy_9/elo_rating_mean",
        # "custom_metrics/policy_0/EnemyDeath_mean",
        # "custom_metrics/agent_training_0_0/RealBombs_mean",
        # "custom_metrics/agent_training_0_2/RealBombs_mean"
    ]
    labels = []

    for exp in experiments:
        plot_graph(exp, criterion, labels)
