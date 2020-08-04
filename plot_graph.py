import matplotlib.pyplot as plt
import pandas as pd

experiments = ["PPO_PommeMultiAgent-1vs1_0_2020-08-01_16-57-57dig8pgx5"]

def plot_graph(experiment, criterion, labels):
    exp = pd.read_csv('/home/lucius/ray_results/1vs1_testing/{}/progress.csv'.format(experiment))
    fig, ax = plt.subplots()
    for i, criteria in enumerate(criterion):
        ax.plot(exp['timesteps_total'], exp[criteria], label='win rate')
        # ax.set_xlim(0, 6000000)

        ax.set_xlabel('timesteps')
        ax.set_ylabel("Win rate")
        ax.set_title("Win rate against static enemy")
        ax.legend()
    plt.show()


if __name__ == '__main__':
    criterion = [
        "custom_metrics/policy_0/win_rate_mean",
        # "custom_metrics/policy_0/EnemyDeath_mean",
        # "custom_metrics/agent_training_0_0/RealBombs_mean",
        # "custom_metrics/agent_training_0_2/RealBombs_mean"
    ]
    labels = []

    for exp in experiments:
        plot_graph(exp, criterion, labels)
