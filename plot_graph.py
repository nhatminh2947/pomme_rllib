import matplotlib.pyplot as plt
import pandas as pd

experiments = ["PPO_PommeMultiAgent-v2_0_2020-07-24_16-36-027wkyeno6"]


def plot_graph(experiment, criterion, labels):
    exp = pd.read_csv('/home/lucius/ray_results/2vs2/{}/progress.csv'.format(experiment))
    fig, ax = plt.subplots()
    for i, criteria in enumerate(criterion):
        ax.plot(exp['timesteps_total'], exp[criteria], label='Agent {}'.format(i))
        ax.set_xlim(0, 6000000)

        ax.set_xlabel('timesteps')
        ax.set_ylabel("Average bomb placed per episode")
        ax.set_title("Number of bomb placed")
        ax.legend()
    plt.show()


if __name__ == '__main__':
    criterion = [
        # "custom_metrics/win_mean",
        # "custom_metrics/agent_training_0_0/EnemyDeath_mean",
        "custom_metrics/agent_training_0_0/RealBombs_mean",
        "custom_metrics/agent_training_0_2/RealBombs_mean"
    ]
    labels = []

    for exp in experiments:
        plot_graph(exp, criterion, labels)
