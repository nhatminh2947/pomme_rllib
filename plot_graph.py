import matplotlib.pyplot as plt
import pandas as pd

exp_1 = pd.read_csv(
    '/home/lucius/ray_results/2vs2_center_obs/PPO_PommeMultiAgent-v2_0_2020-07-16_02-46-00co9kkax2/progress.csv')
exp_2 = pd.read_csv(
    '/home/lucius/ray_results/2vs2_center_obs/PPO_PommeMultiAgent-v2_0_2020-07-15_19-44-100rjtw3d7/progress.csv')

fig, ax = plt.subplots()
# progress['custom_metrics/agent_training_0_0/EnemyDeath_mean'].plot()
ax.plot(exp_1['timesteps_total'], exp_1['custom_metrics/agent_training_0_0/EnemyDeath_mean'],
        label='Experiment 1 - Full board')
ax.plot(exp_2['timesteps_total'], exp_2['custom_metrics/agent_training_0_0/EnemyDeath_mean'],
        label='Experiment 2 - Center observation')
ax.set_xlim(0, 22000000)

ax.set_xlabel('timesteps')
ax.set_ylabel('Enemy Death Mean')
ax.set_title('Enemy Death')
ax.legend()
plt.show()

fig, ax = plt.subplots()
# progress['custom_metrics/agent_training_0_0/EnemyDeath_mean'].plot()
ax.plot(exp_1['timesteps_total'], exp_1['custom_metrics/agent_training_0_0/RealBombs_mean'],
        label='Experiment 1 - Full board - Agent 1')
ax.plot(exp_1['timesteps_total'], exp_1['custom_metrics/agent_training_0_2/RealBombs_mean'],
        label='Experiment 1 - Full board - Agent 2')
ax.plot(exp_2['timesteps_total'], exp_2['custom_metrics/agent_training_0_0/RealBombs_mean'],
        label='Experiment 2 - Center observation - Agent 1')
ax.plot(exp_2['timesteps_total'], exp_2['custom_metrics/agent_training_0_2/RealBombs_mean'],
        label='Experiment 2 - Center observation - Agent 2')
ax.set_xlim(0, 22000000)

ax.set_xlabel('timesteps')
ax.set_ylabel('Average number of bomb placed')
ax.set_title('Number of Bomb placed per episode')
ax.legend()
plt.show()

fig, ax = plt.subplots()
# progress['custom_metrics/agent_training_0_0/EnemyDeath_mean'].plot()
ax.plot(exp_1['timesteps_total'], exp_1['info/learner/policy_0/vf_loss'],
        label='Experiment 1 - Full board')
ax.plot(exp_2['timesteps_total'], exp_2['info/learner/policy_0/vf_loss'],
        label='Experiment 2 - Center observation')
ax.set_xlim(0, 22000000)

ax.set_xlabel('timesteps')
ax.set_ylabel('Loss')
ax.set_title('Value Function Loss')
ax.legend()
plt.show()


fig, ax = plt.subplots()
# progress['custom_metrics/agent_training_0_0/EnemyDeath_mean'].plot()
ax.plot(exp_1['timesteps_total'], exp_1['info/learner/policy_0/entropy'],
        label='Experiment 1 - Full board')
ax.plot(exp_2['timesteps_total'], exp_2['info/learner/policy_0/entropy'],
        label='Experiment 2 - Center observation')
ax.set_xlim(0, 22000000)

ax.set_xlabel('timesteps')
ax.set_ylabel('entropy')
ax.set_title('Entropy')
ax.legend()
plt.show()

print(exp_1.columns)
print(exp_1['info/learner/policy_0/vf_loss'])
print(exp_1['info/learner/policy_0/entropy'])