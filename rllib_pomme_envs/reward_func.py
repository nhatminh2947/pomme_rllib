from pommerman import utility, constants

from metrics import Metrics





def match_reward(prev_obs, current_obs, stat, info):
    reward = 0
    for i in range(10, 14):
        if i in prev_obs['alive'] and i not in current_obs['alive']:
            if constants.Item(value=i) in current_obs['enemies']:
                stat[Metrics.EnemyDeath.name] += 1
                reward += 0
            elif constants.Item(value=i) == current_obs['teammate']:
                reward += 0
            else:
                reward += -1
                stat[Metrics.DeadOrSuicide.name] += 1

    if info['result'] == constants.Result.Tie:
        reward += -1

    return reward


def reward(action, prev_obs, current_obs, info, stat):
    reward = 0
    reward += immediate_reward(action, prev_obs, current_obs, stat)
    reward += match_reward(prev_obs, current_obs, stat, info)

    return reward
