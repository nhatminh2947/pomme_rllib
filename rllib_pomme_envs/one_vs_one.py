import numpy as np
import ray
from pommerman import constants

from metrics import Metrics
from rllib_pomme_envs import v0
from utils import featurize_v4, featurize_v6


# Note: change team for training agents
class RllibPomme(v0.RllibPomme):
    def __init__(self, config):
        super().__init__(config)
        self.policies = None
        self.num_steps = 0

    def step(self, action_dict):
        if self.is_render:
            self.render()

        actions = []
        for i in range(self.num_agents):
            if self.agent_names[i] in action_dict:
                actions.append(int(action_dict[self.agent_names[i]]))
                if action_dict[self.agent_names[i]] == constants.Action.Bomb.value and self.prev_obs[i]['ammo'] > 0:
                    self.stat[i][Metrics.RealBombs.name] += 1
            else:
                actions.append(0)

        obs = {}
        rewards = {}
        dones = {}
        infos = {self.agent_names[i - 10]: {} for i in self.prev_obs[0]['alive']}

        _obs, _reward, _done, _info = self.env.step(actions)
        self.num_steps += 1

        for i in self.prev_obs[0]['alive']:
            if _done or self.is_done(i - 10, self.prev_obs[0]["alive"], _obs[0]['alive']):
                dones[self.agent_names[i - 10]] = True
                infos[self.agent_names[i - 10]]["metrics"] = self.stat[i - 10]
                infos[self.agent_names[i - 10]]["num_steps"] = self.num_steps

        dones["__all__"] = _done

        for i in range(self.num_agents):
            if self.is_agent_alive(i, self.prev_obs[i]["alive"]):
                name, id, _ = self.agent_names[i].split("_")
                policy_name = "{}_{}".format(name, id)

                obs[self.agent_names[i]] = featurize_v6(_obs[i], centering=False, input_size=8)
                rewards[self.agent_names[i]] = self.reward(policy_name, i, actions[i], self.prev_obs[i],
                                                           _obs[i], _info, self.stat[i])
                infos[self.agent_names[i]].update(_info)

        self.prev_obs = _obs

        return obs, rewards, dones, infos

    def reward(self, policy_name, id, action, prev_obs, current_obs, info, stat):
        game_reward = 0

        if id + 10 in prev_obs['alive'] and id + 10 not in current_obs['alive']:  # died
            stat[Metrics.DeadOrSuicide.name] += 1

        if info['result'] == constants.Result.Win and id in info["winners"]:
            for i in range(10, 12):
                if constants.Item(value=i) in current_obs['enemies'] and i not in current_obs['alive']:
                    game_reward += 1
                    stat[Metrics.EnemyDeath.name] += 1
        elif info['result'] == constants.Result.Tie:
            temp_reward = 0
            for i in range(10, 12):
                if constants.Item(value=i) in current_obs['enemies'] and i not in current_obs['alive']:
                    temp_reward += 1
                    stat[Metrics.EnemyDeath.name] += 1

            if temp_reward == 0:
                game_reward += -1.0
            else:
                game_reward += temp_reward
        elif id + 10 in prev_obs['alive'] and id + 10 not in current_obs['alive']:  # died
            game_reward += -1.0
            for i in range(10, 12):
                if constants.Item(value=i) in current_obs['enemies'] and i not in current_obs['alive']:
                    game_reward += 1
                    stat[Metrics.EnemyDeath.name] += 1

        if action == constants.Action.Bomb.value:
            stat[Metrics.ActionBombs.name] += 1
            if prev_obs['ammo'] > 0:
                stat[Metrics.RealBombs.name] += 1

        ers = ray.get_actor("ers")
        alpha = ray.get(ers.get_alpha.remote(policy_name))

        exploration_reward = self.exploration_reward(action, prev_obs, current_obs, stat)
        stat[Metrics.ExplorationReward.name] += exploration_reward
        stat[Metrics.GameReward.name] += game_reward

        return (1 - alpha) * game_reward + alpha * exploration_reward

    def reset(self):
        self.num_steps = 0
        self.prev_obs = self.env.reset()
        obs = {}
        self.reset_stat()
        ers = ray.get_actor("ers")
        self.policies = ray.get(ers.get_training_policies.remote())
        self.agent_names = []
        # print("Called reset")
        # print("self.agent_name:", self.agent_names)

        if np.random.random() < 0.5:
            for i in range(2):
                self.agent_names.append("{}_{}".format(self.policies[i % 2], i))
        else:
            for i in range(2):
                self.agent_names.append("{}_{}".format(self.policies[(i + 1) % 2], i))

        for i in range(self.num_agents):
            if self.is_agent_alive(i, self.prev_obs[i]["alive"]):
                obs[self.agent_names[i]] = featurize_v6(self.prev_obs[i], False, 8)

        return obs
