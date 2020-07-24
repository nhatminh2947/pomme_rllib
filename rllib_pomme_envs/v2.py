import numpy as np
import ray
from pommerman import constants

from memory import Memory
from metrics import Metrics
from rllib_pomme_envs import v0
from utils import featurize_v6


# Note: change team for training agents
class RllibPomme(v0.RllibPomme):
    def __init__(self, config):
        super().__init__(config)
        self._centering = config["center"]
        self._input_size = config["input_size"]
        self.memory = [
            Memory(i) for i in range(self.num_agents)
        ]

    def step(self, action_dict):
        if self.is_render:
            self.render(record_pngs_dir="/home/lucius/ray_results/records/pngs",
                        record_json_dir="/home/lucius/ray_results/records/logs")

        actions = []
        for i in range(4):
            if self.agent_names[i] in action_dict:
                actions.append(int(action_dict[self.agent_names[i]]))
            else:
                actions.append(0)

        for i in range(self.num_agents):
            if actions[i] == constants.Action.Bomb.value:
                self.stat[i][Metrics.ActionBombs.name] += 1
                if self.prev_obs[i]['ammo'] > 0:
                    self.stat[i][Metrics.RealBombs.name] += 1

        obs = {}
        rewards = {}
        dones = {}
        infos = {self.agent_names[i - 10]: {} for i in self.prev_obs[0]['alive']}

        _obs, _reward, _done, _info = self.env.step(actions)

        for i in self.prev_obs[0]['alive']:
            if _done or i not in _obs[i - 10]['alive']:
                dones[self.agent_names[i - 10]] = True
                infos[self.agent_names[i - 10]]["metrics"] = self.stat[i - 10]

        dones["__all__"] = _done

        for i in range(self.num_agents):
            if self.is_agent_alive(i, self.prev_obs[i]['alive']):
                self.memory[i].update_memory(_obs[i])

                obs[self.agent_names[i]] = featurize_v6(self.memory[i].obs, centering=self._centering,
                                                        input_size=self._input_size)
                rewards[self.agent_names[i]] = self.reward(i, actions[i], self.prev_obs[i],
                                                           _obs[i], _info, self.stat[i])
                infos[self.agent_names[i]].update(_info)

                if _done and "training" in self.agent_names[i]:
                    if _info['result'] == constants.Result.Tie:
                        infos[self.agent_names[i]]['training_result'] = constants.Result.Tie
                    else:
                        if i in _info['winners']:
                            infos[self.agent_names[i]]['training_result'] = constants.Result.Win
                        else:
                            infos[self.agent_names[i]]['training_result'] = constants.Result.Loss

        self.prev_obs = _obs

        return obs, rewards, dones, infos

    def reset(self):
        self.prev_obs = self.env.reset()
        obs = {}
        self.reset_stat()
        g_helper = ray.util.get_actor("g_helper")
        self.agent_names = ray.get(g_helper.get_agent_names.remote())

        if np.random.random() > 0.5:
            self.agent_names[0], self.agent_names[1] = self.agent_names[1], self.agent_names[0]
            self.agent_names[2], self.agent_names[3] = self.agent_names[3], self.agent_names[2]

        for i in range(self.num_agents):
            self.memory[i].init_memory(self.prev_obs[i])
            if self.is_agent_alive(i, self.prev_obs[i]['alive']):
                obs[self.agent_names[i]] = featurize_v6(self.prev_obs[i], centering=self._centering,
                                                        input_size=self._input_size)

        return obs

    def reward(self, id, action, prev_obs, current_obs, info, stat):
        game_reward = 0

        if id + 10 in prev_obs['alive'] and id + 10 not in current_obs['alive']:  # died
            game_reward += -1.0
            stat[Metrics.DeadOrSuicide.name] += 1
            for i in range(10, 14):
                if constants.Item(value=i) in current_obs['enemies'] and i not in current_obs['alive']:
                    game_reward += 0.5
                    stat[Metrics.EnemyDeath.name] += 1
        elif info['result'] == constants.Result.Win:
            for i in range(10, 14):
                if constants.Item(value=i) in current_obs['enemies'] and i not in current_obs['alive']:
                    game_reward += 0.5
                    stat[Metrics.EnemyDeath.name] += 1
        elif info['result'] == constants.Result.Tie:
            temp_reward = 0
            for i in range(10, 14):
                if constants.Item(value=i) in current_obs['enemies'] and i not in current_obs['alive']:
                    temp_reward += 0.5
                    stat[Metrics.EnemyDeath.name] += 1

            if temp_reward == 0:
                game_reward += -1.0
            else:
                game_reward += temp_reward

        if action == constants.Action.Bomb.value:
            stat[Metrics.ActionBombs.name] += 1
            if prev_obs['ammo'] > 0:
                stat[Metrics.RealBombs.name] += 1

        g_helper = ray.util.get_actor("g_helper")
        alpha = ray.get(g_helper.get_alpha.remote())

        exploration_reward = self.exploration_reward(action, prev_obs, current_obs, stat)
        stat[Metrics.ExplorationReward.name] += exploration_reward
        stat[Metrics.GameReward.name] += game_reward

        return (1 - alpha) * game_reward + alpha * exploration_reward
