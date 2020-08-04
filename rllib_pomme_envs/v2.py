import numpy as np
import ray
from pommerman import constants
from pommerman import utility

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
        self._evaluate = config["evaluate"]
        self.memory = [
            Memory(i) for i in range(self.num_agents)
        ]
        if self._evaluate:
            self.policies = config["policies"]
        else:
            self.policies = None
        self.agent_names = []
        self.num_steps = 0

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

        self.num_steps += 1

        for i in self.prev_obs[0]['alive']:
            if _done or i not in _obs[i - 10]['alive']:
                dones[self.agent_names[i - 10]] = True
                infos[self.agent_names[i - 10]]["metrics"] = self.stat[i - 10]
                infos[self.agent_names[i - 10]]["num_steps"] = self.num_steps

        dones["__all__"] = _done

        for i in range(self.num_agents):
            if self.is_agent_alive(i, self.prev_obs[i]['alive']):
                name, id, _ = self.agent_names[i].split("_")
                policy_name = "{}_{}".format(name, id)

                self.memory[i].update_memory(_obs[i])

                obs[self.agent_names[i]] = featurize_v6(self.memory[i].obs, centering=self._centering,
                                                        input_size=self._input_size)
                rewards[self.agent_names[i]] = self.reward_v1(policy_name, i, actions[i], self.prev_obs[i],
                                                           _obs[i], _info, self.stat[i])
                infos[self.agent_names[i]].update(_info)

        self.prev_obs = _obs

        return obs, rewards, dones, infos

    def get_policy_name(self, agent_name):
        return "policy_{}".format(agent_name.split("_")[1])

    def reset(self):
        if not self._evaluate:
            ers = ray.util.get_actor("ers")
            self.policies = ray.get(ers.get_training_policies.remote())

        self.agent_names = []
        if np.random.random() < 0.5:
            for i in range(4):
                self.agent_names.append("{}_{}".format(self.policies[i % 2], i))
        else:
            for i in range(4):
                self.agent_names.append("{}_{}".format(self.policies[(i + 1) % 2], i))

        self.reset_stat()
        self.prev_obs = self.env.reset()
        obs = {}
        self.num_steps = 0
        for i in range(self.num_agents):
            self.memory[i].init_memory(self.prev_obs[i])
            if self.is_agent_alive(i, self.prev_obs[i]['alive']):
                obs[self.agent_names[i]] = featurize_v6(self.prev_obs[i], centering=self._centering,
                                                        input_size=self._input_size)

        return obs

    def reward(self, policy_name, id, action, prev_obs, current_obs, info, stat):
        game_reward = 0

        if id + 10 in prev_obs['alive'] and id + 10 not in current_obs['alive']:  # died
            stat[Metrics.DeadOrSuicide.name] += 1

        if info['result'] == constants.Result.Win and id in info["winners"]:
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
        elif id + 10 in prev_obs['alive'] and id + 10 not in current_obs['alive']:  # died
            game_reward += -1.0
            for i in range(10, 14):
                if constants.Item(value=i) in current_obs['enemies'] and i not in current_obs['alive']:
                    game_reward += 0.5
                    stat[Metrics.EnemyDeath.name] += 1

        if action == constants.Action.Bomb.value:
            stat[Metrics.ActionBombs.name] += 1
            if prev_obs['ammo'] > 0:
                stat[Metrics.RealBombs.name] += 1

        ers = ray.util.get_actor("ers")
        alpha = ray.get(ers.get_alpha.remote(policy_name))

        exploration_reward = self.exploration_reward(action, prev_obs, current_obs, stat)
        stat[Metrics.ExplorationReward.name] += exploration_reward
        stat[Metrics.GameReward.name] += game_reward

        return (1 - alpha) * game_reward + alpha * exploration_reward
        # return game_reward + exploration_reward

    def exploration_reward_v1(self, action, prev_obs, current_obs, stat):
        reward = 0
        if prev_obs['blast_strength'] < current_obs['blast_strength']:
            reward += 0.1
            stat[Metrics.IncrRange.name] += 1

        if utility._position_is_item(prev_obs['board'],
                                     current_obs['position'],
                                     constants.Item.ExtraBomb):
            reward += 0.1
            stat[Metrics.ExtraBomb.name] += 1

        if not prev_obs['can_kick'] and current_obs['can_kick']:
            reward += 0.2
            stat[Metrics.Kick.name] = True

        pos = current_obs['position']
        if prev_obs['ammo'] > 0 and action == constants.Action.Bomb.value and prev_obs['bomb_life'][pos] == 0:
            dx = [-1, 0, 0, 1]
            dy = [0, -1, 1, 0]
            reward += 0.005
            for i in range(4):
                for j in range(1, prev_obs['blast_strength']):
                    row = pos[0] + j * dx[i]
                    col = pos[1] + j * dy[i]
                    if 0 <= row < current_obs['board'].shape[0] and 0 <= col < current_obs['board'].shape[0]:
                        if current_obs['board'][row, col] != constants.Item.Passage.value \
                                and current_obs['board'][row, col] == constants.Item.Wood.value:
                            # reward += 0.01
                            stat[Metrics.ExplodeWood.name] += 1
                            break

        see_enemy = False
        for enemy in current_obs['enemies']:
            if (current_obs['board'] == enemy.value).any():
                see_enemy = True

        if not see_enemy:
            reward -= 0.001

        return reward

    def reward_v1(self, policy_name, id, action, prev_obs, current_obs, info, stat):
        ers = ray.util.get_actor("ers")
        alpha = ray.get(ers.get_alpha.remote(policy_name))

        exploration_reward = self.exploration_reward_v1(action, prev_obs, current_obs, stat)
        stat[Metrics.ExplorationReward.name] += exploration_reward

        game_reward = 0

        if id + 10 in prev_obs['alive'] and id + 10 not in current_obs['alive']:  # died
            stat[Metrics.DeadOrSuicide.name] += 1
            game_reward += -1.0
        else:
            for i in range(10, 14):
                if i in prev_obs['alive'] and i not in current_obs['alive']:  # agent i is died
                    if constants.Item(value=i) in current_obs['enemies']:
                        game_reward += 0.5
                        stat[Metrics.EnemyDeath.name] += 1
                    elif constants.Item(value=i) == current_obs['teammate']:
                        game_reward += -0.25

        if action == constants.Action.Bomb.value:
            stat[Metrics.ActionBombs.name] += 1
            if prev_obs['ammo'] > 0:
                stat[Metrics.RealBombs.name] += 1

        stat[Metrics.GameReward.name] += game_reward

        return (1 - alpha) * game_reward + alpha * exploration_reward
