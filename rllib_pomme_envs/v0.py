import pommerman
import ray
from gym import spaces
from pommerman import agents
from pommerman import constants
from pommerman import utility
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import utils
from metrics import Metrics
from utils import featurize, featurize_for_rms


# Note: change team for training agents
class RllibPomme(MultiAgentEnv):
    def __init__(self, config):
        self.num_agents = 2 if config["env_id"] == "OneVsOne-v0" else 4
        self.agent_list = []
        for i in range(self.num_agents):
            self.agent_list.append(agents.StaticAgent())

        self.env = pommerman.make(config["env_id"], self.agent_list, config["game_state_file"])
        self.is_render = config["render"]
        self.action_space = self.env.action_space
        self.prev_obs = None
        self.stat = None
        self.reset_stat()
        if config["env_id"] == "OneVsOne-v0":
            self.observation_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, 8, 8))
        else:
            self.observation_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, 11, 11))
        self.agent_names = None

    def reset_stat(self):
        self.stat = []
        for i in range(self.num_agents):
            metrics = {}
            for key in Metrics:
                metrics[key.name] = 0
            self.stat.append(metrics)

    def render(self, record_pngs_dir=None, record_json_dir=None):
        self.env.render(record_pngs_dir=record_pngs_dir)

    def step(self, action_dict):
        if self.is_render:
            self.render()

        actions = []
        for id in range(self.num_agents):
            if id in action_dict:
                actions.append(action_dict[self.agent_names[id]])
            else:
                actions.append(0)

        obs = {}
        rewards = {}
        dones = {}
        infos = {}

        _obs, _reward, _done, _info = self.env.step(actions)

        for id in range(self.num_agents):
            if self.is_done(id, _obs[0]['alive']):
                dones[self.agent_names[id]] = True

                if id == 0:
                    _done = True
                    _info['result'] = constants.Result.Loss

        dones["__all__"] = _done

        for id in range(self.num_agents):
            if self.is_agent_alive(id):
                obs[self.agent_names[id]] = featurize(_obs[id])
                rewards[self.agent_names[id]] = self.reward(id, actions[id], self.prev_obs[id],
                                                            _obs[id], _info, self.stat[id])
                infos[self.agent_names[id]] = _info

        self.prev_obs = _obs

        return obs, rewards, dones, infos

    def is_done(self, id, prev_obs, current_alive):
        return (id + 10) in prev_obs and (id + 10) not in current_alive

    def is_agent_alive(self, id, alive_agents):
        return (id + 10) in alive_agents

    def reset(self):
        self.prev_obs = self.env.reset()
        obs = {}
        self.reset_stat()
        g_helper = ray.util.get_actor("g_helper")
        self.agent_names = ray.get(g_helper.get_agent_names.remote())
        for i in range(self.num_agents):
            if self.is_agent_alive(i, self.prev_obs[i]['alive']):
                obs[self.agent_names[i]] = featurize(self.prev_obs[i])

        return obs

    def reward(self, id, action, prev_obs, current_obs, info, stat):
        reward = 0
        reward += self.immediate_reward(action, prev_obs, current_obs, stat)

        for i in range(10, 10 + self.num_agents):
            if i in prev_obs['alive'] and i not in current_obs['alive']:
                if constants.Item(value=i) in current_obs['enemies']:
                    reward += 1
                    stat[Metrics.EnemyDeath.name] += 1
                elif i - 10 == id:
                    reward += -1
                    stat[Metrics.DeadOrSuicide.name] += 1
                else:
                    reward += -0.5

        if info['result'] == constants.Result.Tie:
            reward += -1

        return reward

    def immediate_reward(self, action, prev_obs, current_obs, stat):
        reward = 0
        if prev_obs['blast_strength'] < current_obs['blast_strength']:
            reward += 0.01
            stat[Metrics.IncrRange.name] += 1

        if utility._position_is_item(prev_obs['board'],
                                     current_obs['position'],
                                     constants.Item.ExtraBomb):
            reward += 0.01
            stat[Metrics.ExtraBomb.name] += 1

        if not prev_obs['can_kick'] and current_obs['can_kick']:
            reward += 0.02
            stat[Metrics.Kick.name] = True

        pos = current_obs['position']
        if prev_obs['ammo'] > 0 and action == constants.Action.Bomb.value:
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
                            reward += 0.01
                            stat[Metrics.ExplodeWood.name] += 1
                            break

        return reward
