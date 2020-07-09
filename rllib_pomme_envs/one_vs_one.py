import ray
from pommerman import constants

from metrics import Metrics
from rllib_pomme_envs import v0
from utils import featurize_for_rms, featurize, featurize_v1, featurize_v2, featurize_v3, featurize_v4
from pommerman import utility


# Note: change team for training agents
class RllibPomme(v0.RllibPomme):
    def __init__(self, config):
        super().__init__(config)

    def step(self, action_dict):
        if self.is_render:
            self.render()

        actions = []
        for id in range(self.num_agents):
            if self.agent_names[id] in action_dict:
                actions.append(int(action_dict[self.agent_names[id]]))
                if action_dict[self.agent_names[id]] == constants.Action.Bomb.value and self.prev_obs[id]['ammo'] > 0:
                    self.stat[id][Metrics.RealBombs.name] += 1
            else:
                actions.append(0)

        obs = {}
        rewards = {}
        dones = {}
        infos = {self.agent_names[i - 10]: {} for i in self.prev_obs[0]['alive']}

        _obs, _reward, _done, _info = self.env.step(actions)

        for id in self.prev_obs[0]['alive']:
            if _done or self.is_done(id - 10, _obs[0]['alive']):
                dones[self.agent_names[id - 10]] = True
                infos[self.agent_names[id - 10]]["metrics"] = self.stat[id - 10]

        dones["__all__"] = _done

        for id in range(self.num_agents):
            if self.is_agent_alive(id):
                obs[self.agent_names[id]] = featurize_v4(_obs[id])
                rewards[self.agent_names[id]] = self.reward(id, _obs, _info)
                infos[self.agent_names[id]].update(_info)

        self.prev_obs = _obs

        return obs, rewards, dones, infos

    def reset(self):
        self.prev_obs = self.env.reset()
        obs = {}
        self.reset_stat()
        g_helper = ray.util.get_actor("g_helper")
        self.agent_names = ray.get(g_helper.get_agent_names.remote())
        # print("Called reset")
        # print("self.agent_name:", self.agent_names)
        for i in range(self.num_agents):
            if self.is_agent_alive(i):
                obs[self.agent_names[i]] = featurize_v4(self.prev_obs[i])

        return obs

    def reward(self, id, current_obs, info):
        reward = 0

        if utility._position_is_item(self.prev_obs[id]['board'],
                                     current_obs[id]['position'],
                                     constants.Item.IncrRange):
            reward += 0.01
            self.stat[id][Metrics.IncrRange.name] += 1

        if utility._position_is_item(self.prev_obs[id]['board'],
                                     current_obs[id]['position'],
                                     constants.Item.ExtraBomb):
            reward += 0.01
            self.stat[id][Metrics.ExtraBomb.name] += 1

        if utility._position_is_item(self.prev_obs[id]['board'],
                                     current_obs[id]['position'],
                                     constants.Item.Kick) and not self.prev_obs[id]['can_kick']:
            reward += 0.02
            self.stat[id][Metrics.Kick.name] = True

        for i in range(10, 12):
            if i in self.alive_agents and i not in current_obs[id]['alive']:
                if constants.Item(value=i) in current_obs[id]['enemies']:
                    reward += 1
                    self.stat[id][Metrics.EnemyDeath.name] += 1
                elif i - 10 == id:
                    reward += -1
                    self.stat[id][Metrics.DeadOrSuicide.name] += 1

        if info['result'] == constants.Result.Tie:
            reward += -1

        return reward
