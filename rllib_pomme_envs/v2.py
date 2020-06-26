import ray
from pommerman import constants

from rllib_pomme_envs import v0
from utils import featurize


# Note: change team for training agents
class RllibPomme(v0.RllibPomme):
    def __init__(self, config):
        super().__init__(config)
        self.memory = None

    def step(self, action_dict):
        if self.is_render:
            self.render(record_pngs_dir="/home/lucius/ray_results/records/pngs",
                        record_json_dir="/home/lucius/ray_results/records/logs")

        actions = []
        for id in range(4):
            if self.agent_names[id] in action_dict:
                actions.append(int(action_dict[self.agent_names[id]]))
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
                self.update_memory(id, _obs[id])
                obs[self.agent_names[id]] = featurize(self.memory[id])
                rewards[self.agent_names[id]] = self.reward(id, _obs, _info)
                infos[self.agent_names[id]].update(_info)

        self.prev_obs = _obs

        return obs, rewards, dones, infos

    def update_memory(self, id, obs):
        self.memory[id]['alive'] = obs['alive']
        self.memory[id]['bomb_moving_direction'] = obs['bomb_moving_direction']
        self.memory[id]['position'] = obs['position']
        self.memory[id]['blast_strength'] = obs['blast_strength']
        self.memory[id]['can_kick'] = obs['can_kick']
        self.memory[id]['ammo'] = obs['ammo']

        for i in range(11):
            for j in range(11):
                if self.memory[id]['bomb_life'][i][j] == 0:
                    self.memory[id]['bomb_blast_strength'][i][j] = 0
                else:
                    self.memory[id]['bomb_life'][i, j] -= 1

                if self.memory[id]['flame_life'][i, j] != 0:
                    self.memory[id]['flame_life'][i, j] -= 1

        for i in range(11):
            for j in range(11):
                if obs['board'][i, j] != constants.Item.Fog.value:
                    self.memory[id]['board'][i, j] = obs['board'][i, j]
                    self.memory[id]['bomb_life'][i, j] = obs['bomb_life'][i, j]
                    self.memory[id]['bomb_blast_strength'][i, j] = obs['bomb_blast_strength'][i, j]
                    self.memory[id]['flame_life'][i, j] = obs['flame_life'][i, j]

                if constants.Item(self.memory[id]['board'][i, j]) in obs['enemies'] \
                        and self.memory[id]['board'][i, j] not in obs['alive']:
                    self.memory[id]['board'][i, j] = 0

    def init_memory(self, observations):
        self.memory = []
        for i in range(self.num_agents):
            self.memory.append(observations[i])

    def reset(self):
        self.prev_obs = self.env.reset()
        obs = {}
        self.reset_stat()
        g_helper = ray.util.get_actor("g_helper")
        self.agent_names = ray.get(g_helper.get_agent_names.remote())
        for i in range(self.num_agents):
            if self.is_agent_alive(i):
                obs[self.agent_names[i]] = featurize(self.prev_obs[i])
        self.init_memory(self.prev_obs)

        return obs
