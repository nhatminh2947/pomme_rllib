import ray
from pommerman import constants

from metrics import Metrics
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

        for id in range(self.num_agents):
            if actions[id] == constants.Action.Bomb.value:
                self.stat[id][Metrics.NumBombs.name] += 1

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
                rewards[self.agent_names[id]] = self.reward(id, actions, _obs, _info)
                infos[self.agent_names[id]].update(_info)

        self.prev_obs = _obs

        return obs, rewards, dones, infos

    # Note: add memory of enemy position of some previous steps
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
                    self.memory[id]['bomb_moving_direction'][i][j] = 0
                else:
                    self.memory[id]['bomb_life'][i, j] -= 1

                if self.memory[id]['flame_life'][i, j] != 0:
                    self.memory[id]['flame_life'][i, j] -= 1

        for enemy in obs['enemies']:
            if obs['board'][obs['board'] == enemy.value].any() or enemy.value not in obs['alive']:
                self.memory[id]['board'][self.memory[id]['board'] == enemy.value] = constants.Item.Passage.value

        self.memory[id]['board'][obs['board'] != constants.Item.Fog.value] = obs['board'][
            obs['board'] != constants.Item.Fog.value]
        self.memory[id]['bomb_life'][obs['board'] != constants.Item.Fog.value] = obs['bomb_life'][
            obs['board'] != constants.Item.Fog.value]
        self.memory[id]['bomb_blast_strength'][obs['board'] != constants.Item.Fog.value] = obs['bomb_blast_strength'][
            obs['board'] != constants.Item.Fog.value]
        self.memory[id]['flame_life'][obs['board'] != constants.Item.Fog.value] = obs['flame_life'][
            obs['board'] != constants.Item.Fog.value]

    def init_memory(self, observations):
        self.memory = []
        team_pos = [[(1, 1), (9, 9)],
                    [(9, 1), (1, 9)]]
        enemies = [[11, 13],
                   [10, 12]]
        for i in range(self.num_agents):
            for j in range(2):
                observations[i]['board'][team_pos[1 - (i % 2)][j]] = enemies[i % 2][j]
            self.memory.append(observations[i])

    def reset(self):
        self.prev_obs = self.env.reset()
        obs = {}
        self.reset_stat()
        g_helper = ray.util.get_actor("g_helper")
        self.agent_names = ray.get(g_helper.get_agent_names.remote())
        self.init_memory(self.prev_obs)
        for i in range(self.num_agents):
            if self.is_agent_alive(i):
                obs[self.agent_names[i]] = featurize(self.memory[i])

        return obs
