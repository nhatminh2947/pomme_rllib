import ray
from pommerman import constants

from memory import Memory
from metrics import Metrics
from rllib_pomme_envs import v0
from utils import featurize_v4


# Note: change team for training agents
class RllibPomme(v0.RllibPomme):
    def __init__(self, config):
        super().__init__(config)
        self.memory = [
            Memory(i) for i in range(self.num_agents)
        ]

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
                self.stat[id][Metrics.ActionBombs.name] += 1
                if self.prev_obs[id]['ammo'] > 0:
                    self.stat[id][Metrics.RealBombs.name] += 1

        obs = {}
        rewards = {}
        dones = {}
        infos = {self.agent_names[i - 10]: {} for i in self.prev_obs[0]['alive']}

        _obs, _reward, _done, _info = self.env.step(actions)

        for id in self.prev_obs[0]['alive']:
            if _done or id not in _obs[id - 10]['alive']:
                dones[self.agent_names[id - 10]] = True
                infos[self.agent_names[id - 10]]["metrics"] = self.stat[id - 10]

        dones["__all__"] = _done

        for id in range(self.num_agents):
            if self.is_agent_alive(id, self.prev_obs[id]['alive']):
                # self.memory[id].update_memory(_obs[id])
                if self.agent_names[id] == "simple":
                    obs[self.agent_names[id]] = _obs[id]
                else:
                    obs[self.agent_names[id]] = featurize_v4(_obs[id], centering=True)
                rewards[self.agent_names[id]] = self.reward(id, actions[id], self.prev_obs[id],
                                                            _obs[id], _info, self.stat[id])
                infos[self.agent_names[id]].update(_info)

        self.prev_obs = _obs

        return obs, rewards, dones, infos

    def reset(self):
        self.prev_obs = self.env.reset()
        obs = {}
        self.reset_stat()
        g_helper = ray.util.get_actor("g_helper")
        self.agent_names = ray.get(g_helper.get_agent_names.remote())

        # if np.random.random() > 0.5:
        #     self.agent_names[0], self.agent_names[1] = self.agent_names[1], self.agent_names[0]
        #     self.agent_names[2], self.agent_names[3] = self.agent_names[3], self.agent_names[2]

        for i in range(self.num_agents):
            # self.memory[i].init_memory(self.prev_obs[i])
            if self.is_agent_alive(i, self.prev_obs[i]['alive']):
                obs[self.agent_names[i]] = featurize_v4(self.agent_names[i], self.prev_obs[i], centering=True)

        return obs

    def reward(self, id, action, prev_obs, current_obs, info, stat):
        reward = 0

        reward += self.immediate_reward(action, prev_obs, current_obs, stat)

        if id + 10 in prev_obs['alive'] and id + 10 not in current_obs['alive']:
            reward += -1
            stat[Metrics.DeadOrSuicide.name] += 1
            for i in range(10, 14):
                if constants.Item(value=i) in current_obs['enemies'] and i not in current_obs['alive']:
                    reward += 0.5
                    stat[Metrics.EnemyDeath.name] += 1
        elif info['result'] == constants.Result.Win or info['result'] == constants.Result.Tie:
            for i in range(10, 14):
                if constants.Item(value=i) in current_obs['enemies'] and i not in current_obs['alive']:
                    reward += 0.5
                    stat[Metrics.EnemyDeath.name] += 1

            if info['result'] == constants.Result.Tie:
                reward += -1

        if action == constants.Action.Bomb.value:
            stat[Metrics.ActionBombs.name] += 1
            if prev_obs['ammo'] > 0:
                stat[Metrics.RealBombs.name] += 1

        return reward
