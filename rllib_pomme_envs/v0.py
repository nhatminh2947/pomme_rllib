import pommerman
import ray
from gym import spaces
from pommerman import agents
from pommerman import constants
from pommerman import utility
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import utils
from metrics import Metrics
from utils import featurize


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
                rewards[self.agent_names[id]] = self.reward(id, actions, _obs, _info)
                infos[self.agent_names[id]] = _info

        self.prev_obs = _obs

        return obs, rewards, dones, infos

    @property
    def alive_agents(self):
        return self.prev_obs[0]['alive']

    def is_done(self, id, current_alive):
        return (id + 10) in self.alive_agents and (id + 10) not in current_alive

    def reward(self, id, actions, current_obs, info):
        reward = 0

        if self.prev_obs[id]['blast_strength'] < current_obs[id]['blast_strength']:
            reward += 0.01
            self.stat[id][Metrics.IncrRange.name] += 1

        if utility._position_is_item(self.prev_obs[id]['board'],
                                     current_obs[id]['position'],
                                     constants.Item.ExtraBomb):
            reward += 0.01
            self.stat[id][Metrics.ExtraBomb.name] += 1

        if not self.prev_obs[id]['can_kick'] and current_obs[id]['can_kick']:
            reward += 0.02
            self.stat[id][Metrics.Kick.name] = True

        for i in range(10, 14):
            if i in self.alive_agents and i not in current_obs[id]['alive']:
                if constants.Item(value=i) in current_obs[id]['enemies']:
                    reward += 0.75
                    self.stat[id][Metrics.EnemyDeath.name] += 1
                elif i - 10 == id:
                    reward += -1
                    self.stat[id][Metrics.DeadOrSuicide.name] += 1
                else:
                    reward += -0.5

        pos = current_obs[id]['position']
        if actions[id] == constants.Action.Bomb.value:
            dx = [-1, 0, 0, 1]
            dy = [0, -1, 1, 0]

            for i in range(4):
                row = pos[0] + dx[i]
                col = pos[1] + dy[i]
                if 0 <= row < 11 and 0 <= col < 11:
                    if current_obs[id]['board'][row, col] == constants.Item.Wood.value:
                        reward += 0.01

        if info['result'] == constants.Result.Tie:
            reward += -1

        return reward

    def is_agent_alive(self, id, alive_agents=None):
        if alive_agents is None:
            return (id + 10) in self.alive_agents
        return (id + 10) in alive_agents

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
                obs[self.agent_names[i]] = featurize(self.prev_obs[i])

        return obs


if __name__ == '__main__':
    agent_list = [
        agents.RandomAgent(),
        agents.StaticAgent(),
        agents.StaticAgent(),
        agents.StaticAgent()
    ]
    env = pommerman.make('PommeTeam-v0', agent_list,
                         # '/home/lucius/working/projects/pomme_rllib/resources/one_line_state.json'
                         )
    obs = env.reset()

    while True:
        features = featurize(obs[0])
        for i in range(16):
            print(features[i])
        print()
        actions = env.act(obs)
        print(actions)
        obs, reward, done, info = env.step(actions)

        if done:
            break

    print(obs)
    features = featurize(obs[0])
    for i in range(16):
        print(features[i])
    print()
    # print(PommeMultiAgent.featurize(obs[0]))
