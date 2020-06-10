import pommerman
from gym import spaces
from pommerman import agents
from pommerman import constants
from pommerman import utility
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from metrics import Metrics
from utils import featurize


# Note: change team for training agents
class RllibPomme(MultiAgentEnv):
    def __init__(self, config):
        self.agent_list = [
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
        ]
        self.env = pommerman.make(config["env_id"], self.agent_list, config["game_state_file"])
        self.is_render = config["render"]
        self.action_space = self.env.action_space
        self.prev_obs = None
        self.stat = None
        self.reset_stat()
        self.observation_space = spaces.Box(low=0, high=20, shape=(17, 11, 11))

    def reset_stat(self):
        self.stat = []
        for i in range(4):
            metrics = {}
            for key in Metrics:
                metrics[key.name] = 0
            self.stat.append(metrics)

    def render(self):
        self.env.render()

    def step(self, action_dict):
        if self.is_render:
            self.render()

        actions = []
        for id in range(4):
            if id in action_dict:
                actions.append(action_dict[id])
            else:
                actions.append(0)

        obs = {}
        rewards = {}
        dones = {}
        infos = {}

        _obs, _reward, _done, _info = self.env.step(actions)

        for id in range(4):
            if self.is_done(id, _obs[0]['alive']):
                dones[id] = True

                if id == 0:
                    _done = True
                    _info['result'] = constants.Result.Loss

        dones["__all__"] = _done

        for id in range(4):
            if self.is_agent_alive(id):
                obs[id] = featurize(_obs[id])
                rewards[id] = self.reward(id, _obs, _info)
                infos[id] = _info

        self.prev_obs = _obs

        return obs, rewards, dones, infos

    @property
    def alive_agents(self):
        return self.prev_obs[0]['alive']

    def is_done(self, id, current_alive):
        return (id + 10) in self.alive_agents and (id + 10) not in current_alive

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

        for i in range(10, 14):
            if i in self.alive_agents and i not in current_obs[id]['alive']:
                if constants.Item(value=i) in current_obs[id]['enemies']:
                    reward += 0.5
                    self.stat[id][Metrics.EnemyDeath.name] += 1
                elif i - 10 == id:
                    reward += -1
                    self.stat[id][Metrics.DeadOrSuicide.name] += 1
                else:
                    reward += -0.5
                    self.stat[id][Metrics.AllyDeath.name] += 1

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
        for i in range(4):
            if self.is_agent_alive(i):
                obs[i] = featurize(self.prev_obs[i])

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
        for i in range(17):
            print(features[i])
        print()
        actions = env.act(obs)
        print(actions)
        obs, reward, done, info = env.step(actions)

        if done:
            break

    print(obs)
    features = featurize(obs[0])
    for i in range(17):
        print(features[i])
    print()
    # print(PommeMultiAgent.featurize(obs[0]))
