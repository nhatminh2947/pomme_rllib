import numpy as np
import pommerman
from pommerman import agents
from pommerman import constants
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Ability:
    def __init__(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False

    def reset(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False


class PommeMultiAgent(MultiAgentEnv):
    def __init__(self, config):
        self.agent_list = [
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
        ]
        self.env = pommerman.make(config["env_id"], self.agent_list, config["game_state_file"])

        self.is_render = config["render"]
        self._step_count = 0
        self.action_space = self.env.action_space
        self.alive_agents = [10, 11, 12, 13]

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

        if constants.Item.Agent0.value not in _obs[0]['alive']:
            _info['result'] = constants.Result.Loss
            dones[0] = True
            _done = True

        dones["__all__"] = _done
        for id in range(4):
            if self.is_agent_alive(id):
                obs[id] = self.featurize(_obs[id])
                # rewards[id] = _reward[id]  # self.reward(id, self.alive_agents, _obs[id], _info)
                rewards[id] = self.reward(id, self.alive_agents, _obs[id], _info)
                infos[id] = _info

        self.alive_agents = _obs[0]['alive']

        return obs, rewards, dones, infos

    def reward(self, id, alive, obs, info):
        reward = 0

        for i in range(10, 14):
            if i in alive and i not in obs['alive']:
                if constants.Item(value=i) in obs['enemies']:
                    reward += 0.5
                elif i - 10 == id:
                    reward += -1
                else:
                    reward += -0.5

        if info['result'] == constants.Result.Tie:
            reward += -1

        return reward

    # Meaning of channels
    # 0 passage             fow
    # 1 Rigid               fow
    # 2 Wood                fow
    # 3 ExtraBomb           fow
    # 4 IncrRange           fow
    # 5 Kick                fow
    # 6 FlameLife           fow
    # 7 BombLife            fow
    # 8 BombBlastStrength   fow
    # 9 Fog
    # 10 Position
    # 11 Teammate
    # 12 Enemies
    # 13 Ammo
    # 14 BlastStrength
    # 15 CanKick
    @staticmethod
    def featurize(obs):
        board = obs['board']
        features = []
        board_items = [constants.Item.Passage,
                       constants.Item.Rigid,
                       constants.Item.Wood,
                       constants.Item.Fog,
                       constants.Item.ExtraBomb,
                       constants.Item.IncrRange,
                       constants.Item.Kick]

        for item in board_items:
            features.append(board == item.value)

        for feature in ["bomb_life", "bomb_blast_strength", "bomb_moving_direction", "flame_life"]:
            features.append(obs[feature])

        position = np.zeros(board.shape)
        position[obs["position"]] = 1
        features.append(position)

        features.append(board == obs["teammate"].value)

        enemies = np.zeros(board.shape)
        for enemy in obs["enemies"]:
            enemies[(board == enemy.value)] = 1
        features.append(enemies)

        features.append(np.full(board.shape, fill_value=obs["ammo"]))
        features.append(np.full(board.shape, fill_value=obs["blast_strength"]))
        features.append(np.full(board.shape, fill_value=(1 if obs["can_kick"] else 0)))

        return np.stack(features, 0)

    def is_agent_alive(self, id, alive_agents=None):
        if alive_agents is None:
            return (id + 10) in self.alive_agents
        return (id + 10) in alive_agents

    def reset(self):
        _obs = self.env.reset()
        self._step_count = 0
        self.alive_agents = _obs[0]['alive']
        obs = {}

        for i in range(4):
            if self.is_agent_alive(i):
                obs[i] = self.featurize(_obs[i])

        return obs


if __name__ == '__main__':
    agent_list = [
        agents.RandomAgent(),
        agents.StaticAgent(),
        agents.StaticAgent(),
        agents.StaticAgent()
    ]
    env = pommerman.make('PommeTeam-v0', agent_list,
                         '/home/lucius/working/projects/pomme_rllib/resources/one_line_state.json')
    obs = env.reset()

    while True:
        features = PommeMultiAgent.featurize(obs[0])
        for i in range(17):
            print(features[i])
        print()
        actions = env.act(obs)
        print(actions)
        obs, reward, done, info = env.step(actions)

        if done:
            break

    print(obs)
    features = PommeMultiAgent.featurize(obs[0])
    for i in range(17):
        print(features[i])
    print()
    # print(PommeMultiAgent.featurize(obs[0]))
