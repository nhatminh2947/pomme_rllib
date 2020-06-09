import numpy as np
import pommerman
from pommerman import agents
from pommerman import constants
from pommerman import utility
from ray.rllib.env.multi_agent_env import MultiAgentEnv


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
        self._step_count = 0
        self.action_space = self.env.action_space
        self.prev_obs = None

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
                obs[id] = self.featurize(_obs[id])
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

        if utility.position_in_items(self.prev_obs[id]['board'],
                                     current_obs[id]['position'],
                                     [constants.Item.IncrRange,
                                      constants.Item.ExtraBomb]):
            reward += 0.01

        if utility.position_in_items(self.prev_obs[id]['board'],
                                     current_obs[id]['position'],
                                     [constants.Item.Kick]):
            if not self.prev_obs[id]['can_kick']:
                reward += 0.02

        for i in range(10, 14):
            if i in self.alive_agents and i not in current_obs[id]['alive']:
                if constants.Item(value=i) in current_obs[id]['enemies']:
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
        self.prev_obs = self.env.reset()
        self._step_count = 0
        obs = {}

        for i in range(4):
            if self.is_agent_alive(i):
                obs[i] = self.featurize(self.prev_obs[i])

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
        features = RllibPomme.featurize(obs[0])
        for i in range(17):
            print(features[i])
        print()
        actions = env.act(obs)
        print(actions)
        obs, reward, done, info = env.step(actions)

        if done:
            break

    print(obs)
    features = RllibPomme.featurize(obs[0])
    for i in range(17):
        print(features[i])
    print()
    # print(PommeMultiAgent.featurize(obs[0]))