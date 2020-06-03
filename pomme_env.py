import unittest

import gym
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pommerman import utility
import pommerman
from pommerman import agents
from pommerman import constants


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
        self.env = pommerman.make(config["env_id"], self.agent_list)

        self.is_render = config["render"]
        self._step_count = 0
        self._max_steps = self.env._max_steps
        self.action_space = self.env.action_space
        self.eliminated = []
        self.alive_agents = [10, 11, 12, 13]
        self.ability = [Ability()] * 4

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

        _obs, _reward, _done, _info = self.env.step(actions)

        if constants.Item.Agent0.value not in _obs[0]['alive']:
            if not _done:
                _info['result'] = constants.Result.Loss
            _done = True

        obs = {}
        rewards = {}
        dones = {"__all__": _done}
        infos = {}

        for id in range(4):
            if (id + 10) in self.alive_agents:
                obs[id] = self.featurize(_obs[id])
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

    def update_fow(self, obs):
        for i in range(4):
            if i in list(obs.keys()):
                for channel in range(16):
                    if channel < 9:
                        self.fow[i][channel][obs[i][channel] != 0] = obs[i][channel][obs[i][channel] != 0]
                        # print(self.fow[i][channel])
                    else:
                        self.fow[i][channel] = obs[i][channel]
                        # print(self.fow[i][channel])
            else:
                self.fow.pop(i, None)

    def reward_shaping(self, agent_id, new_obs, prev_board, info):
        reward = 0
        current_alive_agents = np.asarray(new_obs['alive']) - constants.Item.Agent0.value

        if info['result'] == constants.Result.Tie:
            return -1

        if agent_id not in current_alive_agents:
            return -1

        if agent_id % 2 == 0:
            enemies = [1, 3]
        else:
            enemies = [0, 2]

        if utility.position_is_powerup(prev_board, new_obs['position']):
            if constants.Item(prev_board[new_obs['position']]) == constants.Item.IncrRange:
                reward += 0.01
                self.ability[agent_id].blast_strength += 1
            elif constants.Item(prev_board[new_obs['position']]) == constants.Item.ExtraBomb:
                reward += 0.01
                self.ability[agent_id].ammo += 1
            elif not self.ability[agent_id].can_kick and constants.Item(
                    prev_board[new_obs['position']]) == constants.Item.Kick:
                reward += 0.05
                self.ability[agent_id].can_kick = True

        for enemy in enemies:
            if enemy not in current_alive_agents and enemy not in self.eliminated:
                reward += 0.5

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
                       constants.Item.Bomb,
                       constants.Item.Flames,
                       constants.Item.Fog,
                       constants.Item.ExtraBomb,
                       constants.Item.IncrRange,
                       constants.Item.Kick]

        for item in board_items:
            features.append(board == item.value)

        for feature in ["bomb_life", "bomb_blast_strength"]:
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

    def reset(self):
        _obs = self.env.reset()
        self._step_count = 0
        self.eliminated = []
        self.alive_agents = [10, 11, 12, 13]
        obs = {}

        for i in range(4):
            self.ability[i].reset()
            obs[i] = self.featurize(_obs[i])

        return obs


if __name__ == '__main__':
    agent_list = [
        agents.StaticAgent(),
        agents.StaticAgent(),
        agents.StaticAgent(),
        agents.StaticAgent()
    ]
    env = pommerman.make('PommeTeam-v0', agent_list)
    obs = env.reset()
    features = PommeMultiAgent.featurize(obs[0])
    # print(PommeMultiAgent.featurize(obs[0]))
