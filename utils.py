from typing import Dict

import numpy as np
import ray
import torch
from pommerman import constants
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from torch import nn

from metrics import Metrics

NUM_FEATURES = 19
# NUM_FEATURES = 21

agents_1 = ["cinjon-simpleagent", "hakozakijunctions", "eisenach", "dypm.1", "navocado", "skynet955",
            "nips19-gorogm.gorogm", "nips19-pauljasek.thing1andthing2", "nips19-sumedhgupta.neoterics",
            "nips19-inspir-ai.inspir"]
agents_2 = ["cinjon-simpleagent", "hakozakijunctions", "eisenach", "dypm.2", "navocado", "skynet955",
            "nips19-gorogm.gorogm", "nips19-pauljasek.thing1andthing2", "nips19-sumedhgupta.neoterics",
            "nips19-inspir-ai.inspir"]


def featurize(obs):
    board = obs['board']
    features = []

    # binary features
    board_items = [constants.Item.Passage,
                   constants.Item.Rigid,
                   constants.Item.Wood,
                   constants.Item.ExtraBomb,
                   constants.Item.IncrRange,
                   constants.Item.Kick]

    for item in board_items:
        features.append(board == item.value)

    # Set walkable feature plan for extrabomb, incrange, kick and bomb if can kick
    for i in range(board.shape[0]):
        for j in range(board.shape[0]):
            if board[i, j] in [constants.Item.ExtraBomb.value,
                               constants.Item.IncrRange.value,
                               constants.Item.Kick.value]:
                features[0][i, j] = 1
            if obs['can_kick'] and board[i, j] == constants.Item.Bomb.value:
                features[0][i, j] = 1

    position = np.zeros(board.shape)
    position[obs["position"]] = 1
    features.append(position)

    features.append(board == obs["teammate"].value)
    features.append(np.full(board.shape, fill_value=1 if obs["teammate"].value in obs["alive"] else 0))

    alive_enemies = 0
    enemies = np.zeros(board.shape)
    for enemy in obs["enemies"]:
        enemies[(board == enemy.value)] = 1
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in obs['alive']:
            alive_enemies += 1
    features.append(enemies)
    features.append(np.full(board.shape, fill_value=alive_enemies / 2))

    for i in range(1, 5):
        features.append(obs["bomb_moving_direction"] == i)

    # normal features
    for feature, max_value in zip(["bomb_life", "bomb_blast_strength", "flame_life"],
                                  [9, 20, 3]):
        features.append(obs[feature] / max_value)

    features.append(np.full(board.shape, fill_value=obs["ammo"] / 20))
    features.append(np.full(board.shape, fill_value=obs["blast_strength"] / 20))
    features.append(np.full(board.shape, fill_value=1 if obs["can_kick"] else 0))

    features = np.stack(features, 0)
    features = np.asarray(features, dtype=np.float)

    return features


def featurize_for_rms(obs):
    board = obs['board']
    features = []

    # binary features
    board_items = [constants.Item.Passage,
                   constants.Item.Rigid,
                   constants.Item.Wood,
                   constants.Item.ExtraBomb,
                   constants.Item.IncrRange,
                   constants.Item.Kick]

    for item in board_items:
        features.append(board == item.value)

    # Set walkable feature plan for extrabomb, incrange, kick and bomb if can kick
    for i in range(board.shape[0]):
        for j in range(board.shape[0]):
            if board[i, j] in [constants.Item.ExtraBomb.value,
                               constants.Item.IncrRange.value,
                               constants.Item.Kick.value]:
                features[0][i, j] = 1
            if obs['can_kick'] and board[i, j] == constants.Item.Bomb.value:
                features[0][i, j] = 1

    position = np.zeros(board.shape)
    position[obs["position"]] = 1
    features.append(position)

    features.append(board == obs["teammate"].value)
    features.append(np.full(board.shape, fill_value=1 if obs["teammate"].value in obs["alive"] else 0))

    alive_enemies = 0
    enemies = np.zeros(board.shape)
    for enemy in obs["enemies"]:
        enemies[(board == enemy.value)] = 1
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in obs['alive']:
            alive_enemies += 1
    features.append(enemies)
    features.append(np.full(board.shape, fill_value=alive_enemies))

    for i in range(1, 5):
        features.append(obs["bomb_moving_direction"] == i)

    # normal features
    for feature in ["bomb_life", "bomb_blast_strength", "flame_life"]:
        features.append(obs[feature])

    features.append(np.full(board.shape, fill_value=obs["ammo"]))
    features.append(np.full(board.shape, fill_value=obs["blast_strength"]))
    features.append(np.full(board.shape, fill_value=1 if obs["can_kick"] else 0))

    features = np.stack(features, 0)
    features = np.asarray(features, dtype=np.float)

    return features


# Working
def featurize_v1(obs):
    board = np.asarray(obs['board'], dtype=np.int)
    one_hot_board = nn.functional.one_hot(torch.tensor(board), 14).transpose(0, 2).transpose(1, 2).numpy()

    position = np.zeros(board.shape)
    position[obs["position"]] = 1
    teammate = board == obs["teammate"].value
    teammate_alive = np.full(board.shape, fill_value=1 if obs["teammate"].value in obs["alive"] else 0)

    alive_enemies = 0
    enemies = np.zeros(board.shape)
    for enemy in obs["enemies"]:
        enemies[(board == enemy.value)] = 1
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in obs['alive']:
            alive_enemies += 1

    enemies_alive = np.full(board.shape, fill_value=alive_enemies)

    ammo = np.full(board.shape, fill_value=obs["ammo"])
    blast_strength = np.full(board.shape, fill_value=obs["blast_strength"])
    can_kick = np.full(board.shape, fill_value=1 if obs["can_kick"] else 0)

    features = np.stack([obs["bomb_life"],
                         obs["bomb_blast_strength"],
                         position,
                         teammate_alive,
                         enemies_alive,
                         ammo, blast_strength, can_kick], 0)

    features = np.concatenate([one_hot_board, features], 0)

    return features


# working without switch side
def featurize_v2(obs):
    board = np.asarray(obs['board'], dtype=np.float)
    one_hot_board = nn.functional.one_hot(torch.tensor(board), 14).transpose(0, 2).transpose(1, 2).numpy()
    one_hot_board = one_hot_board[:9]

    position = np.zeros(board.shape, dtype=np.float)
    position[obs["position"]] = 1
    teammate = np.asarray(board == obs["teammate"].value, dtype=np.float)
    teammate_alive = np.full(board.shape,
                             fill_value=1 if obs["teammate"].value in obs["alive"] else 0,
                             dtype=np.float)

    alive_enemies = 0
    enemies = np.zeros(board.shape, dtype=np.float)
    for enemy in obs["enemies"]:
        enemies[(board == enemy.value)] = 1
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in obs['alive']:
            alive_enemies += 1

    enemies_alive = np.full(board.shape, fill_value=alive_enemies, dtype=np.float)

    ammo = np.full(board.shape, fill_value=obs["ammo"], dtype=np.float)
    blast_strength = np.full(board.shape, fill_value=obs["blast_strength"], dtype=np.float)
    can_kick = np.full(board.shape, fill_value=1 if obs["can_kick"] else 0, dtype=np.float)

    features = np.stack([obs["bomb_life"],
                         obs["bomb_blast_strength"],
                         position,
                         teammate_alive,
                         teammate,
                         enemies_alive,
                         enemies,
                         ammo, blast_strength, can_kick], 0)

    features = np.concatenate([one_hot_board, features], 0)

    return features


def featurize_v3(obs):
    board = np.asarray(obs['board'], dtype=np.int)
    one_hot_board = nn.functional.one_hot(torch.tensor(board), 14).transpose(0, 2).transpose(1, 2).numpy()
    one_hot_board = np.delete(one_hot_board, 9, 0)

    position = np.zeros(board.shape)
    position[obs["position"]] = 1
    teammate = board == obs["teammate"].value
    teammate_alive = np.full(board.shape, fill_value=1 if obs["teammate"].value in obs["alive"] else 0)

    alive_enemies = 0
    enemies = np.zeros(board.shape)
    for enemy in obs["enemies"]:
        enemies[(board == enemy.value)] = 1
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in obs['alive']:
            alive_enemies += 1

    enemies_alive = np.full(board.shape, fill_value=alive_enemies)

    ammo = np.full(board.shape, fill_value=obs["ammo"])
    blast_strength = np.full(board.shape, fill_value=obs["blast_strength"])
    can_kick = np.full(board.shape, fill_value=1 if obs["can_kick"] else 0)

    features = np.stack([obs["bomb_life"],
                         obs["bomb_blast_strength"],
                         teammate_alive,
                         enemies_alive,
                         ammo, blast_strength, can_kick], 0)

    features = np.concatenate([one_hot_board, features], 0)

    return features


def featurize_v4(obs, centering=False):
    agent_id = obs['board'][obs["position"]]

    preprocessed_obs = obs.copy()
    if centering:
        preprocessed_obs = center(obs, size=11)

    board = np.asarray(preprocessed_obs['board'], dtype=np.int)

    one_hot_board = nn.functional.one_hot(torch.tensor(board), 14).transpose(0, 2).transpose(1, 2).numpy()

    if agent_id % 2 == 1:
        one_hot_board[[10, 11]] = one_hot_board[[11, 10]]
        one_hot_board[[12, 13]] = one_hot_board[[13, 12]]

    if agent_id == 12 or agent_id == 13:
        one_hot_board[[10, 12]] = one_hot_board[[12, 10]]

    one_hot_board = np.delete(one_hot_board, 9, 0)

    teammate_alive = np.full(board.shape,
                             fill_value=1 if preprocessed_obs["teammate"].value in preprocessed_obs["alive"] else 0)

    alive_enemies = 0
    for enemy in preprocessed_obs["enemies"]:
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in preprocessed_obs['alive']:
            alive_enemies += 1

    enemies_alive = np.full(board.shape, fill_value=alive_enemies)

    ammo = np.full(board.shape, fill_value=preprocessed_obs["ammo"])
    blast_strength = np.full(board.shape, fill_value=preprocessed_obs["blast_strength"])
    can_kick = np.full(board.shape, fill_value=1 if preprocessed_obs["can_kick"] else 0)

    features = np.stack([preprocessed_obs["bomb_life"],
                         preprocessed_obs["bomb_blast_strength"],
                         teammate_alive,
                         enemies_alive,
                         ammo, blast_strength, can_kick], 0)

    features = np.concatenate([one_hot_board, features], 0)

    return features


def featurize_v5(obs, centering=False, view_range=9):
    agent_id = obs['board'][obs["position"]]

    preprocessed_obs = obs.copy()
    if centering:
        preprocessed_obs = center(obs, view_range)

    board = np.asarray(preprocessed_obs['board'], dtype=np.int)

    one_hot_board = nn.functional.one_hot(torch.tensor(board), 14).transpose(0, 2).transpose(1, 2).numpy()

    if agent_id % 2 == 1:
        one_hot_board[[10, 11]] = one_hot_board[[11, 10]]
        one_hot_board[[12, 13]] = one_hot_board[[13, 12]]

    if agent_id == 12 or agent_id == 13:
        one_hot_board[[10, 12]] = one_hot_board[[12, 10]]

    one_hot_board[13] = one_hot_board[11] + one_hot_board[13]

    one_hot_board = np.delete(one_hot_board, [9, 11], 0)

    teammate_alive = np.full(board.shape,
                             fill_value=1 if preprocessed_obs["teammate"].value in preprocessed_obs["alive"] else 0)

    alive_enemies = 0
    for enemy in preprocessed_obs["enemies"]:
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in preprocessed_obs['alive']:
            alive_enemies += 1

    enemies_alive = np.full(board.shape, fill_value=alive_enemies)

    ammo = np.full(board.shape, fill_value=preprocessed_obs["ammo"])
    blast_strength = np.full(board.shape, fill_value=preprocessed_obs["blast_strength"])
    can_kick = np.full(board.shape, fill_value=1 if preprocessed_obs["can_kick"] else 0)

    features = np.stack([preprocessed_obs["bomb_life"],
                         preprocessed_obs["bomb_blast_strength"],
                         teammate_alive,
                         enemies_alive,
                         ammo, blast_strength, can_kick], 0)

    features = np.concatenate([one_hot_board, features], 0)

    return features


class PommeCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        winners = None

        for k, v in episode.agent_rewards.items():
            agent_name = k[0]
            name = agent_name.split("_")
            if name[0] == "opponent":
                continue
            # print(episode.last_info_for(agent_name))
            info = episode.last_info_for(agent_name)

            if "winners" in info:
                winners = info["winners"]

            agent_stat = info["metrics"]

            for key in Metrics:
                episode.custom_metrics["agent_{}/{}".format(agent_name, key.name)] = agent_stat[key.name]

        if winners is None:
            episode.custom_metrics["Tie"] = 1
        elif winners == [0, 2]:
            episode.custom_metrics["team_0_win"] = 1
            episode.custom_metrics["team_1_win"] = 0
        else:
            episode.custom_metrics["team_0_win"] = 0
            episode.custom_metrics["team_1_win"] = 1

    def on_train_result(self, trainer, result: dict, **kwargs):
        g_helper = ray.util.get_actor("g_helper")
        g_helper.set_agent_names.remote()


def limit_gamma_explore(config):
    config["gamma"] = min(config["gamma"], 0.999)
    return config


def policy_mapping(agent_id):
    # agent_id pattern training/opponent_policy-id_agent-num
    # print("Calling to policy mapping {}".format(agent_id))
    parts = agent_id.split("_")
    team = int(parts[1])

    if parts[0] == "training":
        return "policy_{}".format(team)
    return "static"


def center(obs, size=9):
    centered_obs = obs.copy()

    centered_obs["board"] = np.ones((size, size), dtype=np.float32)
    centered_obs["bomb_blast_strength"] = np.zeros((size, size), dtype=np.float32)
    centered_obs["bomb_life"] = np.zeros((size, size), dtype=np.float32)
    centered_obs["bomb_moving_direction"] = np.zeros((size, size), dtype=np.float32)

    x, y = obs["position"]

    u = size // 2
    v = 11 + u

    centered_obs['board'][max(0, u - x):min(size, v - x), max(0, u - y):min(size, v - y)] \
        = obs["board"][max(0, x - u):min(11, x + u + 1), max(0, y - u):min(11, y + u + 1)]

    centered_obs['bomb_blast_strength'][max(0, u - x):min(size, v - x), max(0, u - y):min(size, v - y)] \
        = obs["bomb_blast_strength"][max(0, x - u):min(11, x + u + 1), max(0, y - u):min(11, y + u + 1)]

    centered_obs['bomb_life'][max(0, u - x):min(size, v - x), max(0, u - y):min(size, v - y)] \
        = obs["bomb_life"][max(0, x - u):min(11, x + u + 1), max(0, y - u):min(11, y + u + 1)]

    centered_obs['bomb_moving_direction'][max(0, u - x):min(size, v - x), max(0, u - y):min(size, v - y)] \
        = obs["bomb_moving_direction"][max(0, x - u):min(11, x + u + 1), max(0, y - u):min(11, y + u + 1)]

    return centered_obs
