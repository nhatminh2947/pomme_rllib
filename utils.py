from typing import Dict

import numpy as np
import ray
from pommerman import constants
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

from metrics import Metrics

# NUM_FEATURES = 20
NUM_FEATURES = 21

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
    features.append(np.full(board.shape, fill_value=(alive_enemies / 2)))

    for i in range(1, 5):
        features.append(obs["bomb_moving_direction"] == i)

    # normal features
    for feature, max_value in zip(["bomb_life", "bomb_blast_strength", "flame_life"],
                                  [9, 20, 3]):
        features.append(obs[feature] / max_value)

    features.append(np.full(board.shape, fill_value=(obs["ammo"] / 20.0)))
    features.append(np.full(board.shape, fill_value=(obs["blast_strength"] / 20.0)))
    features.append(np.full(board.shape, fill_value=(1 if obs["can_kick"] else 0)))

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


def check_nan(array):
    tmp = np.sum(array)
    if np.isnan(tmp) or np.isinf(tmp):
        print('NaN or Inf found in input tensor.')
        print(tmp)
    return array


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


def center(obs):
    centered_obs = np.copy(obs)

    centered_obs["board"] = np.ones((9, 9), dtype=np.float32)
    centered_obs["bomb_blast_strength"] = np.ones((9, 9), dtype=np.float32)
    centered_obs["bomb_life"] = np.ones((9, 9), dtype=np.float32)
    centered_obs["bomb_moving_direction"] = np.ones((9, 9), dtype=np.float32)

    x, y = obs["position"]
    centered_obs['board'][max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] = obs["board"][
                                                                                        max(0, x - 4):min(11, x + 5),
                                                                                        max(0, y - 4):min(11, y + 5)]

    centered_obs["bomb_blast_strength"][max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] \
        = obs["bomb_blast_strength"].astype(np.float32)[max(0, x - 4):min(11, x + 5), max(0, y - 4):min(11, y + 5)]

    centered_obs["bomb_life"][max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] \
        = obs["bomb_life"].astype(np.float32)[max(0, x - 4):min(11, x + 5), max(0, y - 4):min(11, y + 5)]

    centered_obs["bomb_moving_direction"][max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] \
        = obs["bomb_life"].astype(np.float32)[max(0, x - 4):min(11, x + 5), max(0, y - 4):min(11, y + 5)]

    return centered_obs
