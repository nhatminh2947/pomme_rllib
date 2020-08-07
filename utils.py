import copy

import numpy as np
import pommerman
import torch
from gym.spaces import Dict, Discrete, Box
from pommerman import constants
from torch import nn

from agents.static_agent import StaticAgent

NUM_FEATURES = 23 + 16
# NUM_FEATURES = 21

agents_1 = ["cinjon-simpleagent", "hakozakijunctions", "eisenach", "dypm.1", "navocado", "skynet955",
            "nips19-gorogm.gorogm", "nips19-pauljasek.thing1andthing2", "nips19-sumedhgupta.neoterics",
            "nips19-inspir-ai.inspir"]
agents_2 = ["cinjon-simpleagent", "hakozakijunctions", "eisenach", "dypm.2", "navocado", "skynet955",
            "nips19-gorogm.gorogm", "nips19-pauljasek.thing1andthing2", "nips19-sumedhgupta.neoterics",
            "nips19-inspir-ai.inspir"]

original_obs_space = Dict({
    'alive': Box(low=9, high=13, shape=(4,)),
    'board': Box(low=0, high=13, shape=(11, 11)),
    'bomb_blast_strength': Box(low=0, high=13, shape=(11, 11)),
    'bomb_life': Box(low=0, high=13, shape=(11, 11)),
    'position': Box(low=0, high=10, shape=(2,)),
    'ammo': Discrete(11),
    'blast_strength': Discrete(11),
    'can_kick': Discrete(2),
    'teammate': Discrete(14),
    'enemies': Box(low=9, high=13, shape=(3,)),
    'message': Box(low=0, high=7, shape=(2,))
})


def softmax(x, mask=None):
    if mask is not None:
        x = x + np.log(mask + 1e-45)
    return np.exp(x) / sum(np.exp(x))


def copy_weight(trainer, src, dest):
    P0key_P1val = {}
    for (k, v), (k2, v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                trainer.get_policy(src).get_weights().items()):
        P0key_P1val[k] = v2

    trainer.set_weights({dest: P0key_P1val,
                         src: trainer.get_policy(src).get_weights()})

    trainer.workers.local_worker().filters[dest] = trainer.workers.local_worker().filters[src].copy()

    for (k, v), (k2, v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                trainer.get_policy(src).get_weights().items()):
        assert (v == v2).all()


def featurize(obs):
    board = obs["board"]
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
            if obs["can_kick"] and board[i, j] == constants.Item.Bomb.value:
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
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in obs["alive"]:
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
    board = obs["board"]
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
            if obs["can_kick"] and board[i, j] == constants.Item.Bomb.value:
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
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in obs["alive"]:
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
    board = np.asarray(obs["board"], dtype=np.int)
    one_hot_board = nn.functional.one_hot(torch.tensor(board), 14).transpose(0, 2).transpose(1, 2).numpy()

    position = np.zeros(board.shape)
    position[obs["position"]] = 1
    teammate = board == obs["teammate"].value
    teammate_alive = np.full(board.shape, fill_value=1 if obs["teammate"].value in obs["alive"] else 0)

    alive_enemies = 0
    enemies = np.zeros(board.shape)
    for enemy in obs["enemies"]:
        enemies[(board == enemy.value)] = 1
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in obs["alive"]:
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
    board = np.asarray(obs["board"], dtype=np.float)
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
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in obs["alive"]:
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
    board = np.asarray(obs["board"], dtype=np.int)
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
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in obs["alive"]:
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
    agent_id = obs["board"][obs["position"]]

    preprocessed_obs = obs.copy()
    if centering:
        preprocessed_obs = center(obs, size=11)

    board = np.asarray(preprocessed_obs["board"], dtype=np.int)

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
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in preprocessed_obs["alive"]:
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


def featurize_v5(obs, centering=False, input_size=9):
    agent_id = obs["board"][obs["position"]]

    preprocessed_obs = obs.copy()
    if centering:
        preprocessed_obs = center(obs, input_size)

    board = np.asarray(preprocessed_obs["board"], dtype=np.int)

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
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in preprocessed_obs["alive"]:
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


def featurize_v6(obs, centering=False, input_size=9):
    agent_id = obs["board"][obs["position"]]

    preprocessed_obs = copy.deepcopy(obs)
    if centering:
        preprocessed_obs = center(obs, input_size)

    board = np.asarray(preprocessed_obs["board"], dtype=np.int)

    one_hot_board = nn.functional.one_hot(torch.tensor(board), 14).transpose(0, 2).transpose(1, 2).numpy()

    one_hot_board[0] = one_hot_board[0] + one_hot_board[6] + one_hot_board[7] + one_hot_board[8]
    if obs["can_kick"]:
        one_hot_board[0] += one_hot_board[3]

    if agent_id % 2 == 1:
        one_hot_board[[10, 11]] = one_hot_board[[11, 10]]
        one_hot_board[[12, 13]] = one_hot_board[[13, 12]]

    if agent_id == 12 or agent_id == 13:
        one_hot_board[[10, 12]] = one_hot_board[[12, 10]]

    one_hot_board[13] = one_hot_board[11] + one_hot_board[13]

    one_hot_board = np.delete(one_hot_board, [9, 11], 0)

    one_hot_bomb_moving_direction = \
        nn.functional.one_hot(torch.tensor(np.asarray(obs["bomb_moving_direction"], dtype=np.int)),
                              num_classes=5).transpose(0, 2).transpose(1, 2).numpy()

    one_hot_bomb_moving_direction = np.delete(one_hot_bomb_moving_direction, [0], 0)

    teammate_alive = np.full(board.shape,
                             fill_value=1 if preprocessed_obs["teammate"].value in preprocessed_obs["alive"] else 0)

    alive_enemies = 0
    for enemy in preprocessed_obs["enemies"]:
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in preprocessed_obs["alive"]:
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

    features = np.concatenate([one_hot_board, one_hot_bomb_moving_direction, features], 0)

    return features


def featurize_v7(obs, centering=False, input_size=9):
    agent_id = obs["board"][obs["position"]]

    preprocessed_obs = copy.deepcopy(obs)
    if centering:
        preprocessed_obs = center(obs, input_size)

    one_hot_message_1 = np.zeros((8, input_size, input_size))
    one_hot_message_1[preprocessed_obs["message"][0]] = np.ones((input_size, input_size))
    one_hot_message_2 = np.zeros((8, input_size, input_size))
    one_hot_message_2[preprocessed_obs["message"][1]] = np.ones((input_size, input_size))

    board = np.asarray(preprocessed_obs["board"], dtype=np.int)

    one_hot_board = nn.functional.one_hot(torch.tensor(board), 14).transpose(0, 2).transpose(1, 2).numpy()

    one_hot_board[0] = one_hot_board[0] + one_hot_board[6] + one_hot_board[7] + one_hot_board[8]
    if preprocessed_obs["can_kick"]:
        one_hot_board[0] += one_hot_board[3]

    if agent_id % 2 == 1:
        one_hot_board[[10, 11]] = one_hot_board[[11, 10]]
        one_hot_board[[12, 13]] = one_hot_board[[13, 12]]

    if agent_id == 12 or agent_id == 13:
        one_hot_board[[10, 12]] = one_hot_board[[12, 10]]

    one_hot_board[13] = one_hot_board[11] + one_hot_board[13]

    one_hot_board = np.delete(one_hot_board, [9, 11], 0)

    one_hot_bomb_moving_direction = \
        nn.functional.one_hot(torch.tensor(np.asarray(preprocessed_obs["bomb_moving_direction"], dtype=np.int)),
                              num_classes=5).transpose(0, 2).transpose(1, 2).numpy()

    one_hot_bomb_moving_direction = np.delete(one_hot_bomb_moving_direction, [0], 0)

    teammate_alive = np.full(board.shape,
                             fill_value=1 if preprocessed_obs["teammate"].value in preprocessed_obs["alive"] else 0)

    alive_enemies = 0
    for enemy in preprocessed_obs["enemies"]:
        if enemy.value != constants.Item.AgentDummy.value and enemy.value in preprocessed_obs["alive"]:
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

    features = np.concatenate(
        [one_hot_board, one_hot_bomb_moving_direction, features, one_hot_message_1, one_hot_message_2], 0)

    return features


def featurize_non_learning_agent(obs):
    temp_obs = copy.deepcopy(obs)
    featurized_obs = {
        'alive': temp_obs['alive'],
        'board': temp_obs['board'],
        'bomb_blast_strength': temp_obs['bomb_blast_strength'],
        'bomb_life': temp_obs['bomb_life'],
        'position': np.array(temp_obs['position']),
        'ammo': temp_obs['ammo'],
        'blast_strength': temp_obs['blast_strength'],
        'can_kick': temp_obs['can_kick'],
        'teammate': temp_obs['teammate'].value,
        'enemies': [enemy.value for enemy in temp_obs['enemies']],
        'message': np.array(temp_obs['message'])
    }

    while len(featurized_obs['alive']) != 4:
        featurized_obs['alive'].append(constants.Item.AgentDummy.value)
    return featurized_obs


def limit_gamma_explore(config):
    config["gamma"] = min(config["gamma"], 0.999)
    return config


def policy_mapping(agent_id):
    # agent_id pattern training/opponent_policy-id_agent-num
    # print("Calling to policy mapping {}".format(agent_id))
    name, id, _ = agent_id.split("_")

    return "{}_{}".format(name, id)


def center(obs, size=9):
    centered_obs = copy.deepcopy(obs)

    centered_obs["board"] = np.ones((size, size), dtype=np.float32)
    centered_obs["bomb_blast_strength"] = np.zeros((size, size), dtype=np.float32)
    centered_obs["bomb_life"] = np.zeros((size, size), dtype=np.float32)
    centered_obs["bomb_moving_direction"] = np.zeros((size, size), dtype=np.float32)

    x, y = obs["position"]

    u = size // 2
    v = 11 + u

    centered_obs["board"][max(0, u - x):min(size, v - x), max(0, u - y):min(size, v - y)] \
        = obs["board"][max(0, x - u):min(11, x + u + 1), max(0, y - u):min(11, y + u + 1)]

    centered_obs["bomb_blast_strength"][max(0, u - x):min(size, v - x), max(0, u - y):min(size, v - y)] \
        = obs["bomb_blast_strength"][max(0, x - u):min(11, x + u + 1), max(0, y - u):min(11, y + u + 1)]

    centered_obs["bomb_life"][max(0, u - x):min(size, v - x), max(0, u - y):min(size, v - y)] \
        = obs["bomb_life"][max(0, x - u):min(11, x + u + 1), max(0, y - u):min(11, y + u + 1)]

    centered_obs["bomb_moving_direction"][max(0, u - x):min(size, v - x), max(0, u - y):min(size, v - y)] \
        = obs["bomb_moving_direction"][max(0, x - u):min(11, x + u + 1), max(0, y - u):min(11, y + u + 1)]

    return centered_obs


def make_env(num_agents, env_id, game_state_file):
    agent_list = []
    for i in range(num_agents):
        agent_list.append(StaticAgent())

    return pommerman.make(env_id, agent_list, game_state_file)

