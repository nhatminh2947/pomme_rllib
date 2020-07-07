import numpy as np
from pommerman import constants


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
