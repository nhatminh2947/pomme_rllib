import numpy as np
from pommerman import constants

NUM_FEATURES = 19


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
    for i in range(11):
        for j in range(11):
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

    enemies = np.zeros(board.shape)
    for enemy in obs["enemies"]:
        enemies[(board == enemy.value)] = 1
    features.append(enemies)

    for i in range(1, 5):
        features.append(obs["bomb_moving_direction"] == i)

    # normal features
    for feature, max_value in zip(["bomb_life", "bomb_blast_strength", "flame_life"],
                                  [9, 10, 3]):
        features.append(obs[feature] / max_value)

    features.append(np.full(board.shape, fill_value=obs["ammo"] / 10))
    features.append(np.full(board.shape, fill_value=obs["blast_strength"] / 10))
    features.append(np.full(board.shape, fill_value=(1 if obs["can_kick"] else 0)))

    features = np.stack(features, 0)
    features = np.asarray(features, dtype=np.float)

    return features


def check_nan(array):
    tmp = np.sum(array)
    if np.isnan(tmp) or np.isinf(tmp):
        print('NaN or Inf found in input tensor.')
        print(tmp)
    return array
