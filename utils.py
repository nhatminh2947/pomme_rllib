import numpy as np
from pommerman import constants


NUM_FEATURES = 16


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
    board_items = [constants.Item.Passage,
                   constants.Item.Rigid,
                   constants.Item.Wood,
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

    features = np.stack(features, 0)
    features = np.asarray(features, dtype=np.float)
    check_nan(features)
    return features


def check_nan(array):
    tmp = np.sum(array)
    if np.isnan(tmp) or np.isinf(tmp):
        print('NaN or Inf found in input tensor.')
        print(tmp)
    return array
