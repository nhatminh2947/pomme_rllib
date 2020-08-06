# from queue import Queue, PriorityQueue
import queue

import numpy as np


def get_rigid_wall(ob):
    board = np.zeros([11, 11], dtype='float32')
    board[np.where(ob['board'] == 1)] = 1
    return board


def get_wooden_wall(ob):
    board = np.zeros([11, 11], dtype='float32')
    board[np.where(ob['board'] == 2)] = 1
    return board


def get_all_bomb(ob):
    board = np.zeros([11, 11], dtype='float32')
    board[np.where(ob['board'] == 3)] = 1
    return board


def get_flames(ob):
    board = np.zeros([11, 11], dtype='float32')
    board[np.where(ob['board'] == 4)] = 1
    return board


def get_fog(ob):
    board = np.zeros([11, 11], dtype='float32')
    board[np.where(ob['board'] != 5)] = 1
    board[np.where(ob['board'] == 5)] = 0
    return board


def get_my_pos(ob, id=13):
    board = np.zeros([11, 11], dtype='float32')
    board[np.where(ob['board'] == id)] = 1
    return board


def get_enemey_pos(ob):
    board = np.zeros([11, 11], dtype='float32')
    for i in ob['enemies']:
        if i.value > 9:
            board[np.where(ob['board'] == i.value)] = 1
    return board


def get_ally_pos(ob):
    board = np.zeros([11, 11], dtype='float32')
    # print(ob['teammate'].value)
    board[np.where(ob['board'] == ob['teammate'].value)] = 1
    # if np.sum(board) > 0:
    #    print(board)
    return board


def get_power_up(ob):
    board = np.zeros([11, 11], dtype='float32')
    board[ob['board'] == 6] = 1
    board[ob['board'] == 7] = 1
    if ob['can_kick']:
        return board
    else:
        board[ob['board'] == 8] = 1
        return board


def get_ammo(ob):
    board = np.ones([11, 11], dtype='float32')
    return board * ob['ammo'] / 5


def get_can_kick(ob):
    board = np.ones([11, 11], dtype='float32')
    return board * ob['can_kick']


def get_strength(ob):
    board = np.ones([11, 11], dtype='float32')
    return board * ob['blast_strength'] / 5


def get_global_connectivity(ob):
    '''
    return the connectivity on current board
    '''
    board = np.zeros([11, 11], dtype='float32')
    x, y = ob['position']
    if ob['bomb_blast_strength'][x, y] == 0:
        board[x, y] = 1
    else:
        board[x, y] = 0
    board[ob['board'] == 6] = 1
    board[ob['board'] == 7] = 1
    board[ob['board'] == 8] = 1
    board[ob['board'] == 0] = 1
    if ob['can_kick']:
        board[ob['board'] == 3] = 1
    return board


def avoid_death_mask(ob):
    mask = get_global_connectivity(ob)

    x, y = ob['position']
    # if ob['bomb_life'][x, y] > 0:
    #    mask[x, y] = 0
    bomb = get_bomb_range(ob)
    mask[bomb >= 0.7] = 0
    ds = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for dx, dy in ds:
        x, y = ob['position']
        _x = x + dx
        _y = y + dy
        try:
            if np.sum(bfs(mask, (_x, _y))) <= 1 and ob['bomb_blast_strength'][x, y] >= 2:
                mask[_x, _y] = 0
        except IndexError:
            continue
    mask = bfs(mask, (x, y))
    return mask.reshape(121)


def get_local_map(ob):
    '''
    avoid any position that might cause immediate death
    '''
    mask = get_global_connectivity(ob)
    x, y = ob['position']
    '''
    bomb = get_bomb_range(ob)
    mask[bomb >= 0.7] = 0
    ds = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for dx, dy in ds:
        x, y = ob['position']
        _x = x + dx
        _y = y + dy
        try:
            if np.sum(bfs(mask, (_x, _y))) == 1 and ob['bomb_blast_strength'][x, y] == 2:
                mask[_x, _y] = 0
        except IndexError:
            continue
    '''
    mask = bfs(mask, (x, y))
    return mask  # np.array([1,1, 1, 1, 1])


def bfs(con, pos):
    '''
    return connected areas given global connectivity and pos
    '''

    def _adj(con, pos):
        x, y = pos
        adj_pos = []
        if x > 0 and con[x - 1, y] == 1:
            adj_pos.append((x - 1, y))
        if x < 10 and con[x + 1, y] == 1:
            adj_pos.append((x + 1, y))
        if y > 0 and con[x, y - 1] == 1:
            adj_pos.append((x, y - 1))
        if y < 10 and con[x, y + 1] == 1:
            adj_pos.append((x, y + 1))
        return adj_pos

    board = np.zeros([11, 11], dtype='float32')
    visited = set()
    q = queue.Queue()
    q.put(pos)
    while not q.empty():
        pos = q.get()
        visited.add(pos)
        board[pos] = 1
        for pos in _adj(con, pos):
            if pos not in visited:
                q.put(pos)
    return board


def update_bomb_life(ob):
    bomb_life = copy.deepcopy(ob['bomb_life'])
    bomb = get_all_bomb(ob)
    for x, y in zip(np.where(bomb != 0)[0], np.where(bomb != 0)[1]):
        r = int(ob['bomb_blast_strength'][x, y])
        current_blast = get_current_bomb_range(ob, r, (x, y), False)
        current_life = int(bomb_life[x, y])
        for i, j in zip(np.where(current_blast * bomb == 1)[0], np.where(current_blast * bomb == 1)[1]):
            if x == i and y == j:
                continue
            bomb_life[i, j] = np.min([bomb_life[i, j], current_life])
    return bomb_life


def get_bomb_range(ob, pos=None, use_count=True):
    wooden_wall = get_wooden_wall(ob)
    rigid_wall = get_rigid_wall(ob)
    bomb_life = update_bomb_life(ob)
    if pos == None:
        bomb_mask = np.ones([11, 11], dtype='float32')
    else:
        bomb_mask = np.zeros([11, 11], dtype='float32')
        bomb_mask[pos] = 1
    if use_count:
        bomb = np.asarray((10 - bomb_life * bomb_mask), dtype='float32') / 10
        bomb[bomb == 1] = 0
    else:
        bomb = np.asarray(bomb_life * bomb_mask, dtype='float32')
        bomb[bomb > 0] = 1
    for x, y in zip(np.where(bomb != 0)[0], np.where(bomb != 0)[1]):
        v = bomb[x, y]
        r = int(ob['bomb_blast_strength'][x, y])
        for i in range(r):
            bomb[x - i, y] = v if bomb[x - i, y] != 0 else max(v, bomb[x - i, y])
            if x - i == 0 or wooden_wall[x - i, y]:
                break
            if rigid_wall[x - i - 1, y]:
                break
        for i in range(r):
            bomb[x + i, y] = v if bomb[x + i, y] != 0 else max(v, bomb[x + i, y])
            if x + i == 10 or wooden_wall[x + i, y]:
                break
            if rigid_wall[x + i + 1, y]:
                break
        for i in range(r):
            bomb[x, y - i] = v if bomb[x, y - i] != 0 else max(v, bomb[x, y - i])
            if y - i == 0 or wooden_wall[x, y - i]:
                break
            if rigid_wall[x, y - i - 1]:
                break
        for i in range(r):
            bomb[x, y + i] = v if bomb[x, y + i] != 0 else max(v, bomb[x, y + i])
            if y + i == 10 or wooden_wall[x, y + i]:
                break
            if rigid_wall[x, y + i + 1]:
                break
        if use_count:
            bomb[x, y] += 0.1
    return bomb


def get_current_bomb_range(ob, r, pos=None, use_count=False):
    wooden_wall = get_wooden_wall(ob)
    rigid_wall = get_rigid_wall(ob)
    if pos == None:
        bomb_mask = np.ones([11, 11], dtype='float32')
    else:
        bomb_mask = np.zeros([11, 11], dtype='float32')
        bomb_mask[pos] = 1
    if use_count:
        bomb = np.asarray((10 - ob['bomb_life'] * bomb_mask), dtype='float32') / 10
        bomb[bomb == 1] = 0
    else:
        bomb = np.asarray(bomb_mask, dtype='float32')
        bomb[bomb > 0] = 1
    for x, y in zip(np.where(bomb != 0)[0], np.where(bomb != 0)[1]):
        v = bomb[x, y]
        # r = int(ob['bomb_blast_strength'][x, y])
        for i in range(r):
            bomb[x - i, y] = v if bomb[x - i, y] != 0 else max(v, bomb[x - i, y])
            if x - i == 0 or wooden_wall[x - i, y]:
                break
            if rigid_wall[x - i - 1, y]:
                break
        for i in range(r):
            bomb[x + i, y] = v if bomb[x + i, y] != 0 else max(v, bomb[x + i, y])
            if x + i == 10 or wooden_wall[x + i, y]:
                break
            if rigid_wall[x + i + 1, y]:
                break
        for i in range(r):
            bomb[x, y - i] = v if bomb[x, y - i] != 0 else max(v, bomb[x, y - i])
            if y - i == 0 or wooden_wall[x, y - i]:
                break
            if rigid_wall[x, y - i - 1]:
                break
        for i in range(r):
            bomb[x, y + i] = v if bomb[x, y + i] != 0 else max(v, bomb[x, y + i])
            if y + i == 10 or wooden_wall[x, y + i]:
                break
            if rigid_wall[x, y + i + 1]:
                break
        if use_count:
            bomb[x, y] += 0.1
    return bomb


def get_action_mask(ob):
    x, y = ob['position']
    ds = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    mask = np.zeros(6)  # ; mask[-1] = int(ob['ammo']>0)# and is_valid_bomb_pos(ob))
    mask[0] = 1
    for i, d in enumerate(ds):
        dx, dy = d
        try:
            if ob['board'][x + dx, y + dy] in [0, 6, 7, 8] or (
                    ob['can_kick'] and ob['board'][x + dx, y + dy] in [0, 3, 6, 7, 8]):
                mask[i + 1] = 1
        except IndexError:
            continue
    return mask


def get_dest_pos(dest):
    x, y = dest
    board = np.zeros([11, 11], dtype='float32')
    board[x, y] = 1
    return board


def is_valid_bomb_pos(ob):
    m_dist = lambda pos1, pos2: abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    r = ob['blast_strength']
    x, y = ob['position']
    wooden_wall = get_wooden_wall(ob)
    enemies = get_enemey_pos(ob)
    min_range = get_current_bomb_range(ob, 2, ob['position'])
    bomb_range = get_current_bomb_range(ob, ob['blast_strength'], ob['position'])
    if np.sum(wooden_wall * min_range) > 0 or np.sum(enemies * bomb_range) > 0:
        return True
    for i in ob['enemies']:
        if i.value > 9:
            x_, y_ = np.where(ob['board'] == i.value)
            if np.shape(x_)[0]:
                if m_dist((x, y), (x_[0], y_[0])) <= 3:
                    return True
    #            if ob['can_kick'] and (abs(x-x_) < r or abs(y-y_) < r):
    #                return True
    return False


def _process_obs(ob, dest, agent_id=13):
    x, y = ob['position']
    pos = get_ally_pos(ob) - get_enemey_pos(ob)
    bomb = get_current_bomb_range(ob, ob['blast_strength'], ob['position'])
    if ob['ammo'] == 0:
        bomb = bomb * 0
        bomb[x, y] = 1
    else:
        bomb = bomb * 0.1
        bomb[x, y] = 1
    terrain = get_wooden_wall(ob) * 0.3 + get_rigid_wall(ob) - get_flames(ob)
    bomb_range = get_bomb_range(ob) * (-1)
    pills = get_power_up(ob)
    state = np.stack([
        bomb,
        pos,
        terrain,
        bomb_range,
        pills
    ], axis=-1)
    return state, [0, 0, 0]


def process_obs_v2(ob):
    x, y = ob['position']
    pos = get_ally_pos(ob) - get_enemey_pos(ob)
    bomb = get_current_bomb_range(ob, ob['blast_strength'], ob['position'])
    if ob['ammo'] == 0:
        bomb = bomb * 0
        bomb[x, y] = 1
    else:
        bomb = bomb * 0.1
        bomb[x, y] = 1
    terrain = get_wooden_wall(ob) * 0.3 + get_rigid_wall(ob) - get_flames(ob)
    bomb_range = get_bomb_range(ob) * (-1)
    pills = get_power_up(ob)
    state = np.stack([
        bomb,
        pos,
        terrain,
        bomb_range,
        pills
    ], axis=-1)
    return state


def process_obs(ob, agent_id=13):
    my_pos = get_my_pos(ob, agent_id)
    # dest_pos = get_dest_pos(dest)
    ally_pos = get_ally_pos(ob)
    enemy_pos = get_enemey_pos(ob)
    # global_map = get_global_connectivity(ob)
    local_map = get_local_map(ob)
    rigid_wall = get_rigid_wall(ob)
    wooden_wall = get_wooden_wall(ob)
    power_up = get_power_up(ob)
    flame = get_flames(ob)
    bomb = get_bomb_range(ob)
    current_bomb = get_current_bomb_range(ob, ob['blast_strength'], ob['position'])
    state = np.stack([
        my_pos,
        ally_pos,
        # dest_pos,
        enemy_pos,
        # global_map,
        # local_map,
        # current_bomb,
        power_up,
        rigid_wall,
        wooden_wall,
        flame,
        bomb,
    ], axis=-1)
    state[state == 5] = 0
    agent_alive = [int((agent_id - 10 + i) % 4 + 10 in ob['alive']) for i in range(1, 4)]
    bomb_condition = [0, 0, 0, 0]
    x, y = ob['position']
    ds = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for i, d in enumerate(ds):
        dx, dy = d
        try:
            ob['position'] = (x + dx, y + dy)
            if ob['board'][x + dx, y + dy] in [0, 6, 7, 8] and is_valid_bomb_pos(ob) and ob['ammo'] > 0:
                bomb_condition[i] = 1
        except IndexError:
            continue
    # print(bomb_condition)
    ob['position'] = (x, y)
    ability = [ob['ammo'] / 5, int(ob['can_kick']), ob['blast_strength'] / 5] + agent_alive + bomb_condition
    return state, ability


def get_move_reward(ob, ob_, agent_list, self_id, bombs):
    reward = 0
    if ob['can_kick']:
        if ob['board'][ob_['position']] in [6, 7]:
            reward += 0.2
    else:
        if ob['board'][ob_['position']] in [6, 7, 8]:
            reward += 0.2
    x, y = ob_['position']
    dx = ob_['position'][0] - ob['position'][0]
    dy = ob_['position'][1] - ob['position'][1]
    try:
        if ob['board'][ob_['position']] == 3 and ob_['board'][x + dx, y + dy] == 3:
            reward += 0.1
    except IndexError:
        pass
    return reward


def get_bomb_reward(ob, ob_, agent_list, self_id, bombs):
    reward = 0
    for bomb in bombs:
        if bomb.life > 2:
            continue
        if bomb.bomber.agent_id == self_id:
            bomb_map = get_current_bomb_range(ob, bomb.blast_strength, bomb.position)
            reward += (np.sum(bomb_map * get_wooden_wall(ob)) - np.sum(bomb_map * get_wooden_wall(ob_))) * 0.0
            for agent in agent_list:
                agent_id = agent.agent_id
                if agent_id + 10 in ob['alive'] and bomb_map[agent.position] > 0 and agent_id + 10 not in ob_['alive']:
                    # print(ob['alive'])
                    # print('DieDIeDie', agent.agent_id)
                    # print(ob_['alive'])
                    if agent_id == self_id:
                        #    print('I killed myself!')
                        reward -= 0.0
                    if agent_id == (self_id + 2) % 4:
                        #    print('I killed my ally!')
                        reward -= 0.2
                    if agent_id in [(self_id + 1) % 4, (self_id + 3) % 4]:
                        #    print('I killed enenmy!')
                        reward += 1
    # print(reward)
    return reward


def get_assist_reward(ob, ob_, agent_list, self_id, bombs):
    reward = 0
    for bomb in bombs:
        if bomb.life > 2:
            continue
        # if bomb.bomber.agent_id == (self_id+2)%4:
        my_x, my_y = agent_list[self_id].position
        for agent in agent_list:
            bomb_map = get_current_bomb_range(ob, bomb.blast_strength, bomb.position)
            agent_id = agent.agent_id
            if agent_id + 10 in ob['alive'] and bomb_map[agent.position] > 0 and agent_id + 10 not in ob_['alive']:
                if agent_id in [(self_id + 1) % 4, (self_id + 3) % 4]:
                    x, y = agent_list[agent_id].position
                    if abs(my_x - x) + abs(my_y - y) < 4:
                        #    print('I assisted in killing enenmy!')
                        reward += 0.2
            # if agent.agent_id == self_id:
    return reward


def get_kick_reward(ob, ob_, agent_list, self_id, bombs):
    reward = 0
    x, y = ob_['position']
    dx = ob_['position'][0] - ob['position'][0]
    dy = ob_['position'][1] - ob['position'][1]
    try:
        if ob['board'][ob_['position']] == 3 and ob_['board'][x + dx, y + dy] == 3:
            reward += 0.05
    except IndexError:
        pass
    return reward


'''
def get_move_reward(ob, ob_):
    reward = 0
    if ob['can_kick']:
        if ob['board'][ob_['position']] in [6, 7]:
            reward = 0.2
    else:
        if ob['board'][ob_['position']] in [6, 7, 8]:
            reward = 0.2
    return reward

def get_bomb_reward(ob, obs_):
    x, y = ob['position']
    enemy_reward = 0
    bomb = get_current_bomb_range(ob, ob['blast_strength'], (x,y))
    for o in obs_:
        if np.sum(bomb * get_enemey_pos(o)) > 0:
            enemy_reward += 0.03
            break
    enemy_reward += np.sum(bomb * get_enemey_pos(obs_[-1])) * 1
    #wood_reward = np.sum(bomb * get_wooden_wall(ob)) * 0.03
    ally_penalty = np.sum(bomb * get_ally_pos(ob)) * 0.2
    my_pos = np.zeros([11, 11])
    x, y = obs_[-1]['position']
    my_pos[x, y] = 1
    my_penalty = np.sum(bomb * my_pos) * 0.2
    reward = enemy_reward - ally_penalty - my_penalty
    #if reward == 0:
    #    return -0.01
    return reward
'''


def get_bomb_mask(ob):
    return [1, int(ob['ammo'] > 0 and is_valid_bomb_pos(ob))]


def get_move_mask(ob):
    return get_local_map(ob).reshape(121)


def _djikstra(board, my_position, depth=10):
    assert (depth is not None)

    def out_of_range(p_1, p_2):
        '''Determines if two points are out of rang of each other'''
        x_1, y_1 = p_1
        x_2, y_2 = p_2
        return abs(y_2 - y_1) + abs(x_2 - x_1) > depth

    dist = np.full(board.shape, np.inf)
    prev = {}
    Q = queue.Queue()

    my_x, my_y = my_position
    for r in range(max(0, my_x - depth), min(len(board), my_x + depth)):
        for c in range(max(0, my_y - depth), min(len(board), my_y + depth)):
            position = (r, c)
            if out_of_range(my_position, position):
                continue

            if board[position]:
                prev[position] = None

            if position == my_position:
                Q.put(position)
                dist[position] = 0
            else:
                dist[position] = np.inf

    while not Q.empty():
        position = Q.get()

        if board[position]:
            x, y = position
            val = dist[(x, y)] + 1
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (row + x, col + y)
                if new_position not in prev:
                    continue

                if val < dist[new_position]:
                    dist[new_position] = val
                    prev[new_position] = position
                    Q.put(new_position)
                elif (val == dist[new_position] and np.random.random() < .5):
                    dist[new_position] = val
                    prev[new_position] = position

    return dist, prev


def _get_direction(position, next_position):
    if position == next_position:
        # return constants.Action.Stop
        return 0
    x, y = position
    next_x, next_y = next_position
    if x == next_x:
        if y < next_y:
            # return constants.Action.Right
            return 4
        else:
            # return constants.Action.Left
            return 3
    elif y == next_y:
        if x < next_x:
            # return constants.Action.Down
            return 2
        else:
            # return constants.Action.Up
            return 1


def dest2act(board, my_position, dest_position):
    if my_position == dest_position:
        return _get_direction(my_position, dest_position)

    valid_board = _valid_position(board, my_position)
    dist, prev = _djikstra(valid_board,
                           my_position,
                           depth=100)
    next_position = dest_position
    try:
        while prev[next_position] != my_position:
            next_position = prev[next_position]
    except KeyError:
        return None
    return _get_direction(my_position, next_position)


def _valid_position(board, my_position):
    action_mask = (board == 0)
    for x in [6, 7, 8]:
        action_mask += (board == x)
    action_mask[my_position] = True
    return action_mask


import tensorflow as tf


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


import copy


def combine_batches(batches):
    batch = copy.deepcopy(batches[0])
    for b in batches[1:]:
        for i in range(len(b)):
            batch[i] = np.concatenate((batch[i], b[i]), axis=0)
    return batch


def sample(softmax, temperature=0.1):
    EPSILON = 10e-16  # to avoid taking the log of zero
    # print(preds)
    (np.array(softmax) + EPSILON).astype('float64')
    preds = np.log(softmax) / temperature
    # print(preds)
    exp_preds = np.exp(preds)
    # print(exp_preds)
    preds = exp_preds / np.sum(exp_preds)
    # print(preds)
    probas = np.random.multinomial(1, preds, 1)
    return probas[0]
