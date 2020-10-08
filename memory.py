import copy

from pommerman import constants


class Memory:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.obs = None

    def init_memory(self, obs):
        self.obs = copy.deepcopy(obs)

        if obs['board'].shape[0] == 11:
            self.obs['board'][1, 1] = 10
            self.obs['board'][9, 1] = 11
            self.obs['board'][9, 9] = 12
            self.obs['board'][1, 9] = 13

    # Note: add memory of enemy position of some previous steps
    def update_memory(self, obs):
        temp_obs = copy.deepcopy(obs)
        self.obs['alive'] = temp_obs['alive']
        self.obs['bomb_moving_direction'] = temp_obs['bomb_moving_direction']
        self.obs['position'] = temp_obs['position']
        self.obs['blast_strength'] = temp_obs['blast_strength']
        self.obs['can_kick'] = temp_obs['can_kick']
        self.obs['ammo'] = temp_obs['ammo']

        for i in range(11):
            for j in range(11):
                if self.obs['bomb_life'][i][j] == 0:
                    self.obs['bomb_blast_strength'][i][j] = 0
                    self.obs['bomb_moving_direction'][i][j] = 0
                else:
                    self.obs['bomb_life'][i, j] -= 1

                if self.obs['flame_life'][i, j] != 0:
                    self.obs['flame_life'][i, j] -= 1

        for enemy in temp_obs['enemies']:
            if temp_obs['board'][temp_obs['board'] == enemy.value].any() or enemy.value not in temp_obs['alive']:
                self.obs['board'][self.obs['board'] == enemy.value] = constants.Item.Passage.value

        self.obs['board'][temp_obs['board'] != constants.Item.Fog.value] \
            = temp_obs['board'][temp_obs['board'] != constants.Item.Fog.value]
        self.obs['bomb_life'][temp_obs['board'] != constants.Item.Fog.value] \
            = temp_obs['bomb_life'][temp_obs['board'] != constants.Item.Fog.value]
        self.obs['bomb_blast_strength'][temp_obs['board'] != constants.Item.Fog.value] \
            = temp_obs['bomb_blast_strength'][temp_obs['board'] != constants.Item.Fog.value]
        self.obs['flame_life'][temp_obs['board'] != constants.Item.Fog.value] \
            = temp_obs['flame_life'][temp_obs['board'] != constants.Item.Fog.value]
        self.obs['bomb_moving_direction'][temp_obs['board'] != constants.Item.Fog.value] \
            = temp_obs['bomb_moving_direction'][temp_obs['board'] != constants.Item.Fog.value]
