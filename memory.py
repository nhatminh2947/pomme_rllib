from pommerman import constants


class Memory:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.obs = None

    def init_memory(self, obs):
        self.obs = obs.copy()

        if obs['board'].shape[0] == 11:
            self.obs['board'][1, 1] = 10
            self.obs['board'][9, 1] = 11
            self.obs['board'][9, 9] = 12
            self.obs['board'][1, 9] = 13

    # Note: add memory of enemy position of some previous steps
    def update_memory(self, obs):
        self.obs['alive'] = obs['alive'].copy()
        self.obs['bomb_moving_direction'] = obs['bomb_moving_direction'].copy()
        self.obs['position'] = obs['position']
        self.obs['blast_strength'] = obs['blast_strength']
        self.obs['can_kick'] = obs['can_kick']
        self.obs['ammo'] = obs['ammo']

        for i in range(11):
            for j in range(11):
                if self.obs['bomb_life'][i][j] == 0:
                    self.obs['bomb_blast_strength'][i][j] = 0
                    self.obs['bomb_moving_direction'][i][j] = 0
                else:
                    self.obs['bomb_life'][i, j] -= 1

                if self.obs['flame_life'][i, j] != 0:
                    self.obs['flame_life'][i, j] -= 1

        for enemy in obs['enemies']:
            if obs['board'][obs['board'] == enemy.value].any() or enemy.value not in obs['alive']:
                self.obs['board'][self.obs['board'] == enemy.value] = constants.Item.Passage.value

        self.obs['board'][obs['board'] != constants.Item.Fog.value] = obs['board'][
            obs['board'] != constants.Item.Fog.value].copy()
        self.obs['bomb_life'][obs['board'] != constants.Item.Fog.value] = obs['bomb_life'][
            obs['board'] != constants.Item.Fog.value].copy()
        self.obs['bomb_blast_strength'][obs['board'] != constants.Item.Fog.value] = obs['bomb_blast_strength'][
            obs['board'] != constants.Item.Fog.value].copy()
        self.obs['flame_life'][obs['board'] != constants.Item.Fog.value] = obs['flame_life'][
            obs['board'] != constants.Item.Fog.value].copy()
        self.obs['bomb_moving_direction'][obs['board'] != constants.Item.Fog.value] = obs['bomb_moving_direction'][
            obs['board'] != constants.Item.Fog.value].copy()