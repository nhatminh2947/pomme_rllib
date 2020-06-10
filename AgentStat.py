class AgentStat:
    def __init__(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False
        self.eliminate_enemies = 0
        self.eliminate_ally = 0
        self.dead_or_suicide = 0

    def reset(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False
        self.eliminate_enemies = 0
        self.eliminate_ally = 0
        self.dead_or_suicide = 0
