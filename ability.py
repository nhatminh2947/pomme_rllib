class Ability:
    def __init__(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False

    def reset(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False