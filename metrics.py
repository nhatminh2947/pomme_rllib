from enum import Enum


class Metrics(Enum):
    EliminateEnemies = 0
    EliminateAlly = 1
    DeadOrSuicide = 2
    Win = 3
    Tie = 4
    Loss = 5
