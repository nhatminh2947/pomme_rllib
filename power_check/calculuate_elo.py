import numpy as np
from numpy import genfromtxt

from eloranking import EloRatingSystem


class WLTRate:
    def __init__(self, win, loss, tie):
        self._win = win,
        self._loss = loss,
        self._tie = tie


ers = EloRatingSystem(k=0.1)

data = genfromtxt("./prior_power.txt", delimiter='\t', dtype=str)
player_names = data[0]

for player in player_names:
    ers.add_player(player)

ers.list_elo_rating()

prior = np.zeros((3, 9, 9))

data = np.delete(data, 0, 0)
for i, player_a in enumerate(player_names):
    for j, player_b in enumerate(player_names):
        win, loss, tie = data[i][j].split("/")

        prior[0][i][j] = float(win) / 100
        prior[1][i][j] = float(loss) / 100
        prior[2][i][j] = float(tie) / 100

print(prior)
# fix worst agent
for round in range(10000):
    for i, player_a in enumerate(player_names):
        for j, player_b in enumerate(player_names):
            r = np.random.random()
            expected_score = ers.expected_score(player_a, player_b)

            if r < prior[0][i][j]:
                ers.update_rating(player_a, expected_score, 1)
                ers.update_rating(player_b, expected_score, 0)
            elif prior[0][i][j] <= r < prior[1][i][j]:
                ers.update_rating(player_a, expected_score, 0)
                ers.update_rating(player_b, expected_score, 1)
            else:
                ers.update_rating(player_a, expected_score, 0.5)
                ers.update_rating(player_b, expected_score, 0.5)

ers.list_elo_rating()