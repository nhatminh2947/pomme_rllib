import numpy as np
from numpy import genfromtxt

from eloranking import EloRatingSystem

ers = EloRatingSystem(k=1)

data = genfromtxt("./prior_power.txt", delimiter='\t', dtype=str)
player_names = data[0]

for player in player_names:
    ers.add_player(player, elo=1000)

ers.list_elo_rating()

prior = np.zeros((3, 10, 10))

data = np.delete(data, 0, 0)
for i, player_a in enumerate(player_names):
    for j, player_b in enumerate(player_names):
        win, loss, tie = data[i][j].split("/")

        prior[0][i][j] = float(win) / 100
        prior[1][i][j] = float(loss) / 100
        prior[2][i][j] = float(tie) / 100

print(prior)
# fix worst agent
for round in range(1000):
    for i, player_a in enumerate(player_names):
        for j, player_b in enumerate(player_names):
            if i == j:
                continue

            r = np.random.random()

            if r < prior[0][i][j]:
                expected_score = ers.expected_score(player_a, player_b)
                ers.update_rating(player_a, expected_score, 1)
                expected_score = ers.expected_score(player_b, player_a)
                ers.update_rating(player_b, expected_score, 0)
            elif prior[0][i][j] <= r < prior[0][i][j] + prior[1][i][j]:
                expected_score = ers.expected_score(player_a, player_b)
                ers.update_rating(player_a, expected_score, 0)
                expected_score = ers.expected_score(player_b, player_a)
                ers.update_rating(player_b, expected_score, 1)
            else:
                expected_score = ers.expected_score(player_a, player_b)
                ers.update_rating(player_a, expected_score, 0.5)
                expected_score = ers.expected_score(player_b, player_a)
                ers.update_rating(player_b, expected_score, 0.5)

ers.list_elo_rating()
