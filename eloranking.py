import numpy as np


class EloRatingSystem:
    def __init__(self):
        self.population = 0
        self.rating = {}
        self.k = 32

    def add_player(self, name, elo=1000):
        self.rating[name] = elo
        self.population += 1

    def expected_score(self, player_a, player_b):
        return 1 / (1 + 10 ** ((self.rating[player_b] - self.rating[player_a]) / 400))

    def update_rating(self, player_a, expected_score, actual_score):
        self.rating[player_a] = self.rating[player_a] + self.k * (actual_score - expected_score)

    def list_elo_rating(self):
        for i in self.rating:
            print('Player: {} rating: {}'.format(i, self.rating[i]))


if __name__ == '__main__':
    n_player = 5
    ers = EloRatingSystem()
    np.random.seed(1)
    for i in range(n_player):
        ers.add_player('player_{}'.format(i), elo=np.random.randint(1000, 1500))

    result = np.random.randint(0, 2, (5, 5))
    result = (result + result.T) / 2
    print(result)
    ers.list_elo_rating()

    for i in range(n_player):
        name = 'player_{}'.format(i)
        sum_expected_score = 0
        sum_actual_score = np.sum(result[i]) - result[i][i]
        print(sum_actual_score)
        for j in range(n_player):
            if i == j:
                continue
            sum_expected_score += ers.expected_score(name, 'player_{}'.format(j))

        print(sum_expected_score)
        ers.update_rating(name, sum_expected_score, sum_actual_score)

    ers.list_elo_rating()
