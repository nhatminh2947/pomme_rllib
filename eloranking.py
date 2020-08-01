import numpy as np
import ray


@ray.remote(num_cpus=0.1, num_gpus=0)
class EloRatingSystem:
    def __init__(self, n_histories, k=16):
        self.population = 0
        self.rating = {}
        self.policy_names = []
        self.k = k
        self.n_histories = n_histories
        self.capacity = n_histories + 2

        self.add_player("policy_0")
        self.add_player("static_1")
        # self.add_player("random_2")

    def add_player(self, player, elo=1000):
        self.rating[player] = elo
        self.policy_names.append(player)
        self.population += 1

    def expected_score(self, player_a, player_b):
        return 1 / (1 + 10 ** ((self.rating[player_b] - self.rating[player_a]) / 400))

    def update_rating(self, player_a, expected_score, actual_score):
        self.rating[player_a] = self.rating[player_a] + self.k * (actual_score - expected_score)
        return self.rating[player_a]

    def list_elo_rating(self):
        for i in self.rating:
            print('Player: {} - rating: {}'.format(i, self.rating[i]))

    def strong_enough(self):
        for i in range(max(1, self.population - self.n_histories), self.population):
            if self.expected_score("policy_0", self.policy_names[i]) < 0.6:
                return False

        return True

    def make_history(self):
        min_rating = 10000
        weakest_player = None

        if len(self.rating) == self.capacity:
            for i in range(2, self.capacity):
                if self.rating["policy_{}".format(i)] < min_rating:
                    weakest_player = "policy_{}".format(i)
                    min_rating = self.rating[weakest_player]

        if weakest_player is None:
            self.add_player("policy_{}".format(self.population), self.rating["policy_0"])
            return self.policy_names[-1]

        self.rating[weakest_player] = self.rating["policy_0"]
        return weakest_player


if __name__ == '__main__':
    n_player = 9
    ers = EloRatingSystem()
    np.random.seed(1)
    for i in range(n_player):
        ers.add_player('player_{}'.format(i), elo=np.random.randint(1000, 1500))

    result = [
        [0, 35, 35, 53, 75, 28, 32, 73, 8],  # hakojaki
        [26, 0, 54, 87, 85, 21, 53, 87, 35],  # eisenach
        # [3, 10, 0, 24, 10], # dypm
        [],  # navocado
        [],  # skynet
        [],  # gorogm
        [],  # thing1
        [],  # inspir
        [],  # neoteric
    ]
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
