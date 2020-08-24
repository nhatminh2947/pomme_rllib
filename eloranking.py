import numpy as np
import ray

import utils


class Member:
    def __init__(self, policy_name, rating, ready, alpha, num_steps=0):
        self.policy_name = policy_name
        self.rating = rating
        self.ready = ready
        self.alpha = alpha
        self.num_steps = num_steps


@ray.remote(num_cpus=0.1, num_gpus=0)
class EloRatingSystem:
    def __init__(self, policy_names, n_histories, alpha_coeff, burn_in, k=16):
        self.population = {}
        self.k = k
        self.n_histories = n_histories
        self.capacity = n_histories + 2
        self.alpha_coeff = alpha_coeff
        self.burn_in = burn_in

        for policy_name in policy_names:
            self.add_policy(policy_name, False, 0, 1000)

        self.population["policy_0"].alpha = 0.4
        self.population["policy_0"].rating = 1283
        self.population["static_1"].ready = True
        self.population["static_1"].rating = 822
        self.population["smartrandomnobomb_2"].ready = True
        self.population["smartrandomnobomb_2"].rating = 1046
        self.population["smartrandom_3"].ready = True
        self.population["smartrandom_3"].rating = 1074
        # self.population["cautious_4"].ready = True
        # self.population["cautious_4"].rating = 1089
        # self.population["neoteric_5"].ready = True
        # self.population["neoteric_5"].rating = 1295
        # self.population["policy_6"].ready = True
        # self.population["policy_6"].rating = 1295

    def update_alpha(self, policy_name, enemy_death_mean):
        self.population[policy_name].alpha = 1 - np.tanh(self.alpha_coeff * enemy_death_mean)
        if self.population[policy_name].num_steps >= self.burn_in:
            return 0.0
        return self.population[policy_name].alpha

    def get_alpha(self, policy_name):
        return self.population[policy_name].alpha

    def add_policy(self, policy_name, ready, alpha, elo=1000):
        self.population[policy_name] = Member(policy_name, elo, ready, alpha)

    def expected_score(self, policy_a, policy_b):
        return 1 / (1 + 10 ** ((self.population[policy_b].rating - self.population[policy_a].rating) / 400))

    def update_rating(self, policy_a, expected_score, actual_score):
        self.population[policy_a].rating = self.population[policy_a].rating + self.k * (actual_score - expected_score)
        return self.population[policy_a].rating

    def update_num_steps(self, policy_name, num_steps):
        self.population[policy_name].num_steps += num_steps
        return self.population[policy_name].num_steps

    def get_training_policies(self):
        mask = np.asarray([self.population[policy].ready for policy in self.population])
        prob = utils.softmax([np.log(self.population[policy].rating) for policy in self.population], mask)
        enemy = np.random.choice(list(self.population.keys()), size=1, p=prob)[0]
        while not self.population[enemy].ready:
            enemy = np.random.choice(list(self.population.keys()), size=1, p=prob)[0]

        return "policy_0", enemy

    def update_population(self):
        min_rating = 10000
        weakest_policy = None

        is_strongest = True

        for i, policy_name in enumerate(self.population):
            if policy_name == "cautious_4" or policy_name == "neoteric_5":
                continue
            if policy_name != "policy_0" and self.population[policy_name].ready:
                if self.expected_score("policy_0", policy_name) < 0.6:
                    is_strongest = False

        if is_strongest:
            for i, policy_name in enumerate(self.population):
                if policy_name == "cautious_4" or policy_name == "neoteric_5":
                    continue

                if policy_name != "policy_0" and not self.population[policy_name].ready:
                    self.population[policy_name].ready = True
                    self.population[policy_name].rating = self.population["policy_0"].rating
                    if "policy" in policy_name:
                        return policy_name
                    return None

            for i, policy_name in enumerate(self.population):
                if policy_name == "cautious_4" or policy_name == "neoteric_5":
                    continue

                if policy_name != "policy_0" \
                        and "policy" in policy_name \
                        and min_rating > self.population[policy_name].rating:
                    weakest_policy = policy_name
                    min_rating = self.population[policy_name].rating

            self.population[weakest_policy].rating = self.population["policy_0"].rating
        return weakest_policy


if __name__ == '__main__':
    n_player = 9
    ers = EloRatingSystem()
    np.random.seed(1)
    for i in range(n_player):
        ers.add_policy('player_{}'.format(i), elo=np.random.randint(1000, 1500))

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
