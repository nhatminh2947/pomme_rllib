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
    def __init__(self, n_histories, alpha_coeff, burn_in, k=16):
        self.population = {}
        self.k = k
        self.n_histories = n_histories
        self.capacity = n_histories + 2
        self.alpha_coeff = alpha_coeff
        self.burn_in = burn_in

        self.add_policy("policy_0", False)
        self.add_policy("static_1", True)
        for i in range(n_histories):
            self.add_policy("policy_{}".format(i + 2), False)

    def update_alpha(self, policy_name, enemy_death_mean):
        self.population[policy_name].alpha = 1 - np.tanh(self.alpha_coeff * enemy_death_mean)
        if self.population[policy_name].num_steps >= self.burn_in:
            return 0.0
        return self.population[policy_name].alpha

    def get_alpha(self, policy_name):
        return self.population[policy_name].alpha

    def add_policy(self, policy_name, ready, elo=1000):
        self.population[policy_name] = Member(policy_name, elo, ready, 1.0)

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

        return "policy_0", enemy

    # def strong_enough(self):
    #     for i in range(max(1, self.population - self.n_histories), self.population):
    #         if self.expected_score("policy_0", self.policy_names[i]) < 0.6:
    #             return False
    #
    #     return True

    def update_population(self):
        policy_name = "policy_{}".format(np.random.randint(2, self.capacity))

        if self.expected_score("policy_0", policy_name) >= 0.6:
            self.population[policy_name].rating = self.population["policy_0"].rating
            self.population[policy_name].ready = True
            return policy_name

        return None


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
