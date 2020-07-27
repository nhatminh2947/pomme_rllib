import numpy as np
import ray


@ray.remote(num_cpus=0.25, num_gpus=0)
class Helper:
    def __init__(self, population_size, k=1.2, alpha=1.0, enemy="static"):
        self.population_size = population_size
        self.policy_names = ["policy_{}".format(i) for i in range(self.population_size)]
        self._is_init = False
        self.enemy = enemy
        self.alphas = {policy_name: alpha for policy_name in self.policy_names}
        self.num_steps = {policy_name: 0 for policy_name in self.policy_names}
        self.k = k

    def run_pbt(self, trainer):
        self.pbt.run(trainer, self.ers)

    def update_num_steps(self, policy_name, num_steps):
        self.num_steps[policy_name] += num_steps

    def update_rating(self, policy_name, expected_score, actual_score):
        return self.ers.update_rating(policy_name, expected_score, actual_score)

    def get_expected_score(self, player_a, player_b):
        return self.ers.expected_score(player_a, player_b)

    def update_alpha(self, policy_name, win_rate):
        self.alphas[policy_name] = 1 - np.tanh(self.k * win_rate)
        return self.alphas[policy_name]

    def get_alpha(self, policy_name):
        return self.alphas[policy_name]

    def get_num_steps(self, policy_name):
        return self.num_steps[policy_name]

    def set_agent_names(self):
        self.policy_names = np.random.permutation(self.policy_names)

    def get_training_policies(self):
        return self.policy_names[0:2]

    def is_init(self):
        return self._is_init

    def set_init_done(self):
        self._is_init = True
