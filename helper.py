import numpy as np
import ray


@ray.remote(num_cpus=0.25, num_gpus=0)
class Helper:
    def __init__(self, n_histories, policy_names, burn_in, k=1.2, alpha=1.0, enemy="static"):
        self.policy_names = policy_names
        self._is_init = False
        self.n_histories = n_histories
        self.enemy = enemy
        self.alphas = {policy_name: alpha for policy_name in self.policy_names}
        self.num_steps = {policy_name: 0 for policy_name in self.policy_names}
        self.k = k
        self.burn_in = burn_in
        self.low = 1
        self.high = 2

    def update_bounding(self):
        self.high = min(self.high + 1, len(self.policy_names))
        self.low = max(self.low, self.high - self.n_histories)

    def update_num_steps(self, policy_name, num_steps):
        self.num_steps[policy_name] += num_steps
        return self.num_steps[policy_name]

    def update_alpha(self, policy_name, enemy_death_mean):
        self.alphas[policy_name] = 1 - np.tanh(self.k * enemy_death_mean)
        if self.num_steps[policy_name] >= self.burn_in:
            return 0.0
        return self.alphas[policy_name]

    def get_alpha(self, policy_name):
        return self.alphas[policy_name]

    def get_num_steps(self, policy_name):
        return self.num_steps[policy_name]

    def set_policy_names(self):
        # self.policy_names = np.random.permutation(self.policy_names)
        return

    def get_training_policies(self):
        return self.policy_names[0], self.policy_names[np.random.randint(self.low, self.high)]

    def is_init(self):
        return self._is_init

    def set_init_done(self):
        self._is_init = True
