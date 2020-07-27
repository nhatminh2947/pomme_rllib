import numpy as np
import ray


@ray.remote(num_cpus=0.25, num_gpus=0)
class Helper:
    def __init__(self, population_size, burn_in, exploration_steps, k=1.2, alpha=1.0, enemy="static"):
        self.population_size = population_size
        self.policy_names = ["policy_{}".format(i) for i in range(self.population_size)]
        self._is_init = False
        self.enemy = enemy
        self.alphas = {policy_name: alpha for policy_name in self.policy_names}
        self.num_steps = {policy_name: 0 for policy_name in self.policy_names}
        self.k = k
        self.burn_in = burn_in
        self.exploration_steps = exploration_steps
        self.updatable = False

    def is_updatable(self):
        if self.updatable:
            return True

        for policy_name, num_steps in self.num_steps.items():
            if num_steps < self.exploration_steps:
                return False

        self.updatable = True
        return True

    def update_num_steps(self, policy_name, num_steps):
        self.num_steps[policy_name] += num_steps

    def update_alpha(self, policy_name, win_rate):
        if self.updatable:
            self.alphas[policy_name] = np.exp(-self.k * win_rate)
        if self.num_steps[policy_name] >= self.burn_in:
            return 0.0
        return self.alphas[policy_name]

    def get_alpha(self, policy_name):
        return self.alphas[policy_name]

    def get_num_steps(self, policy_name):
        return self.num_steps[policy_name]

    def set_policy_names(self):
        self.policy_names = np.random.permutation(self.policy_names)

    def get_training_policies(self):
        if self.updatable:
            return self.policy_names[0:2]
        return ["static", self.policy_names[0]]

    def is_init(self):
        return self._is_init

    def set_init_done(self):
        self._is_init = True
