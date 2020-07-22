import numpy as np
import ray


@ray.remote(num_cpus=0.25, num_gpus=0)
class Helper:
    def __init__(self, population_size, policies, env, alpha_coeff, enemy="static"):
        self.population_size = population_size
        self.agent_names = {}
        self.policies = policies
        self._is_init = False
        self.env = env
        self.enemy = enemy
        self.alpha = 1.0
        self.alpha_coeff = alpha_coeff

    def update_alpha(self, enemy_death_mean):
        self.alpha = 1 - np.tanh(self.alpha_coeff * enemy_death_mean)

    def get_alpha(self):
        return self.alpha

    def set_agent_names(self):
        if self.env == "1vs1":
            self.agent_names = ['training_0_0', 'static_0_1']
        else:
            self.agent_names = []
            for k in range(4):
                if k % 2 == 1:
                    self.agent_names.append("{}_{}_{}".format(self.enemy, k % 2, k))
                else:
                    self.agent_names.append("training_{}_{}".format(k % 2, k))

        print("called set_agent_names")
        print(self.agent_names)

    def get_agent_names(self):
        return self.agent_names

    def is_init(self):
        return self._is_init

    def set_init_done(self):
        self._is_init = True
