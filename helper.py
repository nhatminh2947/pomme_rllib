import numpy as np

import ray


@ray.remote(num_cpus=0.25, num_gpus=0)
class Helper:
    def __init__(self, population_size, policies):
        self.population_size = population_size
        self.agent_names = {}
        self.policies = policies
        self._is_init = False

    def set_agent_names(self):
        i, j = np.random.randint(low=0, high=self.population_size, size=2)
        while i == j:
            j = np.random.randint(low=0, high=self.population_size, size=None)
        i = 0
        j = 0
        self.agent_names = []
        for k in range(2):
            if k % 2 == 0:
                self.agent_names.append("training_{}_{}".format(i, k))
            else:
                self.agent_names.append("opponent_{}_{}".format(j, k))

        print("called set_agent_names")
        print(self.agent_names)

    def get_agent_names(self):
        return self.agent_names

    def is_init(self):
        return self._is_init

    def set_init_done(self):
        self._is_init = True
