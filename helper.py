import ray
import numpy as np


@ray.remote(num_cpus=0.25, num_gpus=0)
class Helper:
    def __init__(self, population_size, policies):
        self.population_size = population_size
        self.agent_names = {}
        self.policies = policies
        self._is_init = False

    def set_agent_names(self):
        self.agent_names = []
        for k in range(4):
            self.agent_names.append("training_{}_{}".format(k % 2, k))

        if np.random.random() > 0.5:
            self.agent_names[0], self.agent_names[1] = self.agent_names[1], self.agent_names[0]
            self.agent_names[2], self.agent_names[3] = self.agent_names[3], self.agent_names[2]

        print("called set_agent_names")
        print(self.agent_names)

    def get_agent_names(self):
        return self.agent_names

    def is_init(self):
        return self._is_init

    def set_init_done(self):
        self._is_init = True
