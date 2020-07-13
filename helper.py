import ray


@ray.remote(num_cpus=0.25, num_gpus=0)
class Helper:
    def __init__(self, population_size, policies, env):
        self.population_size = population_size
        self.agent_names = {}
        self.policies = policies
        self._is_init = False
        self.env = env

    def set_agent_names(self):
        if self.env == "1vs1":
            self.agent_names = ['training_0_0', 'static_0_1']
        else:
            self.agent_names = []
            for k in range(4):
                if k % 2 == 1:
                    self.agent_names.append("static_{}_{}".format(k % 2, k))
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
