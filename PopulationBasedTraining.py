import numpy as np
import ray


class PopulationBasedTraining:
    def __init__(self, policy_names, t_select=0.45, perturb_prob=0.1, perturb_val=0.2, burn_in=10000000):
        self.population_size = len(policy_names)
        self.t_select = t_select
        self.perturb_prob = perturb_prob
        self.perturb_val = perturb_val
        self.policy_names = policy_names
        self.burn_in = burn_in

        self.hyperparameters = ["lr", "clip_param"]

    def select(self, player_a, ers):
        player_b = self.policy_names[np.random.uniform(0, self.population_size)]
        while player_b == player_a:
            player_b = self.policy_names[np.random.uniform(0, self.population_size)]

        if ers.expected_score(player_b, player_a) < self.t_select:
            return player_b
        return None

    def inherit(self, trainer, src, dest):
        self.copy_weight(trainer, src, dest)

        src_pol = trainer.get_policy(src)
        print("src_pol.config['lr']", src_pol.config["lr"])

        dest_pol = trainer.get_policy(dest)
        print("dest_pol.config['lr']", dest_pol.config["lr"])

        for hyperparameter in self.hyperparameters:
            m = np.random.randint(0, 2)
            dest_pol.config[hyperparameter] = m * dest_pol.config[hyperparameter] + \
                                              (1 - m) * src_pol.config[hyperparameter]

        print("src_pol.config['lr']", src_pol.config["lr"])
        print("dest_pol.config['lr']", dest_pol.config["lr"])

        return None

    def copy_weight(self, trainer, src, dest):
        P0key_P1val = {}
        for (k, v), (k2, v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                    trainer.get_policy(src).get_weights().items()):
            P0key_P1val[k] = v2

        trainer.set_weights({dest: P0key_P1val,
                             src: trainer.get_policy(src).get_weights()})

        for (k, v), (k2, v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                    trainer.get_policy(src).get_weights().items()):
            assert (v == v2).all()

    def mutate(self, trainer, policy_name):
        policy = trainer.get_policy(policy_name)

        for hyperparameter in self.hyperparameters:
            if np.random.random() > self.perturb_prob:  # resample
                policy.config[hyperparameter] = np.random.uniform(low=1e-5, high=1e-1, size=None)
            elif np.random.random() < 0.5:  # perturb_val = 0.8
                policy.config[hyperparameter] = policy.config[hyperparameter] * (1 - self.perturb_val)
            else:  # perturb_val = 1.2
                policy.config[hyperparameter] = policy.config[hyperparameter] * (1 + self.perturb_val)

        # update hyperparameters in storage
        # key = "agt_" + str(pol_i_id)
        # helper = ray.util.get_actor("helper")
        # ray.get(helper.update_hyperparameters.remote(key, pol.config["lr"], pol.config["gamma"]))

    def is_eligible(self, policy_name):
        helper = ray.util.get_actor("helper")
        num_steps = ray.get(helper.get_num_steps.remote(policy_name))

        return num_steps >= self.burn_in

    def run(self, trainer, ers):
        for policy_name in self.policy_names:
            if self.is_eligible(policy_name):
                pol_j_id = self.select(policy_name)
                if pol_j_id is not None:
                    self.inherit(trainer, policy_name, pol_j_id)
                    self.mutate(trainer, policy_name)
