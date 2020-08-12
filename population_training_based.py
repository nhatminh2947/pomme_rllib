import numpy as np
import ray
from ray.rllib.utils.schedules import ConstantSchedule


class PopulationBasedTraining:
    def __init__(self, policy_names, t_select=0.45, perturb_prob=0.1, perturb_val=0.2, burn_in=2e7,
                 ready_num_steps=1e7):
        self.population_size = len(policy_names)
        self.t_select = t_select
        self.perturb_prob = perturb_prob
        self.perturb_val = perturb_val
        self.policy_names = policy_names
        self.burn_in = burn_in
        self.last_num_steps_since_evolution = {policy_name: 0 for policy_name in self.policy_names}
        self.ready_num_steps = ready_num_steps

        self.hyperparameters = {"lr": (1e-5, 1e-3),
                                "clip_param": (0.1, 0.3)}

    def select(self, player_a):
        player_b = self.policy_names[np.random.randint(0, self.population_size)]
        while player_b == player_a:
            player_b = self.policy_names[np.random.randint(0, self.population_size)]
        ers = ray.get_actor("ers")
        if ray.get(ers.expected_score.remote(player_a, player_b)) < self.t_select:
            return player_b
        return None

    def inherit(self, trainer, src, dest):
        self.copy_weight(trainer, src, dest)

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

    def explore(self, trainer, policy_name):
        policy = trainer.get_policy(policy_name)

        new_lr = self.explore_helper(policy.cur_lr, self.hyperparameters["lr"])
        policy.lr_schedule = ConstantSchedule(new_lr, framework="torch")

        new_clip_param = self.explore_helper(policy.config["clip_param"], self.hyperparameters["clip_param"])
        policy.config["clip_param"] = new_clip_param

    def explore_helper(self, old_value, range):
        if np.random.random() > self.perturb_prob:  # resample
            return np.random.uniform(low=range[0], high=range[1], size=None)

        if np.random.random() < 0.5:  # perturb_val = 0.8
            return old_value * (1 - self.perturb_val)

        return old_value * (1 + self.perturb_val)

    def is_eligible(self, policy_name):
        ers = ray.get_actor("ers")
        num_steps = ray.get(ers.get_num_steps.remote(policy_name))

        if num_steps >= self.burn_in:
            return num_steps - self.last_num_steps_since_evolution[policy_name] > self.ready_num_steps

        return False

    def run(self, trainer):
        ers = ray.get_actor("ers")
        population = ray.get(ers.population)

        trainer.get_weights()

        for policy_name in self.policy_names:
            if self.is_eligible(policy_name):
                parent_policy = self.select(policy_name)
                if parent_policy is not None:
                    self.inherit(trainer, parent_policy, policy_name)
                    self.explore(trainer, policy_name)
                    ers = ray.get_actor("ers")
                    num_steps = ray.get(ers.get_num_steps.remote(policy_name))
                    self.last_num_steps_since_evolution[policy_name] = num_steps
