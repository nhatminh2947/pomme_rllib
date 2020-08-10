import numpy as np
import torch
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy import Policy

import agents


class NeotericPolicy(Policy):
    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config)
        self.agent = agents.NeotericAgent()

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    def reset(self):
        self.agent = agents.NeotericAgent()

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                        info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
        return [0 for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        return {}
