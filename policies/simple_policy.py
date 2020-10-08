import numpy as np
import torch
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override

import agents


class SimplePolicy(Policy):
    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config)
        self.agent = agents.SimpleAgent()

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                        info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
        return [0 for _ in obs_batch], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
