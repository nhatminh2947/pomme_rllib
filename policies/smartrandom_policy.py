import numpy as np
import torch
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override

from agents.borealai.random_agent import SmartRandomAgent


class SmartRandomPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = SmartRandomAgent()

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        **kwargs):
        return [0 for _ in obs_batch], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
