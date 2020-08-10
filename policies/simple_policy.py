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
        original_obs = restore_original_dimensions(torch.tensor(obs_batch), self.observation_space, "torch")

        result = []
        for i in range(len(obs_batch)):
            obs = {}
            for item in original_obs:
                obs[item] = original_obs[item][i].numpy()

                if item in ["ammo", "can_kick", "blast_strength", "teammate"]:
                    obs[item] = np.argwhere(obs[item] == 1)[0][0]

            result.append(self.agent.act(obs, None))
        return np.asarray(result), [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
