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
        original_obs = restore_original_dimensions(torch.tensor(obs_batch), self.observation_space, "torch")

        actions = []
        messages_1 = []
        messages_2 = []
        for i in range(len(obs_batch)):
            obs = {}
            for item in original_obs:
                obs[item] = original_obs[item][i].numpy()

                if item in ["ammo", "can_kick", "blast_strength", "teammate"]:
                    obs[item] = np.argwhere(obs[item] == 1)[0][0]

            result = self.agent.act(obs, None)
            actions.append(result[0])
            messages_1.append(result[1])
            messages_2.append(result[2])

        return tuple((np.array(actions), np.array(messages_1), np.array(messages_2))), [], {}

    # def compute_single_action(self,
    #                           obs,
    #                           state=None,
    #                           prev_action=None,
    #                           prev_reward=None,
    #                           info=None,
    #                           episode=None,
    #                           clip_actions=False,
    #                           explore=None,
    #                           timestep=None,
    #                           **kwargs):
    #     return self.agent.act(obs, None), [], {}

    def learn_on_batch(self, samples):
        return {}
