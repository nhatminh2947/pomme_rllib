from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override


class NeotericPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
