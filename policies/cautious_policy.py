from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override

from agents.borealai.simple_agent_cautious_bomb import CautiousAgent


class CautiousPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = CautiousAgent()

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        **kwargs):
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        return [self.agent.act(obs, None) for obs in obs_batch], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
