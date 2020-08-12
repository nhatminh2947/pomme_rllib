import torch
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import LearningRateSchedule, ValueNetworkMixin, \
    EntropyCoeffSchedule, KLCoeffMixin
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import setup_mixins

from . import PBTParamsMixin


# def before_init(policy, obs_space, action_space, config):
#     PBTParamsMixin.__init__(policy, config)


def after_init(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    weights = torch.load("/home/lucius/working/projects/pomme_rllib/resources/model_weight.pt")
    # print(policy.get_weights())
    policy.set_weights(weights)
    # print(weights)
    # print(policy.get_weights())


PBTTorchPolicy = PPOTorchPolicy.with_updates(
    name="PBTTorchPolicy",
    # before_init=before_init,
    after_init=after_init,
    mixins=[PBTParamsMixin, LearningRateSchedule, ValueNetworkMixin, EntropyCoeffSchedule, KLCoeffMixin]
)

PBTTrainer = PPOTrainer.with_updates(
    default_policy=PBTTorchPolicy
)
