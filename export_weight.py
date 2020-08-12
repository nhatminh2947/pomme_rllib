import numpy as np
import pommerman
import ray
from gym import spaces
from pommerman import constants
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_torch

import utils
from agents.static_agent import StaticAgent
from eloranking import EloRatingSystem
from models import fourth_model, fifth_model, eighth_model, eleventh_model, twelfth_model, thirdteenth_model
from models import one_vs_one_model
from policies import SmartRandomPolicy, SimplePolicy, NeotericPolicy, CautiousPolicy, StaticPolicy, \
    SmartRandomNoBombPolicy
from rllib_pomme_envs import v2, v3
from utils import policy_mapping

torch, nn = try_import_torch()

if not ray.is_initialized():
    ray.init(local_mode=True, num_gpus=0)
# env_id = "PommeTeamCompetition-v0"
# env_id = "PommeTeam-v0"
# env_id = "PommeFFACompetitionFast-v0"
# env_id = "OneVsOne-v0"
env_id = "PommeRadioCompetition-v2"

input_size = 9
center = True
n_histories = 4

env = pommerman.make(env_id, [])

obs_space = utils.get_obs_space(input_size, False)
act_space = spaces.Tuple(tuple([spaces.Discrete(6)] + [spaces.Discrete(8)] * 2))


def gen_policy():
    config = {
        "model": {
            "custom_model": "13th_model",
            "custom_model_config": {
                "in_channels": utils.NUM_FEATURES,
                "input_size": input_size
            },
            "no_final_linear": True,
        },
        "framework": "torch"
    }
    return PPOTorchPolicy, obs_space, act_space, config


policies = {
    "policy_0": gen_policy(),
    "static_1": (StaticPolicy, utils.original_obs_space, spaces.Discrete(6), {}),
    "smartrandomnobomb_2": (SmartRandomNoBombPolicy, utils.original_obs_space, spaces.Discrete(6), {}),
    "smartrandom_3": (SmartRandomPolicy, utils.original_obs_space, spaces.Discrete(6), {}),
    "cautious_4": (CautiousPolicy, utils.original_obs_space, spaces.Discrete(6), {}),
    "neoteric_5": (NeotericPolicy, utils.original_obs_space, act_space, {}),
}

policy_names = list(policies.keys())

for i in range(n_histories):
    policies["policy_{}".format(len(policies))] = gen_policy()

ers = EloRatingSystem.options(name="ers").remote(
    policy_names=policy_names,
    n_histories=4,
    alpha_coeff=100,
    burn_in=100000000,
    k=0.1
)

env_config = {
    "env_id": env_id,
    "render": False,
    "game_state_file": None,
    "center": center,
    "input_size": input_size,
    "policies": ["policy_0", "policy_0"],
    "evaluate": True
}

ModelCatalog.register_custom_model("eighth_model", eighth_model.ActorCriticModel)
ModelCatalog.register_custom_model("fifth_model", fifth_model.ActorCriticModel)
ModelCatalog.register_custom_model("fourth_model", fourth_model.ActorCriticModel)
ModelCatalog.register_custom_model("fifth_model", fifth_model.ActorCriticModel)
ModelCatalog.register_custom_model("11th_model", eleventh_model.TorchRNNModel)
ModelCatalog.register_custom_model("12th_model", twelfth_model.TorchRNNModel)
ModelCatalog.register_custom_model("13th_model", thirdteenth_model.TorchRNNModel)

ModelCatalog.register_custom_model("1vs1", one_vs_one_model.ActorCriticModel)
tune.register_env("PommeMultiAgent-v2", lambda x: v2.RllibPomme(env_config))
tune.register_env("PommeMultiAgent-v3", lambda x: v3.RllibPomme(env_config))

ppo_agent = PPOTrainer(config={
    "num_gpus": 1,
    "env_config": env_config,
    "num_workers": 0,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping,
        "policies_to_train": ["policy_0"],
    },
    "observation_filter": "NoFilter",
    "clip_actions": False,
    "framework": "torch"
}, env="PommeMultiAgent-v3")

id = 420
checkpoint_dir = "/home/lucius/ray_results/team_radio/PPO_PommeMultiAgent-v3_0_2020-08-12_01-50-02jkoli2e8"
checkpoint = "{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, id, id)

ppo_agent.restore(checkpoint)

print(ppo_agent.get_policy("policy_0").get_weights())
