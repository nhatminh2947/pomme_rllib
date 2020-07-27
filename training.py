import numpy as np
import ray
from gym import spaces
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog

import arguments
import utils
from PopulationBasedTraining import PopulationBasedTraining
from eloranking import EloRatingSystem
from helper import Helper
from models import one_vs_one_model, third_model, fourth_model, fifth_model, sixth_model, seventh_model, eighth_model, \
    nineth_model, tenth_model
from policies.random_policy import RandomPolicy
from policies.rnd_policy import RNDTrainer, RNDPPOPolicy
from policies.simple_policy import SimplePolicy
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v0, v1, v2, one_vs_one
from utils import PommeCallbacks, policy_mapping


def initialize(params):
    # env_id = "PommeTeamCompetition-v0"
    # env_id = "PommeTeam-v0"
    # env_id = "PommeFFACompetitionFast-v0"
    # env_id = "OneVsOne-v0"
    # env_id = "PommeRadioCompetition-v2"

    env_config = {
        "env_id": params["env_id"],
        "render": params["render"],
        "game_state_file": params["game_state_file"],
        "center": params["center"],
        "input_size": params["input_size"]
    }

    ModelCatalog.register_custom_model("1vs1", one_vs_one_model.ActorCriticModel)
    ModelCatalog.register_custom_model("third_model", third_model.ActorCriticModel)
    ModelCatalog.register_custom_model("fourth_model", fourth_model.ActorCriticModel)
    ModelCatalog.register_custom_model("fifth_model", fifth_model.ActorCriticModel)
    ModelCatalog.register_custom_model("sixth_model", sixth_model.ActorCriticModel)
    ModelCatalog.register_custom_model("seventh_model", seventh_model.ActorCriticModel)
    ModelCatalog.register_custom_model("eighth_model", eighth_model.ActorCriticModel)
    ModelCatalog.register_custom_model("nineth_model", nineth_model.ActorCriticModel)
    ModelCatalog.register_custom_model("tenth_model", tenth_model.ActorCriticModel)

    tune.register_env("PommeMultiAgent-v0", lambda x: v0.RllibPomme(env_config))
    tune.register_env("PommeMultiAgent-v1", lambda x: v1.RllibPomme(env_config))
    tune.register_env("PommeMultiAgent-v2", lambda x: v2.RllibPomme(env_config))
    tune.register_env("PommeMultiAgent-1vs1", lambda x: one_vs_one.RllibPomme(env_config))
    if params["env_id"] == "OneVsOne-v0":
        obs_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, 8, 8))
    elif params["custom_model"] == "eighth_model":
        obs_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, params["input_size"], params["input_size"]))
    else:
        obs_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, 11, 11))
    act_space = spaces.Discrete(6)

    # Policy setting
    def gen_policy():
        config = {
            "model": {
                "custom_model": params["custom_model"],
                "custom_options": {
                    "in_channels": utils.NUM_FEATURES,
                    "input_size": params["input_size"]
                },
                "no_final_linear": True,
            },
            "lr": np.random.uniform(low=1e-5, high=1e-3),
            "clip_param": np.random.uniform(low=0.1, high=0.5),
            "use_pytorch": True
        }
        return RNDPPOPolicy if params['use_rnd'] else PPOTorchPolicy, obs_space, act_space, config

    policies = {
        "policy_{}".format(i): gen_policy() for i in range(params["populations"])
    }

    policies_to_train = list(policies.keys())

    policies["opponent"] = gen_policy()
    policies["random"] = (RandomPolicy, obs_space, act_space, {})
    policies["static"] = (StaticPolicy, obs_space, act_space, {})
    policies["simple"] = (SimplePolicy, obs_space, act_space, {})

    helper = Helper.options(name="helper").remote(params["populations"],
                                                  params["alpha_coeff"])

    helper.set_agent_names.remote()

    ers = EloRatingSystem.options(name="ers").remote(policy_names=policies_to_train, k=1)

    print("Training policies:", policies.keys())

    return env_config, policies, policies_to_train


# How to Implement Self Play with PPO? [rllib]
# https://github.com/ray-project/ray/issues/6669
#
# https://github.com/ray-project/ray/issues/6669#issuecomment-602234412
# 1. Define a trainable policy and several other non-trainable policies up front. The non-trainable policies
#   will be the "prior selves" and we will update them as we train. Also define the sampling distribution
#   for the non-trainable policies in the policy mapping function like @josjo80 did above.
# 2. Train until a certain metric is met (trainable policy wins greater than 60% of the time).
# 3. Update a list of "prior selves" weights that can be sampled from to update each of the non-trainable policies.
# 4. Update the weights of the non-trainable policies by sampling from the list of "prior selves" weights.
# 5. Back to step 2. Continue process until agent is satisfactorily trained.

def train(config, reporter):
    trainer = PPOTrainer(config=config, env="PommeMultiAgent-v2")

    pbt = PopulationBasedTraining(config["multiagent"]["policies_to_train"])

    while True:
        result = trainer.train()
        reporter(**result)


def training_team(params):
    env_config, policies, policies_to_train = initialize(params)

    trainer = PPOTrainer
    if params['use_rnd']:
        trainer = RNDTrainer

    trials = tune.run(
        train,
        restore=params["restore"],
        resume=params["resume"],
        name=params["name"],
        num_samples=params['num_samples'],
        queue_trials=params["queue_trials"],
        stop={
            # "training_iteration": params["training_iteration"],
            "timesteps_total": params["timesteps_total"]
        },
        checkpoint_freq=params["checkpoint_freq"],
        checkpoint_at_end=True,
        verbose=1,
        config={
            "gamma": params["gamma"],
            # "lr": params["lr"],
            "entropy_coeff": params["entropy_coeff"],
            "kl_coeff": params["kl_coeff"],  # disable KL
            "batch_mode": "complete_episodes" if params["complete_episodes"] else "truncate_episodes",
            "rollout_fragment_length": params["rollout_fragment_length"],
            "env": "PommeMultiAgent-{}".format(params["env"]),
            "env_config": env_config,
            "num_workers": params["num_workers"],
            "num_cpus_per_worker": params["num_cpus_per_worker"],
            "num_envs_per_worker": params["num_envs_per_worker"],
            "num_gpus_per_worker": params["num_gpus_per_worker"],
            "num_gpus": params["num_gpus"],
            "train_batch_size": params["train_batch_size"],
            "sgd_minibatch_size": params["sgd_minibatch_size"],
            # "clip_param": params["clip_param"],
            "lambda": params["lambda"],
            "num_sgd_iter": params["num_sgd_iter"],
            "vf_share_layers": True,
            "vf_loss_coeff": params["vf_loss_coeff"],
            "vf_clip_param": params["vf_clip_param"],
            "callbacks": PommeCallbacks,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": policies_to_train,
            },
            "observation_filter": params["filter"],  # should use MeanStdFilter
            "evaluation_num_episodes": params["evaluation_num_episodes"],
            "evaluation_interval": params["evaluation_interval"],
            "metrics_smoothing_episodes": 100,
            "log_level": "ERROR",
            "use_pytorch": True
        }
    )


if __name__ == "__main__":
    parser = arguments.get_parser()
    args = parser.parse_args()
    params = vars(args)
    print(params)

    ray.shutdown()
    ray.init(local_mode=params["local_mode"], memory=52428800, object_store_memory=4e10)

    training_team(params)
