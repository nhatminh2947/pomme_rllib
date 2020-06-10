import random

import pommerman
import ray
from gym import spaces
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import PopulationBasedTraining

import arguments
from customize_rllib import PommeCallbacks, limit_gamma_explore
from models.third_model import ActorCriticModel
from policies.random_policy import RandomPolicy
from policies.rnd_policy import RNDTrainer, RNDPPOPolicy
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v0
from rllib_pomme_envs import v1
from utils import policy_mapping


def initialize(params):
    # env_id = "PommeTeamCompetition-v0"
    # env_id = "PommeFFACompetitionFast-v0"
    env_id = "PommeTeam-v0"

    env_config = {
        "env_id": env_id,
        "render": params["render"],
        "game_state_file": params["game_state_file"]
    }
    ModelCatalog.register_custom_model("torch_conv_0", ActorCriticModel)
    tune.register_env("PommeMultiAgent-v0", lambda x: v0.RllibPomme(env_config))
    tune.register_env("PommeMultiAgent-v1", lambda x: v1.RllibPomme(env_config))

    env = pommerman.make(env_id, [])
    obs_space = spaces.Box(low=0, high=20, shape=(17, 11, 11))
    act_space = env.action_space

    # Policy setting
    def gen_policy():
        config = {
            "model": {
                "custom_model": "torch_conv_0",
                "custom_options": {
                    "in_channels": 17,
                    "feature_dim": 512
                },
                "no_final_linear": True,
            },
            "use_pytorch": True
        }
        return RNDPPOPolicy if params['use_rnd'] else PPOTorchPolicy, obs_space, act_space, config

    policies = {
        "policy_{}".format(i): gen_policy() for i in range(2)
    }

    policies["random"] = (RandomPolicy, obs_space, act_space, {})
    policies["static"] = (StaticPolicy, obs_space, act_space, {})

    return env_config, policies


def training_team(params):
    env_config, policies = initialize(params)

    # PBT setting
    pbt_scheduler = PopulationBasedTraining(
        time_attr=params["time_attr"],
        metric="policy_reward_mean/policy_0",
        mode="max",
        perturbation_interval=params["perturbation_interval"],
        custom_explore_fn=limit_gamma_explore,
        hyperparam_mutations={
            "lr": lambda: random.uniform(0.0001, 0.1),
            # "gamma": lambda: random.uniform(0.85, 0.999)
        })

    if params['use_rnd']:
        trainer = RNDTrainer
    else:
        trainer = PPOTrainer

    trials = tune.run(
        trainer,
        restore=params["restore"],
        resume=params["resume"],
        name=params["name"],
        queue_trials=params["queue_trials"],
        scheduler=pbt_scheduler,
        num_samples=params["num_samples"],
        stop={
            # "training_iteration": params["training_iteration"],
            "timesteps_total": 1000000000
        },
        checkpoint_freq=params["checkpoint_freq"],
        checkpoint_at_end=True,
        verbose=1,
        config={
            "gamma": params["gamma"],
            "lr": params["lr"],
            "entropy_coeff": params["entropy_coeff"],
            "kl_coeff": params["kl_coeff"],  # disable KL
            "batch_mode": "complete_episodes" if params["complete_episodes"] else "truncate_episodes",
            "rollout_fragment_length": params["rollout_fragment_length"],
            "env": "PommeMultiAgent-v1",
            "env_config": env_config,
            "num_workers": params["num_workers"],
            "num_envs_per_worker": params["num_envs_per_worker"],
            "num_gpus_per_worker": params["num_gpus_per_worker"],
            "num_gpus": params["num_gpus"],
            "train_batch_size": params["train_batch_size"],
            "sgd_minibatch_size": params["sgd_minibatch_size"],
            "clip_param": params["clip_param"],
            "lambda": params["lambda"],
            "num_sgd_iter": params["num_sgd_iter"],
            "vf_share_layers": True,
            "vf_loss_coeff": params["vf_loss_coeff"],
            "vf_clip_param": params["vf_clip_param"],
            "callbacks": PommeCallbacks,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": ["policy_0"],
            },
            "observation_filter": "MeanStdFilter",  # should use MeanStdFilter
            "evaluation_num_episodes": params["evaluation_num_episodes"],
            "evaluation_interval": params["evaluation_interval"],
            "log_level": "WARN",
            "use_pytorch": True
        }
    )


def validate(params):
    env_config, policies = initialize(params)
    trainer = PPOTrainer(
        config={
            "gamma": params["gamma"],
            "lr": params["lr"],
            "entropy_coeff": params["entropy_coeff"],
            "kl_coeff": params["kl_coeff"],  # disable KL
            "batch_mode": "complete_episodes" if params["complete_episodes"] else "truncate_episodes",
            "rollout_fragment_length": params["rollout_fragment_length"],
            "env": "PommeMultiAgent-v1",
            "env_config": env_config,
            "num_workers": params["num_workers"],
            "num_envs_per_worker": params["num_envs_per_worker"],
            "num_gpus_per_worker": params["num_gpus_per_worker"],
            "num_gpus": params["num_gpus"],
            "train_batch_size": params["train_batch_size"],
            "sgd_minibatch_size": params["sgd_minibatch_size"],
            "clip_param": params["clip_param"],
            "lambda": params["lambda"],
            "num_sgd_iter": params["num_sgd_iter"],
            "vf_share_layers": True,
            "vf_loss_coeff": params["vf_loss_coeff"],
            "vf_clip_param": params["vf_clip_param"],
            "callbacks": PommeCallbacks,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": ["policy_0"],
            },
            "observation_filter": "MeanStdFilter",  # should use MeanStdFilter
            "evaluation_num_episodes": params["evaluation_num_episodes"],
            "evaluation_interval": params["evaluation_interval"],
            "log_level": "WARN",
            "use_pytorch": True
        }, env="CartPole-v0")


if __name__ == "__main__":
    parser = arguments.get_parser()
    args = parser.parse_args()
    params = vars(args)
    print(params)

    ray.shutdown()
    ray.init(local_mode=params["local_mode"], memory=52428800, object_store_memory=4e10)

    if params["validate"]:
        validate(params)
    else:
        training_team(params)
