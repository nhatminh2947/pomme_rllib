import argparse
import random

import pommerman
import ray
from gym import spaces
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import PopulationBasedTraining

from customize_rllib import PommeCallbacks, limit_gamma_explore
from models.third_model import ActorCriticModel
from policies.random_policy import RandomPolicy
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v0
from rllib_pomme_envs import v1


def training_team(params):
    # env_id = "PommeTeamCompetition-v0"
    # env_id = "PommeFFACompetitionFast-v0"
    env_id = "PommeTeam-v0"

    env_config = {
        "env_id": env_id,
        "render": params["render"],
        "game_state_file": params["game_state_file"]
    }

    env = pommerman.make(env_id, [])
    obs_space = spaces.Box(low=0, high=20, shape=(17, 11, 11))
    act_space = env.action_space

    ModelCatalog.register_custom_model("torch_conv_0", ActorCriticModel)

    tune.register_env("PommeMultiAgent-v0", lambda x: v0.RllibPomme(env_config))
    tune.register_env("PommeMultiAgent-v1", lambda x: v1.RllibPomme(env_config))

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
        return PPOTorchPolicy, obs_space, act_space, config

    policies = {
        "policy_{}".format(i): gen_policy() for i in range(2)
    }
    policies["random"] = (RandomPolicy, obs_space, act_space, {})
    policies["static"] = (StaticPolicy, obs_space, act_space, {})
    print(policies.keys())

    def policy_mapping(agent_id):
        if agent_id % 2 == 0:
            return "policy_0"
        return "static"

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

    trials = tune.run(
        PPOTrainer,
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
            "env": v1.RllibPomme,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_workers", type=int, default=0, help="number of worker")
    parser.add_argument("--num_gpus", type=float, default=1, help="number of gpu")
    parser.add_argument("--num_envs_per_worker", type=int, default=1)
    parser.add_argument("--num_gpus_per_worker", type=float, default=0.0)
    parser.add_argument("--num_sgd_iter", type=int, default=3)
    parser.add_argument("--sgd_minibatch_size", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=8192)
    parser.add_argument("--complete_episodes", action="store_true")
    parser.add_argument("--rollout_fragment_length", type=int, default=128)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--vf_clip_param", type=float, default=2.0)
    parser.add_argument("--vf_loss_coeff", type=float, default=0.5)
    parser.add_argument("--kl_coeff", type=float, default=0.2)
    parser.add_argument("--entropy_coeff", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--training_iteration", type=int, default=1000)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--name", type=str, default="experiment")
    parser.add_argument("--game_state_file", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--local_mode", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--perturbation_interval", type=int, default=10)
    parser.add_argument("--evaluation_num_episodes", type=int, default=10)
    parser.add_argument("--evaluation_interval", type=int, default=None)
    parser.add_argument("--queue_trials", action="store_true")
    parser.add_argument("--time_attr", type=str, default="timesteps_total")

    args = parser.parse_args()
    params = vars(args)
    print(params)

    ray.shutdown()
    ray.init(local_mode=params["local_mode"], object_store_memory=4e10)

    training_team(params)
