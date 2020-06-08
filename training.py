import argparse
import random
from typing import Dict

import pommerman
import ray
from gym import spaces
from pommerman import constants
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.tune.schedulers import PopulationBasedTraining

from models.third_model import ActorCriticModel
from policies.random_policy import RandomPolicy
from policies.static_policy import StaticPolicy
from pomme_env import PommeMultiAgent


class PommeCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        last_info = None
        for agent_name in range(4):
            if episode.last_info_for(agent_name)["result"] != constants.Result.Incomplete:
                last_info = episode.last_info_for(agent_name)
                break

        if last_info["result"] == constants.Result.Win:
            episode.custom_metrics["win"] += 1
        elif last_info["result"] == constants.Result.Tie:
            episode.custom_metrics["tie"] += 1
        else:
            episode.custom_metrics["loss"] += 1

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        for agent_name in range(4):
            action = episode.last_action_for(agent_name)
            if action == constants.Action.Bomb.value:
                episode.custom_metrics["bomb_agent_{}".format(agent_name)] += 1

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        for agent_name in range(4):
            episode.custom_metrics["bomb_agent_{}".format(agent_name)] = 0

        if "win" not in episode.custom_metrics:
            episode.custom_metrics["win"] = 0
        if "loss" not in episode.custom_metrics:
            episode.custom_metrics["loss"] = 0
        if "tie" not in episode.custom_metrics:
            episode.custom_metrics["tie"] = 0


def training_team(params):
    # env_id = "PommeTeamCompetition-v0"
    # env_id = "PommeFFACompetitionFast-v0"
    env_id = "PommeTeam-v0"

    env_config = {
        "env_id": env_id,
        "render": params['render'],
        "game_state_file": params['game_state_file']
    }

    env = pommerman.make(env_id, [])
    obs_space = spaces.Box(low=0, high=20, shape=(17, 11, 11))
    act_space = env.action_space

    ModelCatalog.register_custom_model("torch_conv", ActorCriticModel)

    tune.register_env("PommeMultiAgent-v0", lambda x: PommeMultiAgent(env_config))

    def gen_policy():
        config = {
            "model": {
                "custom_model": "torch_conv",
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
        if agent_id == 0:
            return "policy_0"
        return "static"

    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        perturbation_interval=50,
        hyperparam_mutations={
            "lr": lambda: random.uniform(0.00001, 0.05),
            "gamma": lambda: random.uniform(0.5, 0.999)
        })

    trials = tune.run(
        PPOTrainer,
        restore=params["restore"],
        resume=params["resume"],
        name=params["name"],
        queue_trials=True,
        scheduler=pbt_scheduler,
        num_samples=params['num_samples'],
        stop={
            # "training_iteration": params["training_iteration"],
            "timesteps_total": 100000000
        },
        checkpoint_freq=params["checkpoint_freq"],
        checkpoint_at_end=True,
        config={
            "gamma": params['gamma'],
            "lr": params["lr"],
            "entropy_coeff": params["entropy_coeff"],
            "kl_coeff": 0.0,  # disable KL
            "batch_mode": "complete_episodes" if params['complete_episodes'] else "truncate_episodes",
            "rollout_fragment_length": params['rollout_fragment_length'],
            "env": PommeMultiAgent,
            "env_config": env_config,
            "num_workers": params['num_workers'],
            "num_envs_per_worker": params['num_envs_per_worker'],
            "num_gpus_per_worker": params['num_gpus_per_worker'],
            "num_gpus": params['num_gpus'],
            "train_batch_size": params['train_batch_size'],
            "sgd_minibatch_size": params['sgd_minibatch_size'],
            "clip_param": params['clip_param'],
            "lambda": params['lambda'],
            "num_sgd_iter": params['num_sgd_iter'],
            "vf_share_layers": True,
            "vf_loss_coeff": params['vf_loss_coeff'],
            "vf_clip_param": params["vf_clip_param"],
            "callbacks": PommeCallbacks,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": ["policy_0"],
            },
            # "evaluation_num_episodes": 10,
            # "evaluation_interval": 5,
            "log_level": "WARN",
            "use_pytorch": True
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker')
    parser.add_argument('--num_gpus', type=float, default=1, help='number of gpu')
    parser.add_argument('--train_batch_size', type=int, default=8192)
    parser.add_argument('--sgd_minibatch_size', type=int, default=128)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--num_sgd_iter', type=int, default=3)
    parser.add_argument('--vf_loss_coeff', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--entropy_coeff', type=float, default=0.001)
    parser.add_argument('--complete_episodes', action='store_true')
    parser.add_argument('--rollout_fragment_length', type=int, default=128)
    parser.add_argument('--training_iteration', type=int, default=1000)
    parser.add_argument('--checkpoint_freq', type=int, default=100)
    parser.add_argument('--num_envs_per_worker', type=int, default=1)
    parser.add_argument('--num_gpus_per_worker', type=float, default=0.0)
    parser.add_argument('--vf_clip_param', type=float, default=2.0)
    parser.add_argument('--name', type=str, default="experiment")
    parser.add_argument('--game_state_file', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--local_mode', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=4)
    args = parser.parse_args()
    params = vars(args)

    ray.shutdown()
    ray.init(local_mode=params['local_mode'], object_store_memory=4e10)

    training_team(params)
