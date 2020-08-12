from typing import Dict

import ray
from gym import spaces
from pommerman import constants
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy

import arguments
import utils
from eloranking import EloRatingSystem
from metrics import Metrics
from models import one_vs_one_model, eighth_model, eleventh_model, twelfth_model, thirdteenth_model
from policies import SmartRandomPolicy, StaticPolicy, NeotericPolicy, CautiousPolicy, \
    SmartRandomNoBombPolicy
from rllib_pomme_envs import v0, v1, v2, v3, one_vs_one
from utils import policy_mapping

parser = arguments.get_parser()
args = parser.parse_args()
params = vars(args)

pbt = None


class PommeCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        info = None
        ers = ray.get_actor("ers")

        policies = list(set([policy for _, policy in episode.agent_rewards]))
        policy_0_win = False

        for (agent_name, policy), v in episode.agent_rewards.items():
            if policy == "policy_0":
                info = episode.last_info_for(agent_name)

                agent_stat = info["metrics"]
                num_steps = ray.get(ers.update_num_steps.remote(policy, info["num_steps"]))
                episode.custom_metrics["policy_0/num_steps"] = num_steps

                for key in Metrics:
                    if "{}/{}".format(policy, key.name) not in episode.custom_metrics:
                        episode.custom_metrics["{}/{}".format(policy, key.name)] = 0
                    episode.custom_metrics["{}/{}".format(policy, key.name)] += agent_stat[key.name] / 2

                if info["result"] == constants.Result.Win:
                    _, _, agent_id = agent_name.split("_")

                    if int(agent_id) in info["winners"]:
                        policy_0_win = True

        if policy_0_win and policies[0] != "policy_0":
            policies[0], policies[1] = policies[1], policies[0]
        elif not policy_0_win and policies[0] == "policy_0":
            policies[0], policies[1] = policies[1], policies[0]

        score = 1
        if info["result"] == constants.Result.Tie:
            score /= 2

        expected_score = ray.get(ers.expected_score.remote(policies[0], policies[1]))
        rating_0 = ray.get(ers.update_rating.remote(policies[0], expected_score, score))
        expected_score = ray.get(ers.expected_score.remote(policies[1], policies[0]))
        rating_1 = ray.get(ers.update_rating.remote(policies[1], expected_score, 1 - score))

        episode.custom_metrics["{}/elo_rating".format(policies[0])] = rating_0
        episode.custom_metrics["{}/elo_rating".format(policies[1])] = rating_1

        if info["result"] == constants.Result.Tie:
            episode.custom_metrics["{}/tie_rate".format(policies[0])] = 1
            episode.custom_metrics["{}/tie_rate".format(policies[1])] = 1
            episode.custom_metrics["{}/win_rate".format(policies[0])] = 0
            episode.custom_metrics["{}/win_rate".format(policies[1])] = 0
        else:
            episode.custom_metrics["{}/tie_rate".format(policies[0])] = 0
            episode.custom_metrics["{}/tie_rate".format(policies[1])] = 0
            episode.custom_metrics["{}/win_rate".format(policies[0])] = 1
            episode.custom_metrics["{}/win_rate".format(policies[1])] = 0

    def on_train_result(self, trainer, result: dict, **kwargs):
        ers = ray.get_actor("ers")

        if result["custom_metrics"]:
            if "policy_0/EnemyDeath_mean" in result["custom_metrics"]:
                alpha = ray.get(ers.update_alpha.remote("policy_0",
                                                        result["custom_metrics"]["policy_0/EnemyDeath_mean"]))
                result["custom_metrics"]["{}/alpha".format("policy_0")] = alpha

        policy_name = ray.get(ers.update_population.remote())
        if policy_name is not None:
            utils.copy_weight(trainer, "policy_0", policy_name)


def initialize():
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
        "input_size": params["input_size"],
        "evaluate": False
    }

    ModelCatalog.register_custom_model("1vs1", one_vs_one_model.ActorCriticModel)
    ModelCatalog.register_custom_model("8th_model", eighth_model.ActorCriticModel)
    ModelCatalog.register_custom_model("11th_model", eleventh_model.TorchRNNModel)
    ModelCatalog.register_custom_model("12th_model", twelfth_model.TorchRNNModel)
    ModelCatalog.register_custom_model("13th_model", thirdteenth_model.TorchRNNModel)

    tune.register_env("PommeMultiAgent-v0", lambda x: v0.RllibPomme(env_config))
    tune.register_env("PommeMultiAgent-v1", lambda x: v1.RllibPomme(env_config))
    tune.register_env("PommeMultiAgent-v2", lambda x: v2.RllibPomme(env_config))
    tune.register_env("PommeMultiAgent-v3", lambda x: v3.RllibPomme(env_config))
    tune.register_env("PommeMultiAgent-1vs1", lambda x: one_vs_one.RllibPomme(env_config))
    if params["env_id"] == "OneVsOne-v0":
        obs_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, 8, 8))
    else:
        obs_space = utils.get_obs_space(params["input_size"], is_full_conv=params["full_conv"])

    act_space = spaces.MultiDiscrete([6, 8, 8])

    # Policy setting
    def gen_policy(explore=True):
        config = {
            "model": {
                "custom_model": params["custom_model"],
                "custom_model_config": {
                    "in_channels": utils.NUM_FEATURES,
                    "input_size": params["input_size"]
                },
                "no_final_linear": True,
            },
            "explore": explore,
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

    for i in range(params["n_histories"]):
        policies["policy_{}".format(len(policies))] = gen_policy(False)

    policy_names = list(policies.keys())

    ers = EloRatingSystem.options(name="ers").remote(
        policy_names=policy_names,
        n_histories=params["n_histories"],
        alpha_coeff=params["alpha_coeff"],
        burn_in=params["burn_in"],
        k=0.1
    )

    print("Training policies:", policies.keys())

    return env_config, policies, policy_names


def training_team():
    env_config, policies, policies_to_train = initialize()

    trainer = PPOTrainer

    trials = tune.run(
        trainer,
        restore=params["restore"],
        resume=params["resume"],
        name=params["name"],
        num_samples=params['num_samples'],
        queue_trials=params["queue_trials"],
        stop={
            "timesteps_total": params["timesteps_total"]
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
            "env": "PommeMultiAgent-{}".format(params["env"]),
            "env_config": env_config,
            "num_workers": params["num_workers"],
            "num_cpus_per_worker": params["num_cpus_per_worker"],
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
            "clip_actions": False,
            "observation_filter": params["filter"],  # NoFilter/MeanStdFilter
            "evaluation_num_episodes": params["evaluation_num_episodes"],
            "evaluation_interval": params["evaluation_interval"],
            "metrics_smoothing_episodes": 100,
            "log_level": "ERROR",
            "framework": "torch"
        }
    )


if __name__ == "__main__":
    print(params)

    ray.shutdown()
    ray.init(local_mode=params["local_mode"], memory=52428800, object_store_memory=4e10)

    training_team()
