from typing import Dict

import ray
from pommerman import constants
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

from metrics import Metrics


class PommeCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        winners = None

        for k, v in episode.agent_rewards.items():
            agent_name = k[0]
            name = agent_name.split("_")
            if name[0] == "opponent":
                continue
            # print(episode.last_info_for(agent_name))
            info = episode.last_info_for(agent_name)

            if "winners" in info:
                winners = info["winners"]

            agent_stat = info["metrics"]

            for key in Metrics:
                episode.custom_metrics["agent_{}/{}".format(agent_name, key.name)] = agent_stat[key.name]

        if winners is None:
            episode.custom_metrics["Tie"] = 1
        elif winners == [0, 2]:
            episode.custom_metrics["team_0_win"] = 1
            episode.custom_metrics["team_1_win"] = 0
        else:
            episode.custom_metrics["team_0_win"] = 0
            episode.custom_metrics["team_1_win"] = 1

    # def on_train_result(self, trainer, result: dict, **kwargs):
    #     g_helper = ray.util.get_actor("g_helper")
    #     g_helper.set_agent_names.remote()


def limit_gamma_explore(config):
    config["gamma"] = min(config["gamma"], 0.999)
    return config


def policy_mapping(agent_id):
    parts = agent_id.split("_")
    team = int(parts[1])

    if parts[0] == "training":
        return "policy_{}".format(team)
    return "static"
