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
        print("Callback on_episode_end")
        print(worker.worker_index)
        winners = None
        g_helper = ray.util.get_actor("g_helper")

        for k, v in episode.agent_rewards.items():
            agent_name = k[0]
            name = agent_name.split("_")
            if name[0] == "opponent":
                continue
            print(episode.last_info_for(agent_name))
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

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        for k, v in episode.agent_rewards.items():
            agent_name = k[0]
            name = agent_name.split("_")
            if name[0] == "opponent":
                continue
            action = episode.last_action_for(agent_name)
            if action == constants.Action.Bomb.value:
                episode.custom_metrics["agent_{}_num_bombs".format(agent_name)] += 1

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        print("Callback on_episode_start")
        for k, v in episode.agent_rewards.items():
            agent_name = k[0]
            name = agent_name.split("_")
            if name[0] == "opponent":
                continue
            episode.custom_metrics["agent_{}/num_bombs".format(agent_name)] = 0

            for key in Metrics:
                if key.name not in episode.custom_metrics:
                    episode.custom_metrics["agent_{}/{}".format(agent_name, key.name)] = 0

    def on_train_result(self, trainer, result: dict, **kwargs):
        g_helper = ray.util.get_actor("g_helper")
        ray.get(g_helper.set_agent_names.remote())
        print("New pair generated")
        print(ray.get(g_helper.get_agent_names.remote()))


def limit_gamma_explore(config):
    config["gamma"] = min(config["gamma"], 0.999)
    return config
