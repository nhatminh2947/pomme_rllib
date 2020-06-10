from typing import Dict

from pommerman import constants
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

from metrics import Metrics


class PommeCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        last_info = None
        for agent_name in range(4):
            if episode.last_info_for(agent_name)["result"] != constants.Result.Incomplete:
                last_info = episode.last_info_for(agent_name)
                break



    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        for agent_name in range(4):
            action = episode.last_action_for(agent_name)
            if action == constants.Action.Bomb.value:
                episode.custom_metrics["bomb_agent_{}".format(agent_name)] += 1

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        for agent_name in range(4):
            episode.custom_metrics["bomb_agent_{}".format(agent_name)] = 0

        for key in Metrics:
            if key.name not in episode.custom_metrics:
                episode.custom_metrics[key.name] = 0


def limit_gamma_explore(config):
    config["gamma"] = min(config["gamma"], 0.999)
    return config
