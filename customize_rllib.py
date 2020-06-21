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

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        g_helper = ray.util.get_actor("g_helper")
        if not ray.get(g_helper.is_init.remote()):
            for k, v in episode.agent_rewards.items():
                agent_name = k[0]
                name = agent_name.split("_")
                if name[0] == "opponent":
                    continue
                episode.custom_metrics["agent_{}/num_bombs".format(agent_name)] = 0

                for key in Metrics:
                    if key.name not in episode.custom_metrics:
                        episode.custom_metrics["agent_{}/{}".format(agent_name, key.name)] = 0

            g_helper.set_init_done.remote()
            # print("Init DONE")

        for k, v in episode.agent_rewards.items():
            agent_name = k[0]
            name = agent_name.split("_")
            # print("agent name:", agent_name)
            if name[0] == "opponent":
                continue
            action = episode.last_action_for(agent_name)
            if action == constants.Action.Bomb.value:
                if "agent_{}/num_bombs".format(agent_name) not in episode.custom_metrics:
                    episode.custom_metrics["agent_{}/num_bombs".format(agent_name)] = 0
                episode.custom_metrics["agent_{}/num_bombs".format(agent_name)] += 1

    def on_train_result(self, trainer, result: dict, **kwargs):
        g_helper = ray.util.get_actor("g_helper")
        # ray.get(g_helper.set_agent_names.remote())


def limit_gamma_explore(config):
    config["gamma"] = min(config["gamma"], 0.999)
    return config


def policy_mapping(agent_id):
    # agent_id pattern training/opponent_policy-id_agent-num
    # print("Calling to policy mapping {}".format(agent_id))
    name = agent_id.split("_")
    if name[0] == "opponent":
        return "static"
    return "policy_{}".format(name[1])
