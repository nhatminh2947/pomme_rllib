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
from models import fourth_model, fifth_model, eighth_model, eleventh_model, twelfth_model
from models import one_vs_one_model
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v2, v3
from utils import policy_mapping

torch, nn = try_import_torch()

ray.init(local_mode=True, num_gpus=1)
# env_id = "PommeTeamCompetition-v0"
# env_id = "PommeTeam-v0"
# env_id = "PommeFFACompetitionFast-v0"
# env_id = "OneVsOne-v0"
env_id = "PommeRadioCompetition-v2"

env = pommerman.make(env_id, [])

obs_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, 11, 11))
act_space = spaces.Tuple(tuple([spaces.Discrete(6)] + [spaces.Discrete(8)] * 2))

ers = EloRatingSystem.options(name="ers").remote(n_histories=4,
                                                 alpha_coeff=0.8,
                                                 burn_in=100000000,
                                                 k=0.1)

env_config = {
    "env_id": env_id,
    "render": False,
    "game_state_file": None,
    "center": False,
    "input_size": 11,
    "policies": ["policy_0", "policy_2"],
    "evaluate": True
}

ModelCatalog.register_custom_model("eighth_model", eighth_model.ActorCriticModel)
ModelCatalog.register_custom_model("fifth_model", fifth_model.ActorCriticModel)
ModelCatalog.register_custom_model("fourth_model", fourth_model.ActorCriticModel)
ModelCatalog.register_custom_model("fifth_model", fifth_model.ActorCriticModel)
ModelCatalog.register_custom_model("11th_model", eleventh_model.TorchRNNModel)
ModelCatalog.register_custom_model("12th_model", twelfth_model.TorchRNNModel)

ModelCatalog.register_custom_model("1vs1", one_vs_one_model.ActorCriticModel)
tune.register_env("PommeMultiAgent-v2", lambda x: v2.RllibPomme(env_config))
tune.register_env("PommeMultiAgent-v3", lambda x: v3.RllibPomme(env_config))


def gen_policy():
    config = {
        "model": {
            "custom_model": "12th_model",
            "custom_model_config": {
                "in_channels": utils.NUM_FEATURES,
                "input_size": 11
            },
            "no_final_linear": True,
        },
        "framework": "torch"
    }
    return PPOTorchPolicy, obs_space, act_space, config


policies = {
    "policy_0": gen_policy(),
    "static_1": (StaticPolicy, obs_space, spaces.Discrete(6), {}),
    # "random_2": (RandomPolicy, obs_space, act_space, {})
}

for i in range(4):
    policies["policy_{}".format(i + 2)] = gen_policy()

ppo_agent = PPOTrainer(config={
    "env_config": env_config,
    "num_workers": 0,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping,
        "policies_to_train": ["policy_0"],
    },
    "observation_filter": "MeanStdFilter",
    "clip_actions": False,
    "framework": "torch"
}, env="PommeMultiAgent-v3")

checkpoint = 200
checkpoint_dir = "/home/lucius/ray_results/2vs2_radio_sp/PPO_PommeMultiAgent-v3_0_2020-08-05_21-37-52w187rxz8"
ppo_agent.restore("{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, checkpoint, checkpoint))

agent_list = []
for agent_id in range(4):
    agent_list.append(StaticAgent())

env = v3.RllibPomme(env_config)

policy = ppo_agent.get_policy("policy_0")
weights = policy.get_weights()

win = 0
loss = 0
tie = 0

for i in range(100):
    state = [[
        np.zeros(128, dtype=np.float),
        np.zeros(128, dtype=np.float)
    ], [
        np.zeros(128, dtype=np.float),
        np.zeros(128, dtype=np.float)
    ], [
        np.zeros(128, dtype=np.float),
        np.zeros(128, dtype=np.float)
    ], [
        np.zeros(128, dtype=np.float),
        np.zeros(128, dtype=np.float)
    ]]

    obs = env.reset()

    agent_names = env.agent_names
    print(agent_names)
    id = i % 2

    total_reward = 0
    while True:
        env.render()
        actions = {agent_name: 0 for agent_name in agent_names}

        for i in range(4):
            name, id, _ = agent_names[i].split("_")
            policy_id = "{}_{}".format(name, id)

            if agent_names[i] in obs:
                if "static" in agent_names[i]:
                    actions[agent_names[i]] = 0
                else:
                    actions[agent_names[i]], state[i], _ = ppo_agent.compute_action(observation=obs[agent_names[i]],
                                                                                    state=state[i],
                                                                                    policy_id=policy_id,
                                                                                    explore=True)
                # print("agent_name:", agent_names[i])
                # print(obs[agent_names[i]][11])
        # actions[id] = int(actions[2])
        #         print(obs[agent_names[i]])

        obs, reward, done, info = env.step(actions)

        if done["__all__"]:
            # print("info:", info)
            print("=========")

            for agent_name in agent_names:
                if agent_name not in info:
                    continue
                if info[agent_name]["result"] != constants.Result.Incomplete:
                    if info[agent_name]["result"] == constants.Result.Tie:
                        print("tie")
                        tie += 1
                    else:
                        _, _, id = agent_name.split("_")
                        if int(id) in info[agent_name]["winners"]:
                            if "policy" in agent_name:
                                print("win")
                                win += 1
                            else:
                                print("loss")
                                loss += 1
                        else:
                            if "policy" not in agent_name:
                                print("win")
                                win += 1
                            else:
                                print("loss")
                                loss += 1
                    break

            # time.sleep(5)
            break
    # env.render(close=True)
    # env.close()

print("Win/loss/tie: {}/{}/{}".format(win, loss, tie))
