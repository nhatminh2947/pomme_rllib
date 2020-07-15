import numpy as np
import pommerman
import ray
from ray import tune
from gym import spaces
from pommerman import agents
from pommerman import constants
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog

import utils
from models import fourth_model, fifth_model, eighth_model
from models import one_vs_one_model
from policies.random_policy import RandomPolicy
from policies.simple_policy import SimplePolicy
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v2
from utils import featurize_v4
from utils import policy_mapping

ray.init()
env_id = "PommeTeamCompetition-v0"
# env_id = "PommeTeam-v0"
# env_id = "PommeFFACompetitionFast-v0"
# env_id = "OneVsOne-v0"
# env_id = "PommeRadioCompetition-v2"

env = pommerman.make(env_id, [])

obs_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, 9, 9))
act_space = spaces.Discrete(6)


def gen_policy():
    config = {
        "model": {
            "custom_model": "eighth_model",
            "custom_options": {
                "in_channels": utils.NUM_FEATURES
            },
            "no_final_linear": True,
        },
        "use_pytorch": True
    }
    return PPOTorchPolicy, obs_space, act_space, config


policies = {
    "policy_{}".format(i): gen_policy() for i in range(2)
}
policies["opponent"] = gen_policy()
policies["random"] = (RandomPolicy, obs_space, act_space, {})
policies["static"] = (StaticPolicy, obs_space, act_space, {})
policies["simple"] = (SimplePolicy, obs_space, act_space, {})

env_config = {
    "env_id": env_id,
    "render": False,
    "game_state_file": None,
    "center": True
}
ModelCatalog.register_custom_model("eighth_model", eighth_model.ActorCriticModel)
ModelCatalog.register_custom_model("fifth_model", fifth_model.ActorCriticModel)
ModelCatalog.register_custom_model("fourth_model", fourth_model.ActorCriticModel)
ModelCatalog.register_custom_model("fifth_model", fifth_model.ActorCriticModel)
ModelCatalog.register_custom_model("1vs1", one_vs_one_model.ActorCriticModel)
tune.register_env("PommeMultiAgent-v2", lambda x: v2.RllibPomme(env_config))

ppo_agent = PPOTrainer(config={
    "env_config": env_config,
    "num_workers": 0,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping,
        "policies_to_train": ["policy_0"],
    },
    "observation_filter": "MeanStdFilter",
    "use_pytorch": True
}, env="PommeMultiAgent-v2")

# fdb733b6
checkpoint = 180
checkpoint_dir = "/home/lucius/ray_results/2vs2_center_obs/PPO_PommeMultiAgent-v2_0_2020-07-14_20-37-52uc9tjyot"
ppo_agent.restore("{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, checkpoint, checkpoint))

agent_list = []
for agent_id in range(4):
    agent_list.append(agents.StaticAgent())
env = pommerman.make(env_id, agent_list=agent_list)

# memories = [
# Memory(0),
# Memory(1),
# Memory(2),
# Memory(2),
# ]

policy = ppo_agent.get_policy("policy_0")
weights = policy.get_weights()

win = 0
loss = 0
tie = 0

for i in range(100):
    obs = env.reset()
    id = np.random.randint(2)

    done = False
    total_reward = 0
    while not done:
        env.render()
        actions = env.act(obs)
        actions[0] = ppo_agent.compute_action(observation=featurize_v4(obs[0], True), policy_id="policy_0",
                                              explore=True)
        # actions[id] = int(actions[0])
        actions[2] = ppo_agent.compute_action(observation=featurize_v4(obs[2], True), policy_id="policy_0",
                                              explore=True)
        # actions[id] = int(actions[2])

        obs, reward, done, info = env.step(actions)

        # memories[0].update_memory(obs[0])
        # memories[2].update_memory(obs[2])
        total_reward += reward[id]
        if done:
            if reward[id] == 1:
                win += 1
            elif info["result"] == constants.Result.Tie:
                tie += 1
            else:
                loss += 1

            print("info:", info)
            print("reward:", total_reward)
            print("=========")
    env.render(close=True)
    # env.close()

print("Win/loss/tie: {}/{}/{}".format(win, loss, tie))
