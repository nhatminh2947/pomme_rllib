import pommerman
import ray
from gym import spaces
from pommerman import agents
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog

import utils
from customize_rllib import policy_mapping
from memory import Memory
from models import one_vs_one_model
from models import third_model
from policies.random_policy import RandomPolicy
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v0, v2
from utils import featurize

ray.init()
env_id = "PommeTeamCompetition-v0"
env = pommerman.make(env_id, [])

obs_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, 11, 11))
act_space = spaces.Discrete(6)


def gen_policy():
    config = {
        "model": {
            "custom_model": "torch_conv_0",
            "custom_options": {
                "in_channels": utils.NUM_FEATURES,
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
policies["opponent"] = gen_policy()
policies["random"] = (RandomPolicy, obs_space, act_space, {})
policies["static"] = (StaticPolicy, obs_space, act_space, {})

env_config = {
    "env_id": env_id,
    "render": False,
    "game_state_file": None
}

ModelCatalog.register_custom_model("torch_conv_0", third_model.ActorCriticModel)
ModelCatalog.register_custom_model("1vs1", one_vs_one_model.ActorCriticModel)

ppo_agent = PPOTrainer(config={
    "env_config": env_config,
    "num_workers": 0,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping,
        "policies_to_train": ["policy_0"],
    },
    "observation_filter": "NoFilter",
    "use_pytorch": True
}, env=v2.RllibPomme)

# fdb733b6
checkpoint = 400
checkpoint_dir = "/home/lucius/ray_results/team_radio/PPO_PommeMultiAgent-v2_0_2020-06-29_15-38-45iow9_yax"
ppo_agent.restore("{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, checkpoint, checkpoint))

agent_list = []
for agent_id in range(4):
    agent_list.append(agents.StaticAgent())
env = pommerman.make(env_id, agent_list=agent_list)

memories = [
    Memory(0),
    Memory(1),
    Memory(2),
    Memory(2),
]

policy = ppo_agent.get_policy("policy_0")
weights = policy.get_weights()
for i in range(1):
    obs = env.reset()
    for i, memory in enumerate(memories):
        memory.init_memory(obs[i])

    done = False
    total_reward = 0
    while not done:
        env.render()
        actions = env.act(obs)

        actions[0] = ppo_agent.compute_action(observation=featurize(memories[0].obs), policy_id="policy_0")
        actions[0] = int(actions[0])
        actions[2] = ppo_agent.compute_action(observation=featurize(memories[2].obs), policy_id="policy_0")
        actions[2] = int(actions[2])
        obs, reward, done, info = env.step(actions)

        memories[0].update_memory(obs[0])
        memories[2].update_memory(obs[2])
        total_reward += reward[0]
        if done:
            print("info:", info)
            print("reward:", total_reward)
            print("=========")
    env.render(close=True)
    # env.close()
