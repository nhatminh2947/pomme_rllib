import pommerman
import ray
from gym import spaces
from pommerman import agents
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog

import utils
from customize_rllib import policy_mapping
from models import one_vs_one_model
from policies.random_policy import RandomPolicy
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v0
from utils import featurize

ray.init()
env_id = "OneVsOne-v0"
env = pommerman.make(env_id, [])

obs_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, 8, 8))
act_space = env.action_space


def gen_policy():
    config = {
        "model": {
            "custom_model": "1vs1",
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

# ModelCatalog.register_custom_model("torch_conv_0", ActorCriticModel)
ModelCatalog.register_custom_model("1vs1", one_vs_one_model.ActorCriticModel)

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
}, env=v0.RllibPomme)

# fdb733b6
checkpoint = 500
checkpoint_dir = "/home/lucius/ray_results/one_vs_one/PPO_PommeMultiAgent-1vs1_0_2020-06-21_01-16-48xgm6jedn"
ppo_agent.restore("{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, checkpoint, checkpoint))

agent_list = []
for agent_id in range(2):
    agent_list.append(agents.StaticAgent())
env = pommerman.make("OneVsOne-v0", agent_list=agent_list)

for i in range(1):
    obs = env.reset()

    done = False
    total_reward = 0
    while not done:
        env.render()
        actions = env.act(obs)
        actions[0] = ppo_agent.compute_action(observation=featurize(obs[0]), policy_id="policy_0")
        obs, reward, done, info = env.step(actions)
        total_reward += reward[0]
        if done:
            print("info:", info)
            print("reward:", total_reward)
            print("=========")
    env.render(close=True)
    # env.close()
