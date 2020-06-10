import pommerman
import ray
from gym import spaces
from pommerman import agents
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog

from models.third_model import ActorCriticModel
from policies.random_policy import RandomPolicy
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v0
from utils import policy_mapping, featurize

ray.init()
env_id = "PommeTeam-v0"
env = pommerman.make(env_id, [])

obs_space = spaces.Box(low=0, high=20, shape=(17, 11, 11))
act_space = env.action_space


def gen_policy():
    config = {
        "model": {
            "custom_model": "torch_conv_0",
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

env_config = {
    "env_id": env_id,
    "render": False,
    "game_state_file": None
}

ModelCatalog.register_custom_model("torch_conv_0", ActorCriticModel)

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
checkpoint = 600
checkpoint_dir = "/home/lucius/ray_results/two_policies_vs_static_agents/PPO_RllibPomme_0_2020-06-09_23-39-347whmqdrs"
ppo_agent.restore("{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, checkpoint, checkpoint))

agent_list = []
for agent_id in range(4):
    agent_list.append(agents.StaticAgent())
env = pommerman.make("PommeTeam-v0", agent_list=agent_list)

for i in range(1):
    obs = env.reset()

    done = False
    while not done:
        env.render()
        actions = env.act(obs)
        actions[0] = ppo_agent.compute_action(observation=featurize(obs[0]), policy_id="policy_0")
        actions[2] = ppo_agent.compute_action(observation=featurize(obs[2]), policy_id="policy_0")
        obs, reward, done, info = env.step(actions)
        print("reward:", reward)
        print("done:", done)
        print("info:", info)
        print("=========")
    env.render(close=True)
    # env.close()
