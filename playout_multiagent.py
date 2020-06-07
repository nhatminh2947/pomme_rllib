import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.models import ModelCatalog

import pommerman
from pommerman import agents
from pommerman import configs
from pommerman import constants
from pommerman.envs.v0 import Pomme
from models.third_model import ActorCriticModel
import pomme_env

ray.init(num_cpus=5, num_gpus=1)

env_config = configs.phase_0_team_v0_env()
env = Pomme(**env_config['env_kwargs'])
act_space = env.action_space
ModelCatalog.register_custom_model("torch_conv", ActorCriticModel)
agent_names = ["ppo_agent_1", "ppo_agent_2"]

ppo_agent = PPOTrainer(config={
    "env_config": {
        "agent_names": agent_names,
        "env_id": "Mines-PommeTeam-v0",
        "phase": 0
    },
    "num_workers": 0,
    "num_gpus": 0,
    "multiagent": {
        "policies": {
            "ppo_policy": (PPOTFPolicy, obs_space, act_space, {
                "model": {
                    "custom_model": "3rd_model",
                    "use_lstm": True,
                }
            }),
        },
        "policy_mapping_fn": (lambda agent_id: "ppo_policy"),
        "policies_to_train": ["ppo_policy"],
    },
}, env=pomme_env.PommeMultiAgent)

# fdb733b6
checkpoint = 950
checkpoint_dir = "/home/nhatminh2947/ray_results/3rd_model_no_wood_static/PPO_PommeMultiAgent_283d4406_0_2020-03-24_04-09-09mjgzr90e"
ppo_agent.restore("{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, checkpoint, checkpoint))

agents_list = [agents.StaticAgent(),
               agents.StaticAgent(),
               agents.StaticAgent(),
               agents.StaticAgent()]
env_id = "PommeTeam-nowood-v0"
env = pommerman.make(env_id, agents_list)

penv = pomme_env.PommeMultiAgent({
    "agent_names": agent_names,
    "env_id": env_id,
    "phase": 0
})

for i in range(1):
    obs = env.reset()

    done = False
    step = 0
    while not done:
        env.render()
        actions = env.act(obs)

        actions[1] = ppo_agent.compute_action(observation=penv.featurize(obs[1]), policy_id="ppo_policy")
        actions[3] = ppo_agent.compute_action(observation=penv.featurize(obs[3]), policy_id="ppo_policy")

        obs, reward, done, info = env.step(actions)
        features = penv.featurize(obs[1])
        for i in range(13):
            print("i:", i)
            print(features["board"][:, :, i])
            print("======")
        print(obs[1]["board"])
        print()
        print(obs[1]["bomb_life"])
        print("step:", step)
        print("alive:", obs[1]["alive"])
        print("actions:", [constants.Action(action) for action in actions])

        print("reward:", reward)
        print("done:", done)
        print("info:", info)
        print("=========")
        step += 1
    env.render(close=True)
    # env.close()
