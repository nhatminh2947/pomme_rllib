from gym import spaces
from pommerman.agents import BaseAgent
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

import utils
from memory import Memory
from policies.static_policy import StaticPolicy
from utils import featurize_v6
from utils import policy_mapping


class ray_agent(BaseAgent):
    def __init__(self):
        super().__init__()
        obs_space = spaces.Box(low=0, high=20, shape=(utils.NUM_FEATURES, 11, 11))
        act_space = spaces.Discrete(6)

        def gen_policy():
            config = {
                "model": {
                    "custom_model": "eighth_model",
                    "custom_options": {
                        "in_channels": utils.NUM_FEATURES,
                        "input_size": 11
                    },
                    "no_final_linear": True,
                },
                "use_pytorch": True
            }
            return PPOTorchPolicy, obs_space, act_space, config

        policies = {
            "policy_0": gen_policy(),
            "static_1": (StaticPolicy, obs_space, act_space, {}),
            # "random_2": (RandomPolicy, obs_space, act_space, {})
        }
        env_id = "PommeTeamCompetition-v0"

        env_config = {
            "env_id": env_id,
            "render": False,
            "game_state_file": None,
            "center": False,
            "input_size": 11,
            "policies": ["policy_0", "policy_2"],
            "evaluate": True
        }

        self.ppo_agent = PPOTrainer(config={
            "env_config": env_config,
            "num_workers": 2,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": ["policy_0"],
            },
            "observation_filter": "MeanStdFilter",
            "use_pytorch": True
        }, env="PommeMultiAgent-v2")

        checkpoint = 300
        checkpoint_dir = "/home/lucius/ray_results/2vs2_sp/PPO_PommeMultiAgent-v2_0_2020-08-03_17-04-08zag8lm3i"
        self.ppo_agent.restore("{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, checkpoint, checkpoint))

        self.memory = Memory(0)

    def act(self, obs, action_space):
        self.memory.update_memory(obs)
        self.ppo_agent.compute_action(observation=featurize_v6(self.memory.obs, centering=False, input_size=11),
                                      policy_id="policy_0",
                                      explore=True)
