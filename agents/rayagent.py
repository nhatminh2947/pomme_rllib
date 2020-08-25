import numpy as np
from gym import spaces
from pommerman.agents import BaseAgent
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog

import utils
from memory import Memory
from models import thirdteenth_model
from policies import SmartRandomPolicy, SimplePolicy, NeotericPolicy, CautiousPolicy, \
    SmartRandomNoBombPolicy
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v3


class RayAgent(BaseAgent):
    def __init__(self, checkpoint):
        super().__init__()

        env_id = "PommeTeamCompetition-v0"
        env_config = {
            "env_id": env_id,
            "render": False,
            "game_state_file": None,
            "center": True,
            "input_size": 9,
            "policies": ["policy_0", "static_1"],
            "evaluate": True
        }

        tune.register_env("PommeMultiAgent-v3", lambda x: v3.RllibPomme(env_config))
        ModelCatalog.register_custom_model("13th_model", thirdteenth_model.TorchRNNModel)

        obs_space = utils.get_obs_space(9, is_full_conv=False)
        act_space = spaces.MultiDiscrete([6, 8, 8])

        def gen_policy():
            config = {
                "model": {
                    "custom_model": "13th_model",
                    "custom_model_config": {
                        "in_channels": utils.NUM_FEATURES,
                        "input_size": 9
                    },
                    "no_final_linear": True,
                },
                "framework": "torch",
                "explore": True
            }
            return PPOTorchPolicy, obs_space, act_space, config

        policies = {
            "policy_0": gen_policy(),
            "static_1": (StaticPolicy, utils.original_obs_space, spaces.Discrete(6), {}),
            "smartrandomnobomb_2": (SmartRandomNoBombPolicy, utils.original_obs_space, spaces.Discrete(6), {}),
            "smartrandom_3": (SmartRandomPolicy, utils.original_obs_space, spaces.Discrete(6), {}),
            "cautious_4": (CautiousPolicy, utils.original_obs_space, spaces.Discrete(6), {}),
            "neoteric_5": (NeotericPolicy, utils.original_obs_space, act_space, {}),
        }

        for i in range(4):
            policies["policy_{}".format(len(policies))] = gen_policy()

        self.ppo_agent = PPOTrainer(config={
            "env_config": env_config,
            "num_workers": 0,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": utils.policy_mapping,
                "policies_to_train": ["policy_0"],
            },
            "observation_filter": "NoFilter",
            "framework": "torch"
        }, env="PommeMultiAgent-v3")

        self.ppo_agent.restore(checkpoint)

        self.memory = Memory(0)

        self.state = [
            np.zeros(128, dtype=np.float),
            np.zeros(128, dtype=np.float)
        ]
        self.prev_action = np.zeros(3, )
        self.prev_reward = 0

    def set_init_state(self):
        self.state = [
            np.zeros(128, dtype=np.float),
            np.zeros(128, dtype=np.float)
        ]

        self.prev_action = np.zeros(3, )
        self.prev_reward = 0
        self.memory = Memory(0)

    def episode_end(self, reward):
        self.set_init_state()

    def act(self, obs, action_space):
        if self.memory.obs is None:
            self.memory.init_memory(obs)
        else:
            self.memory.update_memory(obs)

        action, self.state, _ = self.ppo_agent.compute_action(
            observation=utils.featurize_v8(obs, centering=True, input_size=9),
            state=self.state,
            policy_id="policy_0",
            prev_action=self.prev_action,
            prev_reward=self.prev_reward,
            explore=False)

        self.prev_action = action

        return tuple(action)
