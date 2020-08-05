from gym import spaces
from pommerman.agents import BaseAgent
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog

import utils
from memory import Memory
from models import eighth_model
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v2
from utils import featurize_v6
from utils import policy_mapping


class RayAgent(BaseAgent):
    def __init__(self, checkpoint):
        super().__init__()

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

        tune.register_env("PommeMultiAgent-v2", lambda x: v2.RllibPomme(env_config))
        ModelCatalog.register_custom_model("eighth_model", eighth_model.ActorCriticModel)

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
        for i in range(4):
            policies["policy_{}".format(i + 2)] = gen_policy()

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

        self.ppo_agent.restore(checkpoint)

        self.memory = Memory(0)

    def act(self, obs, action_space):
        if self.memory.obs is None:
            self.memory.init_memory(obs)
        else:
            self.memory.update_memory(obs)
        action = self.ppo_agent.compute_action(
            observation=featurize_v6(self.memory.obs, centering=False, input_size=11),
            policy_id="policy_0",
            explore=False
        )

        return int(action)
