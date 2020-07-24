import logging
import os
import shutil
import unittest

import ray
from gym import spaces
from pommerman import agents, constants
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

from helper import Helper
from policies.random_policy import RandomPolicy
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v2


def setup_logging():
    logger = logging.getLogger('testing')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log.txt')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        ray.init(local_mode=True)
        agent_list = [
            agents.PlayerAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
        ]
        obs_space = spaces.Box(low=0, high=20, shape=(22, 11, 11))
        act_space = spaces.Discrete(6)

        def gen_policy():
            config = {
                "model": {
                    "custom_model": "torch_conv_0",
                    "custom_options": {
                        "in_channels": 23,
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

        env_id = "PommeTeam-v0"
        g_helper = Helper.options(name="g_helper").remote(2, policies, env_id, k=1.2, alpha=0)
        g_helper.set_agent_names.remote()

        env_config = {
            "env_id": env_id,
            "render": False,
            "center": False,
            "input_size": 11,
            "game_state_file": "000.json"
        }
        shutil.rmtree("./pngs")
        os.mkdir("pngs")
        if os.path.exists("log.txt"):
            os.remove("log.txt")

        self.logger = setup_logging()
        self.env = v2.RllibPomme(env_config)

    def test_reward(self):
        actions = [constants.Action.Down.value,
                   constants.Action.Down.value,
                   constants.Action.Bomb.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Down.value,
                   constants.Action.Down.value,
                   constants.Action.Down.value,
                   constants.Action.Up.value,
                   constants.Action.Bomb.value,
                   constants.Action.Up.value,
                   constants.Action.Right.value,
                   constants.Action.Bomb.value,
                   constants.Action.Bomb.value,
                   constants.Action.Bomb.value,
                   constants.Action.Bomb.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Bomb.value,
                   constants.Action.Left.value,
                   constants.Action.Up.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Down.value,
                   constants.Action.Down.value,
                   constants.Action.Down.value,
                   constants.Action.Right.value,
                   constants.Action.Left.value,
                   constants.Action.Bomb.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Down.value,
                   constants.Action.Bomb.value,
                   constants.Action.Up.value,
                   constants.Action.Down.value,
                   constants.Action.Down.value,
                   constants.Action.Down.value,
                   constants.Action.Down.value,
                   constants.Action.Down.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Up.value,
                   constants.Action.Right.value,
                   constants.Action.Right.value,
                   constants.Action.Bomb.value,
                   constants.Action.Down.value,
                   constants.Action.Left.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Up.value,
                   constants.Action.Right.value,
                   constants.Action.Right.value,
                   constants.Action.Bomb.value,
                   constants.Action.Left.value,
                   constants.Action.Left.value,
                   constants.Action.Down.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Up.value,
                   constants.Action.Right.value,
                   constants.Action.Right.value,
                   constants.Action.Right.value,
                   constants.Action.Bomb.value,
                   constants.Action.Left.value,
                   constants.Action.Left.value,
                   constants.Action.Left.value,
                   constants.Action.Down.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Up.value,
                   constants.Action.Right.value,
                   constants.Action.Right.value,
                   constants.Action.Right.value,
                   constants.Action.Right.value,
                   constants.Action.Right.value,
                   constants.Action.Bomb.value,
                   constants.Action.Left.value,
                   constants.Action.Left.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   constants.Action.Stop.value,
                   ]

        obs = self.env.reset()
        total_reward = 0
        for i, action in enumerate(actions):
            self.logger.info("step: {}".format(i))
            self.env.render(record_pngs_dir="./pngs/")
            self.logger.info("enemies of teammate: \n{}".format(obs["training_0_2"][10, :, :]))
            if "opponent_0_1" in obs:
                self.logger.info("enemies of enemies: \n{}".format(obs["opponent_0_1"][10, :, :]))
            for i, feature in enumerate(["Passage", "Rigid", "Wood", "ExtraBomb", "Incrange", "Kick", "Fog", "Position",
                                         "Teammate", "IsAliveTeammate", "Enemies", "NumEnemies", "Bomb_move_1",
                                         "Bomb_move_2", "Bomb_move_3", "Bomb_move_4", "bomb_life",
                                         "bomb_blast_strength", "flame_life", "ammo", "blast_strength", "can_kick"]):
                self.logger.info("id: {} feature: {}\n{}".format(i, feature, obs["training_0_0"][i, :, :]))

            self.logger.info("  actions: {}".format(action))
            actions_dict = {"training_0_0": 5,
                            "opponent_0_1": 0,
                            "training_0_2": 5,
                            "opponent_0_3": 0}
            obs, rewards, done, _info = self.env.step(actions_dict)
            total_reward += rewards["training_0_0"]
            self.logger.info("rewards: {}".format(rewards))
            if done["__all__"]:
                break

        self.logger.info("total reward: {}".format(total_reward))
