import pommerman
from pommerman import envs, agents
from rllib_training.envs.pomme_env import PommeMultiAgent
import unittest
import numpy as np


class TestFeaturize(unittest.TestCase):
    def setUp(self):
        agent_list = [
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            # agents.DockerAgent("pommerman/simple-agent", port=12345),
        ]
        self.env = pommerman.make("PommeTeam-v0", agent_list, game_state_file='record/json/000.json')
        self.env.reset()
        self.env.render()

    def test_observation(self):
        a = [[0, 1, 2, 0, 2, 1, 1, 1, 2, 1, 0],
             [1, 10, 0, 0, 2, 2, 2, 0, 0, 13, 2],
             [2, 0, 0, 1, 1, 0, 2, 1, 1, 0, 2],
             [0, 0, 1, 0, 0, 0, 1, 1, 2, 0, 0],
             [2, 2, 1, 0, 0, 1, 0, 1, 2, 2, 2],
             [1, 2, 0, 0, 1, 0, 2, 0, 1, 2, 1],
             [1, 2, 2, 1, 0, 2, 0, 0, 2, 2, 0],
             [1, 0, 1, 1, 1, 0, 0, 0, 2, 0, 1],
             [2, 0, 1, 2, 2, 1, 2, 2, 0, 0, 1],
             [1, 11, 0, 0, 2, 2, 2, 0, 0, 12, 1],
             [0, 2, 2, 0, 2, 1, 0, 1, 1, 1, 0]]

        result = np.array([
            [[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
             [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
             [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
             [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
             [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]],
            [[0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
             [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
             [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0]]
        ], dtype=np.float)
        observations = self.env.get_observations()

        featurized_obs = PommeMultiAgent.featurize(observations[0])
        print(featurized_obs[1] == result[1])
        self.assertTrue(np.alltrue(featurized_obs[0, :, :] == result[0, :, :]))
        self.assertTrue(np.alltrue(featurized_obs[1, :, :] == result[1, :, :]))

        # print(featurized_obs[0, :, :])
        # print(result[0])


if __name__ == '__main__':
    unittest.main()
