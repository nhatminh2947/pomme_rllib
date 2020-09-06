from pommerman.agents import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    def act(self, obs, action_space):
        return np.random.randint(0, 6)
