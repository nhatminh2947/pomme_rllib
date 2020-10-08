from pommerman.agents import BaseAgent


class StaticAgent(BaseAgent):
    def act(self, obs, action_space):
        return 0
