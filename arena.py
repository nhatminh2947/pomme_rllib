import pommerman
import ray
from pommerman import agents
from pommerman import constants

from agents.rayagent import RayAgent

ray.init(local_mode=True)

agent_list = [
    RayAgent(),
    agents.DockerAgent("multiagentlearning/skynet955", port=12001),
    RayAgent(),
    agents.DockerAgent("multiagentlearning/skynet955", port=12002),
]
env = pommerman.make('PommeRadioCompetition-v2', agent_list)
# Run the episodes just like OpenAI Gym
for i_episode in range(50):
    state = env.reset()
    done = False
    while not done:
        # env.render()
        actions = env.act(state)
        state, reward, done, info = env.step(actions)

        if done:
            result = constants.Result.Loss
            if info["result"] == constants.Result.Tie:
                result = constants.Result.Tie
            elif info["winners"] == [0, 2]:
                result = constants.Result.Win

            print("{} {} {}".format(agent_list[0], agent_list[1], result.name))

env.close()
