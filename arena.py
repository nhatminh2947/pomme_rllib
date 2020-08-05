import pommerman
import ray
from pommerman import agents
from pommerman import constants

from agents.rayagent import RayAgent

ray.init(local_mode=True)
id = 450
checkpoint_dir = "/home/lucius/ray_results/2vs2_sp/PPO_PommeMultiAgent-v2_0_2020-08-03_17-04-08zag8lm3i"
checkpoint = "{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, id, id)

agent_list = [
    RayAgent(checkpoint),
    agents.DockerAgent("multiagentlearning/skynet955", port=12001),
    RayAgent(checkpoint),
    agents.DockerAgent("multiagentlearning/skynet955", port=12002),
]
env = pommerman.make('PommeRadioCompetition-v2', agent_list)
# Run the episodes just like OpenAI Gym
for i_episode in range(50):
    state = env.reset()
    done = False
    while not done:
        env.render()
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
