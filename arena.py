import pommerman
import ray
from pommerman import constants

import agents

ray.init(local_mode=True)
id = 450
checkpoint_dir = "/home/lucius/ray_results/2vs2_sp/PPO_PommeMultiAgent-v2_0_2020-08-03_17-04-08zag8lm3i"
checkpoint = "{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, id, id)

agent_list = [[
    # RayAgent(checkpoint),
    agents.CautiousAgent(),
    agents.NeotericAgent(),
    agents.CautiousAgent(),
    agents.NeotericAgent(),
    # agents.DockerAgent("multiagentlearning/skynet955", port=12001),
    # RayAgent(checkpoint),
    # agents.DockerAgent("multiagentlearning/skynet955", port=12002),
], [
    # RayAgent(checkpoint),
    agents.NeotericAgent(),
    agents.CautiousAgent(),
    agents.NeotericAgent(),
    agents.CautiousAgent(),
    # agents.DockerAgent("multiagentlearning/skynet955", port=12001),
    # RayAgent(checkpoint),
    # agents.DockerAgent("multiagentlearning/skynet955", port=12002),
]]

env = [pommerman.make('PommeRadioCompetition-v2', agent_list[0]),
       pommerman.make('PommeRadioCompetition-v2', agent_list[1])]
# Run the episodes just like OpenAI Gym
win = 0
loss = 0
tie = 0

for i in range(100):
    state = env[i % 2].reset()
    done = False
    while not done:
        # env.render()
        actions = env[i % 2].act(state)
        state, reward, done, info = env[i % 2].step(actions)

        if done:
            result = constants.Result.Loss
            if info["result"] == constants.Result.Tie:
                result = constants.Result.Tie
                tie += 1
            elif info["winners"] == [[0, 2], [1, 3]][i % 2]:
                result = constants.Result.Win
                win += 1

            print("{} {} {}".format(agent_list[0][0], agent_list[0][1], result.name))

print("{}/{}/{}".format(win, 100 - tie - win, tie))

env[0].close()
env[1].close()
