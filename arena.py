import pommerman
import ray
from pommerman import constants
from pommerman.agents import DockerAgent
import agents
from agents import RayAgent

import time
ray.init(local_mode=True)
id = 420
checkpoint_dir = "/home/lucius/ray_results/team_radio/PPO_PommeMultiAgent-v3_0_2020-08-12_01-50-02jkoli2e8"
checkpoint = "{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, id, id)

#   Agent   win/loss/tie
#   skynet  56/15/29
#   dypm    2/71/27

agent_list = [[
    RayAgent(checkpoint),
    RayAgent(checkpoint),
    RayAgent(checkpoint),
    # agents.StaticAgent(),
    RayAgent(checkpoint),
    # agents.StaticAgent(),
    # DockerAgent("multiagentlearning/dypm.1", port=12001),
    # DockerAgent("multiagentlearning/dypm.2", port=12002),
], [
    RayAgent(checkpoint),
    RayAgent(checkpoint),
    RayAgent(checkpoint),
    RayAgent(checkpoint),
    # agents.SimpleAgent(),
    # agents.StaticAgent(),
    # agents.SimpleAgent(),
    # DockerAgent("multiagentlearning/dypm.1", port=12003),
    # DockerAgent("multiagentlearning/dypm.2", port=12004),
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
        env[i % 2].render()
        actions = env[i % 2].act(state)

        start_time = time.time()
        state, reward, done, info = env[i % 2].step(actions)
        # print("--- %s seconds ---" % (time.time() - start_time))

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
