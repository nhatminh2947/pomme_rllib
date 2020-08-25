import time

import pommerman
import ray
from pommerman import constants

from agents import RayAgent

ray.init(local_mode=True)
id = 500
checkpoint_dir = "/home/lucius/ray_results/team_radio_1/PPO_PommeMultiAgent-v3_0_2020-08-13_03-33-55jymb7f2n"
checkpoint = "{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, id, id)

#   Agent                               win/loss/tie
#   skynet                              51/ 2/47
#   navocado                            53/ 7/40
#   dypm                                 6/14/80
#   hakozaki                             3/55/42
#   eisenach                             8/82/10
#   nips19-inspir-ai.inspir              1/24/75
#   nips19-sumedhgupta.neoterics        57/ 8/35
#   nips19-pauljasek.thing1andthing2     1/56/43
#   nips19-gorogm.gorogm                 4/88/ 8

agent_list = [[
    RayAgent(checkpoint),
    RayAgent(checkpoint),
    # agents.SimpleAgent(),
    # DockerAgent("multiagentlearning/nips19-pauljasek.thing1andthing2", port=12001),
    # DockerAgent("multiagentlearning/eisenach", port=12001),
    # DockerAgent("multiagentlearning/dypm.1", port=12001),
    RayAgent(checkpoint),
    # DockerAgent("multiagentlearning/nips19-pauljasek.thing1andthing2", port=12002),
    # DockerAgent("multiagentlearning/eisenach", port=12002),
    # DockerAgent("multiagentlearning/dypm.2", port=12001),
    RayAgent(checkpoint),
    # agents.SimpleAgent(),
    # agents.StaticAgent(),
], [
    # agents.SimpleAgent(),
    # DockerAgent("multiagentlearning/nips19-pauljasek.thing1andthing2", port=12003),
    # DockerAgent("multiagentlearning/eisenach", port=12003),
    # DockerAgent("multiagentlearning/dypm.1", port=12003),
    RayAgent(checkpoint),
    RayAgent(checkpoint),
    # agents.SimpleAgent(),
    # DockerAgent("multiagentlearning/nips19-pauljasek.thing1andthing2", port=12004),
    # DockerAgent("multiagentlearning/nips19-gorogm.gorogm", port=12004),
    # DockerAgent("multiagentlearning/dypm.2", port=12004),
    RayAgent(checkpoint),
    RayAgent(checkpoint),
    # RayAgent(checkpoint),
    # RayAgent(checkpoint),
    # agents.StaticAgent(),
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

            print(f"Match {i} {agent_list[0][0]} {agent_list[0][1]} {result.name}")

print("{}/{}/{}".format(win, 100 - tie - win, tie))

env[0].close()
env[1].close()
