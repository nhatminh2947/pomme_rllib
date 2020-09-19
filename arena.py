import time

import pommerman
import ray
from pommerman import constants
from pommerman.agents import DockerAgent, PlayerAgent

from agents import RayAgent, SimpleAgent, StaticAgent, SmartRandomAgent, SmartRandomAgentNoBomb, RandomAgent
ray.shutdown()
ray.init(local_mode=True, ignore_reinit_error=True)
id = 980
# checkpoint_dir = "/home/lucius/ray_results/team_radio_1/PPO_PommeMultiAgent-v3_0_2020-08-13_03-33-55jymb7f2n"
checkpoint_dir = "/home/lucius/ray_results/team_radio_2/PPO_PommeMultiAgent-v3_0_2020-08-31_01-58-50vgng3gfu"
# checkpoint_dir = "/home/lucius/ray_results/team_radio_3/PPO_PommeMultiAgent-v3_0_2020-09-06_20-29-41zltnm7ll"
# checkpoint_dir = "/home/lucius/ray_results/team_radio_1/PPO_PommeMultiAgent-v3_0_2020-08-15_18-19-350x7kw7f5"
# checkpoint_dir = "/home/lucius/ray_results/team_radio_1/PPO_PommeMultiAgent-v3_0_2020-08-24_17-27-10z5na_vwh"
checkpoint = "{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, id, id)

#   Agent                               win/loss/tie    260         2300        2700
#   skynet                                              55/4/41     43/7/50     36/8/56
#   navocado                                            28/22/50    34/13/53    34/18/48
#   dypm                                                            3/21/76
#   hakozaki                             3/55/42
#   eisenach                             8/82/10
#   nips19-inspir-ai.inspir              1/24/75
#   nips19-sumedhgupta.neoterics        57/ 8/35                    46/28/26
#   nips19-pauljasek.thing1andthing2     1/56/43
#   nips19-gorogm.gorogm                 4/88/ 8

agent_list = [[
    # RandomAgent(),
    # RandomAgent(),
    # RandomAgent(),
    # RandomAgent(),
    RayAgent(checkpoint),
    # RayAgent(checkpoint),
    # PlayerAgent(),
    # SimpleAgent(),
    # SmartRandomAgentNoBomb(),
    # SmartRandomAgent(),
    # DockerAgent("multiagentlearning/nips19-pauljasek.thing1andthing2", port=12051),
    # DockerAgent("multiagentlearning/nips19-sumedhgupta.neoterics", port=12041),
    # DockerAgent("multiagentlearning/nips19-inspir-ai.inspir", port=12011),
    # DockerAgent("multiagentlearning/navocado", port=12001),
    DockerAgent("multiagentlearning/skynet955", port=12005),
    # DockerAgent("multiagentlearning/dypm.1", port=12021),
    # StaticAgent(),
    RayAgent(checkpoint),
    # DockerAgent("multiagentlearning/nips19-pauljasek.thing1andthing2", port=12052),
    # DockerAgent("multiagentlearning/nips19-sumedhgupta.neoterics", port=12042),
    # DockerAgent("multiagentlearning/nips19-inspir-ai.inspir", port=12012),
    # DockerAgent("multiagentlearning/navocado", port=12002),
    # StaticAgent(),
    DockerAgent("multiagentlearning/skynet955", port=12006),
    # DockerAgent("multiagentlearning/dypm.2", port=12022),
    # RayAgent(checkpoint),
    # SimpleAgent(),
    # SmartRandomAgentNoBomb(),
    # SmartRandomAgent(),
    # agents.StaticAgent(),
], [
    # SimpleAgent(),
    # RandomAgent(),
    # DockerAgent("multiagentlearning/nips19-pauljasek.thing1andthing2", port=12053),
    # DockerAgent("multiagentlearning/nips19-sumedhgupta.neoterics", port=12043),
    # StaticAgent(),
    # DockerAgent("multiagentlearning/navocado", port=12003),
    # DockerAgent("multiagentlearning/nips19-inspir-ai.inspir", port=12013),
    DockerAgent("multiagentlearning/skynet955", port=12007),
    # DockerAgent("multiagentlearning/dypm.1", port=12023),
    # StaticAgent(),
    # SmartRandomAgentNoBomb(),
    # SmartRandomAgent(),
    # RayAgent(checkpoint),
    # RayAgent(checkpoint),
    RayAgent(checkpoint),
    # RayAgent(checkpoint),
    # SimpleAgent(),
    # DockerAgent("multiagentlearning/nips19-pauljasek.thing1andthing2", port=12054),
    # DockerAgent("multiagentlearning/nips19-sumedhgupta.neoterics", port=12044),
    # DockerAgent("multiagentlearning/nips19-gorogm.gorogm", port=12004),
    # DockerAgent("multiagentlearning/nips19-inspir-ai.inspir", port=12014),
    # DockerAgent("multiagentlearning/dypm.2", port=12024),
    # DockerAgent("multiagentlearning/navocado", port=12004),
    DockerAgent("multiagentlearning/skynet955", port=12008),
    # StaticAgent(),
    # SmartRandomAgentNoBomb(),
    # SmartRandomAgent(),
    RayAgent(checkpoint),
    # agents.StaticAgent(),
]]

env = [
    pommerman.make('PommeRadioCompetition-v2', agent_list[0]),
    pommerman.make('PommeRadioCompetition-v2', agent_list[1])
]
# Run the episodes just like OpenAI Gym
win = 0
loss = 0
tie = 0

for i in range(100):
    state = env[i % 2].reset()
    done = False
    while not done:
        # env[i % 2].render(record_pngs_dir='/home/lucius/working/projects/pomme_rllib/logs/pngs/suicide')
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
ray.shutdown()
