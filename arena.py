import pommerman
import ray
from pommerman import constants
from pommerman.agents import DockerAgent
import agents
from agents import RayAgent

ray.init(local_mode=True)
id = 380
checkpoint_dir = "/home/lucius/ray_results/2vs2_radio_sp/PPO_PommeMultiAgent-v3_0_2020-08-09_14-31-21_o9o3yp8"
checkpoint = "{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, id, id)

#   Agent   win/loss/tie
#   skynet  56/15/29
#   dypm    2/71/27

agent_list = [[
    # RayAgent(checkpoint),
    agents.CautiousAgent(),
    agents.SimpleAgent(),
    agents.CautiousAgent(),
    agents.SimpleAgent(),
    # DockerAgent("multiagentlearning/dypm.1", port=12001),
    # RayAgent(checkpoint),
    # DockerAgent("multiagentlearning/dypm.2", port=12002),
], [
    agents.SimpleAgent(),
    agents.CautiousAgent(),
    agents.SimpleAgent(),
    agents.CautiousAgent(),
    # DockerAgent("multiagentlearning/dypm.1", port=12003),
    # RayAgent(checkpoint),
    # DockerAgent("multiagentlearning/dypm.2", port=12004),
    # RayAgent(checkpoint),
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
