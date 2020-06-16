'''An example to show how to set up an pommerman game programmatically'''
import argparse
import logging

import pommerman
from pommerman import agents, constants

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--i", type=int, default=0, help="first agent")
parser.add_argument("--j", type=int, default=0, help="second agent")
parser.add_argument("--port", type=int, default=0, help="start port for agents")
parser.add_argument("--first_half", type=int, default=50, help="number of first half matches to play")
parser.add_argument("--second_half", type=int, default=50, help="number of second half matches to play")
parser.add_argument("--log", action="store_true")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()
params = vars(args)


def ranking(port, agent_names, params):
    logger = logging.getLogger('ranking_champions')
    logger.setLevel(logging.INFO)

    # Create a set of agents (exactly four)
    if params["first_half"] != 0:
        agent_list = []
        for i in range(4):
            agent_list.append(agents.DockerAgent("multiagentlearning/{}".format(agent_names[i]), port=port + i))

        env = pommerman.make('PommeRadioCompetition-v2', agent_list)

        # Run the episodes just like OpenAI Gym
        for i_episode in range(params["first_half"]):
            state = env.reset()
            done = False
            while not done:
                if params["render"]:
                    env.render()
                actions = env.act(state)
                state, reward, done, info = env.step(actions)

                if done:
                    result = constants.Result.Loss
                    if info["result"] == constants.Result.Tie:
                        result = constants.Result.Tie
                    elif info["winners"] == [0, 2]:
                        result = constants.Result.Win

                    if params["log"]:
                        logger.info("{} {} {}".format(agent_names[0], agent_names[1], result.name))

                    logger.debug("Match {} result: {}".format(i_episode, result.name))

        env.close()

    agent_names[0], agent_names[1] = agent_names[1], agent_names[0]
    agent_names[2], agent_names[3] = agent_names[3], agent_names[2]

    agent_list = []
    for i in range(4):
        agent_list.append(agents.DockerAgent("multiagentlearning/{}".format(agent_names[i]), port=port + i))

    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    # Run the episodes just like OpenAI Gym
    for i_episode in range(params["second_half"]):
        state = env.reset()
        done = False
        while not done:
            if params["render"]:
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            if done:
                result = constants.Result.Loss
                if info["result"] == constants.Result.Tie:
                    result = constants.Result.Tie
                elif info["winners"] == [0, 2]:
                    result = constants.Result.Win

                if params["log"]:
                    logger.info("{} {} {}".format(agent_names[0], agent_names[1], result.name))
                logger.debug("Match {} result: {}".format(i_episode, result.name))
    env.close()
