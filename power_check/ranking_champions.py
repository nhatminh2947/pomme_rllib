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

logger = logging.getLogger('ranking_champions')
logger.setLevel(logging.INFO)


def ranking(i, j, port, dir, params):
    agents_1 = ["hakozakijunctions", "eisenach", "dypm.1", "navocado", "skynet955", "nips19-gorogm.gorogm",
                "nips19-pauljasek.thing1andthing2", "nips19-sumedhgupta.neoterics", "nips19-inspir-ai.inspir"]
    agents_2 = ["hakozakijunctions", "eisenach", "dypm.2", "navocado", "skynet955", "nips19-gorogm.gorogm",
                "nips19-pauljasek.thing1andthing2", "nips19-sumedhgupta.neoterics", "nips19-inspir-ai.inspir"]
    file_name = "{}/{}_vs_{}.txt".format(dir, agents_1[i], agents_1[j])
    fh = logging.FileHandler(file_name)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(ch)

    with open(file_name, "a") as log:
        # Create a set of agents (exactly four)
        logger.info("Current match: {} {} {} {}".format(agents_1[i], agents_2[i], agents_1[j], agents_2[j]))

        if params["first_half"] != 0:
            agent_list = [
                agents.DockerAgent("multiagentlearning/{}".format(agents_1[i]), port=port),
                agents.DockerAgent("multiagentlearning/{}".format(agents_1[j]), port=port + 1),
                agents.DockerAgent("multiagentlearning/{}".format(agents_2[i]), port=port + 2),
                agents.DockerAgent("multiagentlearning/{}".format(agents_2[j]), port=port + 3)
            ]

            env = pommerman.make('PommeRadioCompetition-v2', agent_list)

            # Run the episodes just like OpenAI Gym
            state, reward, done, info = None, None, None, None
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
                            logger.info("{} {} {}".format(agents_1[i], agents_1[j], result.name))

                        logger.debug("Match {} result: {}".format(i_episode, result.name))

            env.close()
        logger.debug("Done half match")

        agent_list = [
            agents.DockerAgent("multiagentlearning/{}".format(agents_1[j]), port=port),
            agents.DockerAgent("multiagentlearning/{}".format(agents_1[i]), port=port + 1),
            agents.DockerAgent("multiagentlearning/{}".format(agents_2[j]), port=port + 2),
            agents.DockerAgent("multiagentlearning/{}".format(agents_2[i]), port=port + 3)
        ]

        env = pommerman.make('PommeRadioCompetition-v2', agent_list)
        print("Current match: {} {} {} {}".format(agents_1[j], agents_2[j], agents_1[i], agents_2[i]))

        # Run the episodes just like OpenAI Gym
        state, reward, done, info = None, None, None, None
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
                    elif info["winners"] == [1, 3]:
                        result = constants.Result.Win

                    if params["log"]:
                        logger.info("{} {} {}".format(agents_1[i], agents_1[j], result.name))
                    logger.debug("Match {} result: {}".format(i_episode, result.name))
        env.close()
