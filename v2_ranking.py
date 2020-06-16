'''An example to show how to set up an pommerman game programmatically'''
import argparse
import logging

from power_check.ranking_champions import ranking

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

agents_1 = ["hakozakijunctions", "eisenach", "dypm.1", "navocado", "skynet955", "nips19-gorogm.gorogm",
            "nips19-pauljasek.thing1andthing2", "nips19-sumedhgupta.neoterics", "nips19-inspir-ai.inspir"]
agents_2 = ["hakozakijunctions", "eisenach", "dypm.2", "navocado", "skynet955", "nips19-gorogm.gorogm",
            "nips19-pauljasek.thing1andthing2", "nips19-sumedhgupta.neoterics", "nips19-inspir-ai.inspir"]


def main():
    # ranking(params["i"], params["j"], 12000, "power_check/new_logs", params)
    for i in range(0, 5):
        for j in range(5, 9):
            agent_names = [agents_1[i], agents_1[j], agents_2[i], agents_2[j]]
            file_name = "power_check/logs/{}_vs_{}.txt".format(agents_1[i], agents_1[j])
            logger = logging.getLogger('ranking_champions')
            logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(file_name)
            fh.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)

            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)

            logger.addHandler(fh)
            logger.addHandler(ch)
            print("agent names v2 ranking: {}".format(agent_names))
            ranking(12000, agent_names, params)

            logger.removeHandler(fh)
            logger.removeHandler(ch)


if __name__ == '__main__':
    main()
