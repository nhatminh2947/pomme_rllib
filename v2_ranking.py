'''An example to show how to set up an pommerman game programmatically'''
import argparse
import logging

from power_check.ranking_champions import ranking

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--i", type=int, default=-1, help="first agent")
parser.add_argument("--j", type=int, default=-1, help="second agent")
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
    id_a = params["i"]
    id_b = params["j"]

    logger = logging.getLogger('ranking_champions')
    logger.setLevel(logging.DEBUG)

    if id_a != -1 and id_b != -1:
        agent_names = [agents_1[id_a], agents_1[id_b], agents_2[id_a], agents_2[id_b]]
        file_name = "power_check/logs/{}_vs_{}.txt".format(agents_1[id_a], agents_1[id_b])

        fh = logging.FileHandler(file_name)
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        logger.addHandler(fh)
        logger.addHandler(ch)
        print("agent names v2 ranking: {}".format(agent_names))
        port = 12000 + id_a * 100 + id_b * 10
        ranking(port, agent_names, params)

        logger.removeHandler(fh)
        logger.removeHandler(ch)
    else:
        for i in [3, 5]:
            for j in [7, 8]:
                agent_names = [agents_1[i], agents_1[j], agents_2[i], agents_2[j]]
                file_name = "power_check/logs/{}_vs_{}.txt".format(agents_1[i], agents_1[j])

                fh = logging.FileHandler(file_name)
                fh.setLevel(logging.INFO)

                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                fh.setFormatter(formatter)

                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)

                logger.addHandler(fh)
                logger.addHandler(ch)
                print("agent names v2 ranking: {}".format(agent_names))
                port = 12000 + i * 100 + j * 10
                ranking(port, agent_names, params)

                logger.removeHandler(fh)
                logger.removeHandler(ch)


if __name__ == '__main__':
    main()
