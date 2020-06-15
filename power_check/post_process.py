import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--i", type=int, default=0, help="first agent")
parser.add_argument("--j", type=int, default=0, help="second agent")
parser.add_argument("--port", type=int, default=0, help="start port for agents")
parser.add_argument("--first_half", type=int, default=50, help="number of first half matches to play")
parser.add_argument("--second_half", type=int, default=50, help="number of second half matches to play")
parser.add_argument("--log", action="store_true")
args = parser.parse_args()
params = vars(args)

agents_1 = ["dypm.1", "eisenach", "hakozakijunctions", "navocado", "skynet955", "nips19-gorogm.gorogm",
            "nips19-pauljasek.thing1andthing2", "nips19-sumedhgupta.neoterics", "nips19-inspir-ai.inspir"]
agents_2 = ["dypm.2", "eisenach", "hakozakijunctions", "navocado", "skynet955", "nips19-gorogm.gorogm",
            "nips19-pauljasek.thing1andthing2", "nips19-sumedhgupta.neoterics", "nips19-inspir-ai.inspir"]

i = params["i"]
j = params["j"]
port = params["port"]

first_half_result = {"Win": 0,
                     "Loss": 0,
                     "Tie": 0}

second_half_result = {"Win": 0,
                      "Loss": 0,
                      "Tie": 0}

with open("new_logs/{}_vs_{}.txt".format(agents_1[i], agents_1[j]), "r") as file:
    lines = file.readlines()

    for i, line in enumerate(lines):
        result = line.split()[2]
        if i < 50:
            first_half_result[result] += 1
        else:
            second_half_result[result] += 1

    print(first_half_result)
    print(second_half_result)