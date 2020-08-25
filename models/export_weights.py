import ray
import torch
from ray.rllib.agents.ppo import PPOTrainer
from models.thirdteenth_model import TorchRNNModel
from training import initialize
from utils import policy_mapping

ray.init(num_cpus=4)

env_config, policies, policies_to_train = initialize()

ppo_agent = PPOTrainer(config={
    "num_gpus": 1,
    "env_config": env_config,
    "num_workers": 0,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping,
        "policies_to_train": policies_to_train,
    },
    "observation_filter": "NoFilter",
    "clip_actions": False,
    "framework": "torch"
}, env="PommeMultiAgent-v3")

id = 560
checkpoint_dir = "/home/lucius/ray_results/team_radio_1/PPO_PommeMultiAgent-v3_0_2020-08-24_17-27-10z5na_vwh"
checkpoint = "{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, id, id)

ppo_agent.restore(checkpoint)

for i in range(4):
    mem_size = 0
    weights = ppo_agent.get_policy(f"policy_{i}").get_weights()
    for key in weights:
        parameters = 1
        for value in weights[key].shape:
            parameters *= value

        mem_size += parameters

        weights[key] = torch.tensor(weights[key])
    print(mem_size)
    torch.save(weights,
               f"/home/lucius/working/projects/pomme_rllib/trained_models/model_{i}.pt")

    # model = FourthModel(constants.OBS_SPACE, constants.ACT_SPACE, 6, {}, "model", constants.NUM_FEATURES)
    # model.load_state_dict(torch.load(f"/home/lucius/working/projects/gold_miner/resources/TrainedModels/model_{i}.pt"))

