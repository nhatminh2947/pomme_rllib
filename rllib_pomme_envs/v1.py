from rllib_pomme_envs import v0
from utils import featurize


# Note: change team for training agents
class RllibPomme(v0.RllibPomme):
    def __init__(self, config):
        super().__init__(config)

    def step(self, action_dict):
        if self.is_render:
            self.render()

        actions = []
        for id in range(4):
            if id in action_dict:
                actions.append(action_dict[id])
            else:
                actions.append(0)

        obs = {}
        rewards = {}
        dones = {}
        infos = {i - 10: {} for i in self.prev_obs[0]['alive']}

        _obs, _reward, _done, _info = self.env.step(actions)

        for id in self.prev_obs[0]['alive']:
            if _done or self.is_done(id - 10, _obs[0]['alive']):
                dones[id - 10] = True
                infos[id - 10]["metrics"] = self.stat[id - 10]

        dones["__all__"] = _done

        for id in range(4):
            if self.is_agent_alive(id):
                obs[id] = featurize(_obs[id])
                rewards[id] = self.reward(id, _obs, _info)
                infos[id].update(_info)

        self.prev_obs = _obs

        return obs, rewards, dones, infos
