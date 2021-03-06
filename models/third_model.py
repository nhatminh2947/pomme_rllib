import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()


class ActorCriticModel(nn.Module, TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, in_channels, feature_dim):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.shared_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(9 * 9 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )

        self.actor_layers = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

        self._shared_layer_out = None

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for i in range(len(self.actor_layers)):
            if type(self.actor_layers[i]) == nn.Linear:
                nn.init.orthogonal_(self.actor_layers[i].weight, 0.01)
                self.actor_layers[i].bias.data.zero_()

        for i in range(len(self.critic_layers)):
            if type(self.critic_layers[i]) == nn.Linear:
                nn.init.orthogonal_(self.critic_layers[i].weight, 0.1)
                self.critic_layers[i].bias.data.zero_()

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        assert not np.any(torch.isnan(x).tolist()), "x has nan {}".format(x)
        assert not np.any(torch.isinf(x).tolist()), "x has inf {}".format(x)

        self._shared_layer_out = self.shared_layers(x)
        assert not np.any(torch.isnan(self._shared_layer_out).tolist()), "self._shared_layer_out has nan {}".format(
            self._shared_layer_out)
        assert not np.any(torch.isinf(self._shared_layer_out).tolist()), "self._shared_layer_out has inf {}".format(
            self._shared_layer_out)
        logits = self.actor_layers(self._shared_layer_out)

        return logits, state

    def value_function(self):
        return torch.reshape(self.critic_layers(self._shared_layer_out), [-1])


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
