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
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            Flatten()
        )

        self.actor_layers = nn.Sequential(
            nn.Linear(128, action_space.n)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(128, 1),
        )

        self._shared_layer_out = None

        for p in self.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

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
