from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()


class ActorCriticModel(nn.Module, TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, in_channels):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.shared_layers = nn.Sequential()
        out_channel = 64
        for i in range(4):
            self.shared_layers.add_module(
                "conv_{}".format(i),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1
                )
            )
            self.shared_layers.add_module("conv_relu_{}".format(i), nn.ReLU())
            in_channels = out_channel

        self.shared_layers.add_module("flatten", nn.Flatten())
        in_channels = 11 * 11 * out_channel
        out_channel = 4096

        for i in range(0, 4):
            self.shared_layers.add_module(
                "linear_{}".format(i),
                nn.Linear(in_features=in_channels, out_features=out_channel)
            )
            self.shared_layers.add_module("fc_relu_{}".format(i), nn.ReLU())
            in_channels = out_channel
            out_channel = out_channel // 2

        self.actor_layers = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, action_space.n)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )

        self._shared_layer_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        self._shared_layer_out = self.shared_layers(x)
        logits = self.actor_layers(self._shared_layer_out)

        return logits, state

    def value_function(self):
        return torch.reshape(self.critic_layers(self._shared_layer_out), [-1])
