from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()


class TorchRNNModel(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, in_channels, input_size):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.shared_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=5,
                padding=2,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2,
                stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_size * input_size * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(512, 512)

        self.actor_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            self.shared_layers.weight.new(1, 512).zero_().squeeze(0),
            self.shared_layers.weight.new(1, 512).zero_().squeeze(0)
        ]
        return h

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        x = self.shared_layers(inputs)

        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),
                torch.unsqueeze(state[1], 0)]
        )

        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def value_function(self):
        return torch.reshape(self.critic_layers(self._features), [-1])
