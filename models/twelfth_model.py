import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
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
                kernel_size=3,
                stride=1),
            nn.ELU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.ELU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1),
            nn.ELU(),
            nn.Flatten()
        )

        self.lstm = nn.LSTM(256, 256, batch_first=True)

        self.actor_layers = nn.Linear(256, 22)

        self.critic_layers = nn.Linear(256, 1)

        self._shared_layer_out = None
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            torch.zeros(128, dtype=torch.float),
            torch.zeros(128, dtype=torch.float)
        ]
        return h

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()

        x = input_dict["obs"]
        x = self.shared_layers(x)

        output, new_state = self.forward_rnn(
            add_time_dimension(x.float(), seq_lens, framework="torch"),
            state,
            seq_lens
        )

        return torch.reshape(output, [-1, self.num_outputs]), new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        self._features, [h, c] = self.lstm(
            inputs,
            [torch.unsqueeze(state[0], 0),
             torch.unsqueeze(state[1], 0)]
        )

        action_out = self.actor_layers(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def value_function(self):
        return torch.reshape(self.critic_layers(self._features), [-1])
