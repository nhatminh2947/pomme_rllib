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
                in_channels=14,
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

        self.lstm = nn.LSTM(302, 128, batch_first=True)

        self.actor_layers = nn.Linear(128, 22)

        self.critic_layers = nn.Linear(128, 1)

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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        x = input_dict["obs"]["conv_features"]
        x = self.shared_layers(x)

        if type(input_dict["prev_rewards"]) != torch.Tensor:
            input_dict["prev_rewards"] = torch.tensor(input_dict["prev_rewards"], device=device)

        last_reward = torch.reshape(input_dict["prev_rewards"], [-1, 1]).float()

        if type(input_dict["prev_actions"]) != torch.Tensor:
            prev_actions = np.array(input_dict["prev_actions"], dtype=np.int)
        else:
            prev_actions = np.array(input_dict["prev_actions"].cpu().numpy(), dtype=np.int)

        # prev_actions = [np.asarray(prev_action, dtype=np.int) for prev_action in prev_actions]
        prev_actions = [prev_actions[:, i] for i in range(self.action_space.shape[0])]
        one_hot_prev_actions = torch.cat(
            [nn.functional.one_hot(torch.tensor(a), space) for a, space in zip(prev_actions, self.action_space.nvec)],
            axis=-1
        )

        x = torch.cat((x, input_dict["obs"]["features"], last_reward, one_hot_prev_actions.float().to(device)), dim=1)

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
