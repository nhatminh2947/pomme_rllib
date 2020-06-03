from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from torch.nn import init
import numpy as np

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

        self.intrinsic_value_out = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

        self.extrinsic_value_out = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

        self.rnd = RNDModel(in_channels)

        self._shared_layer_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        self._shared_layer_out = self.shared_layers(x)
        logits = self.actor_layers(self._shared_layer_out)

        return logits, []

    def value_function(self):
        return torch.reshape(self.extrinsic_value_out(self._shared_layer_out), [-1])

    def intrinsic_value_function(self):
        return torch.reshape(self.intrinsic_value_out(self._shared_layer_out), [-1])

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        return policy_loss

    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.from_numpy(next_obs).float().to(self.device)

        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class RNDModel(nn.Module):
    def __init__(self, in_channels):
        super(RNDModel, self).__init__()

        self.in_channels = in_channels

        feature_output = 9 * 9 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature
