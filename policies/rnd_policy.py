import numpy as np
import pommerman
import ray
from gym import spaces
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing, discount
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.torch_ops import sequence_mask

from models.rnd_model import RNDActorCriticModel
from policies.random_policy import RandomPolicy
from policies.static_policy import StaticPolicy
from rllib_pomme_envs import v0, v1
torch, nn = try_import_torch()

INTRINSIC_VF_PREDS = 'intrinsic_vf_preds'
INTRINSIC_REWARD = 'intrinsic_reward'
INTRINSIC_VALUE_TARGETS = 'intrinsic_targets'
INTRINSIC_ADV = 'intrinsic_adv'

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


discounted_reward = RewardForwardFilter(0.99)
reward_rms = RunningMeanStd()


class PPOLoss:
    def __init__(self,
                 dist_class,
                 model,
                 next_obs,
                 intrinsic_value_targets,
                 value_targets,
                 advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 intrinsic_vf_preds,
                 vf_preds,
                 curr_action_dist,
                 intrinsic_value_fn,
                 value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for prob output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Tensor): A bool mask of valid input elements (#2992).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """
        if valid_mask is not None:
            num_valid = torch.sum(valid_mask)

            def reduce_mean_valid(t):
                return torch.sum(t * valid_mask) / num_valid

        else:

            def reduce_mean_valid(t):
                return torch.mean(t)

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = torch.exp(
            curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages * torch.clamp(logp_ratio, 1 - clip_param,
                                     1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if use_gae:
            vf_loss1 = torch.pow(value_fn - value_targets, 2.0)
            vf_clipped = vf_preds + torch.clamp(value_fn - vf_preds,
                                                -vf_clip_param, vf_clip_param)
            vf_loss2 = torch.pow(vf_clipped - value_targets, 2.0)
            vf_loss = torch.max(vf_loss1, vf_loss2)

            intrinsic_vf_loss = torch.pow(intrinsic_value_fn - intrinsic_value_targets, 2.0)

            self.mean_intrinsic_vf_loss = reduce_mean_valid(intrinsic_vf_loss)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            self.total_vf_loss = self.mean_intrinsic_vf_loss + self.mean_vf_loss

            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * (vf_loss + intrinsic_vf_loss) - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = 0.0
            loss = reduce_mean_valid(-surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)

        forward_mse = nn.MSELoss(reduction='none')
        predict_next_state_feature = model.predictor(next_obs)
        target_next_state_feature = model.target(next_obs)
        forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
        mask = torch.rand(len(forward_loss))
        mask = (mask < 0.25).type(torch.cuda.FloatTensor)
        forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.cuda.FloatTensor([1]))

        self.forward_loss = forward_loss

        self.loss = loss + forward_loss


def rnd_ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    mask = None
    if state:
        max_seq_len = torch.max(train_batch["seq_lens"])
        mask = sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = torch.reshape(mask, [-1])

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[SampleBatch.NEXT_OBS],
        train_batch[INTRINSIC_VALUE_TARGETS],
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[INTRINSIC_VF_PREDS],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.intrinsic_value_function(),
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
    )

    return policy.loss_obj.loss


def compute_advantages(rollout,
                       last_r,
                       gamma=0.9,
                       lambda_=1.0,
                       use_gae=True,
                       use_critic=True):
    """
    Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_critic (bool): Whether to use critic (value estimates). Setting
                           this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"

    if use_gae:
        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([last_r])])
        delta_t = (traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        traj[Postprocessing.VALUE_TARGETS] = (
                traj[Postprocessing.ADVANTAGES] +
                traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)

        intrinsic_vpred_t = np.concatenate(
            [rollout[INTRINSIC_VF_PREDS],
             np.array([last_r])])  # double check last r
        delta_t = (traj[INTRINSIC_REWARD] + gamma * intrinsic_vpred_t[1:] - intrinsic_vpred_t[:-1])
        traj[INTRINSIC_ADV] = discount(delta_t, 0.99 * lambda_)
        traj[INTRINSIC_VALUE_TARGETS] = (
                traj[INTRINSIC_ADV] +
                traj[INTRINSIC_VF_PREDS]).copy().astype(np.float32)

    traj[INTRINSIC_ADV] = traj[INTRINSIC_ADV].copy().astype(np.float32)

    traj[Postprocessing.ADVANTAGES] = traj[Postprocessing.ADVANTAGES].copy().astype(np.float32) \
                                      + traj[INTRINSIC_ADV].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)


def postprocess_rnd_ppo_gae(policy,
                            sample_batch,
                            other_agent_batches=None,
                            episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""
    new_sample_batch = {}
    for key in sample_batch:
        new_sample_batch[key] = np.stack(sample_batch[key])

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)

    new_sample_batch[INTRINSIC_REWARD] = policy.model.compute_intrinsic_reward(sample_batch[SampleBatch.NEXT_OBS])
    new_sample_batch = SampleBatch(new_sample_batch)

    total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                     new_sample_batch[INTRINSIC_REWARD]])
    mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
    reward_rms.update_from_moments(mean, std ** 2, count)

    # normalize intrinsic reward
    new_sample_batch[INTRINSIC_REWARD] /= np.sqrt(reward_rms.var)

    batch = compute_advantages(
        new_sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])

    return batch


def rnd_vf_preds_fetches(policy, input_dict, state_batches, model, action_dist):
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        INTRINSIC_VF_PREDS: policy.model.intrinsic_value_function(),
        # INTRINSIC_REWARD: policy.model.compute_intrinsic_reward()
    }


def kl_and_loss_stats(policy, train_batch):
    return {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "intrinsic_vf_loss": policy.loss_obj.mean_intrinsic_vf_loss,
        "total_vf_loss": policy.loss_obj.total_vf_loss,
        "forward_loss": policy.loss_obj.forward_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function(),
            framework="torch"),
        "intrinsic_vf_explained_var": explained_variance(
            train_batch[INTRINSIC_VALUE_TARGETS],
            policy.model.intrinsic_value_function(),
            framework="torch"),
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": policy.entropy_coeff,
    }


RNDPPOPolicy = PPOTorchPolicy.with_updates(
    name="RNDPPOTorchPolicy",
    postprocess_fn=postprocess_rnd_ppo_gae,
    extra_action_out_fn=rnd_vf_preds_fetches,
    stats_fn=kl_and_loss_stats,
    loss_fn=rnd_ppo_surrogate_loss
)

RNDTrainer = PPOTrainer.with_updates(
    name="RNDTrainer",
    default_policy=RNDPPOPolicy
)

if __name__ == '__main__':
    ray.init(local_mode=True)

    ModelCatalog.register_custom_model("rnd_torch_conv", RNDActorCriticModel)
    env_id = "PommeTeam-v0"

    env_config = {
        "env_id": env_id,
        "render": False
    }
    env = pommerman.make(env_id, [])
    obs_space = spaces.Box(low=0, high=20, shape=(17, 11, 11))
    act_space = env.action_space
    tune.register_env("PommeMultiAgent-v1", lambda x: v1.RllibPomme(env_config))


    def gen_policy():
        config = {
            "model": {
                "custom_model": "rnd_torch_conv",
                "custom_options": {
                    "in_channels": 17,
                    "feature_dim": 512
                }
            },
            "gamma": 0.999,
            "use_pytorch": True
        }
        return RNDPPOPolicy, obs_space, act_space, config


    policies = {
        "policy_{}".format(i): gen_policy() for i in range(2)
    }
    policies["random"] = (RandomPolicy, obs_space, act_space, {})
    policies["static"] = (StaticPolicy, obs_space, act_space, {})
    print(policies.keys())


    def policy_mapping(agent_id):
        if agent_id == 0:
            return "policy_0"
        return "static"


    tune.run(
        RNDTrainer,
        name="rnd",
        config={
            "num_workers": 0,
            "num_gpus": 1,
            "env": "PommeMultiAgent-v1",
            "env_config": env_config,
            "train_batch_size": 512,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": ["policy_0"],
            },
            "use_pytorch": True,
            "vf_share_layers": True,
            "vf_loss_coeff": 0.001,
            'log_level': 'INFO',
        }
    )
