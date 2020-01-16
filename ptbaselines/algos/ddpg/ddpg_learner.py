import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, parameters_to_vector, vector_to_parameters
import torch.optim as optim
import copy
import numpy as np
from ptbaselines.algos.common.models import get_network_builder
from ptbaselines.algos.common.torch_utils import init_weight
from ptbaselines.algos.common import torch_utils
from ptbaselines.common.mpi_running_mean_std import RunningMeanStd
from ptbaselines.common import mpi_util
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def normalize(x, stats):
    if stats is None:
        return x
    state_mean = stats.mean()
    state_std = stats.std()
    state_mean = np.array(state_mean, dtype=np.float32) if not isinstance(state_mean, np.ndarray) else state_mean.astype(np.float32)
    state_std = np.array(state_std, dtype=np.float32) if not isinstance(state_std, np.ndarray) else state_std.astype(np.float32)
    return (x - torch_utils.toTensor(state_mean)) / (torch_utils.toTensor(state_std) + 1e-8)

def denormalize(x, stats):
    if stats is None:
        return x
    state_mean = stats.mean()
    state_std = stats.std()
    state_mean = np.array(state_mean, dtype=np.float32) if not isinstance(state_mean, np.ndarray) else state_mean.astype(np.float32)
    state_std = np.array(state_std, dtype=np.float32) if not isinstance(state_std, np.ndarray) else state_std.astype(np.float32)
    return x * torch_utils.toTensor(state_std) + torch_utils.toTensor(state_mean)

class DDPG(object):
    def __init__(self, actor, critic, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.):

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        # Observation normalization.
        if self.normalize_observations:
            self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None

        # Return normalization.
        if self.normalize_returns:
            self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # target networK
        self.target_critic = None
        self.target_actor = None
        self.initialize()

        if param_noise:
            self.adaptive_actor = copy.deepcopy(self.actor)
            self.pertubed_actor = copy.deepcopy(self.actor)
            self.adaptive_actor.to(torch_utils.device)
            self.pertubed_actor.to(torch_utils.device)
            for param in self.adaptive_actor.parameters():
                param.requires_grad = False
            for param in self.pertubed_actor.parameters():
                param.requires_grad = False
            self.reset()

        # optim
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = self.critic_lr, weight_decay=self.critic_l2_reg)

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale

        B = obs0.shape[0]
        for b in range(B):
            self.memory.append(obs0[b], action[b], reward[b], obs1[b], terminal1[b])
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs0[b]]))

    def normalize_obs(self, obs):
        return torch.clamp(normalize(obs, self.obs_rms), self.observation_range[0], self.observation_range[1])

    def perturb_params(self, src_net, dst_net, param_noise_stddev):
        params = parameters_to_vector(src_net.parameters())
        vector_to_parameters(params, dst_net.parameters())
        for param in dst_net.parameters():
            param.data += torch.randn_like(param.data) * param_noise_stddev

    def adapt_param_noise(self):
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None

        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        obs0 = torch_utils.toTensor(batch['obs0']).float()
        normalize_obs0 = self.normalize_obs(obs0)
        self.perturb_params(self.actor, self.adaptive_actor, self.param_noise.current_stddev)
        with torch.no_grad():
            actions = self.actor(normalize_obs0)
            adaptive_actions = self.adaptive_actor(normalize_obs0)
            distance = torch.sqrt(torch.pow(actions - adaptive_actions, 2.0).mean())

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        self.param_noise.adapt(mean_distance.data.cpu().item())
        return mean_distance

    def popart(self, old_mean, old_std):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        new_std = self.ret_rms.std()
        new_mean = self.ret_rms.mean()

        self.critic.fc.weight.data *= old_std / new_std
        self.critic.fc.bias.data = (self.critic.fc.bias.data * old_std + old_mean - new_mean) / new_std
        self.target_critic.fc.weight.data *= old_std / new_std
        self.target_critic.fc.bias.data = (self.target_critic.fc.bias.data * old_std + old_mean - new_mean) / new_std

    def step(self, obs, apply_noise=True, compute_Q=True):
        if isinstance(obs, np.ndarray):
            obs = torch_utils.toTensor(obs).float()
        norm_obs = self.normalize_obs(obs)
        if self.param_noise is not None and apply_noise:
            action = self.pertubed_actor(norm_obs)
        else:
            action = self.actor(norm_obs)
        if compute_Q:
            normalize_value = self.critic(norm_obs, action)
            q = denormalize(torch.clamp(normalize_value, self.return_range[0], self.return_range[1]), self.ret_rms)
        else:
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action[0].shape
            action += torch_utils.toTensor(noise[np.newaxis]).float()
        action = torch.clamp(action, self.action_range[0], self.action_range[1])

        return torch_utils.toNumpy(action), q.data.cpu().item(), None, None

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)
        obs0 = torch_utils.toTensor(batch['obs0'])
        obs1 = torch_utils.toTensor(batch['obs1'])
        rewards = torch_utils.toTensor(batch['rewards'])
        terminals1= torch_utils.toTensor(batch['terminals1'].astype('float32'))
        actions = torch_utils.toTensor(batch['actions'])
        
        normalize_obs0 = self.normalize_obs(obs0)
        normalize_obs1 = self.normalize_obs(obs1)

        # compute target
        Q_obs1 = denormalize(self.target_critic(normalize_obs1, self.target_actor(normalize_obs1)), self.ret_rms)
        target_Q = rewards + (1. - terminals1) * self.gamma * Q_obs1
        critic_target = torch.clamp(normalize(target_Q, self.ret_rms), self.return_range[0], self.return_range[1]).detach()

        if self.normalize_returns and self.enable_popart:
            old_mean = self.ret_rms.mean()
            old_std = self.ret_rms.std()
            self.ret_rms.update(torch_utils.toNumpy(target_Q.view(-1)))
            self.popart(old_mean, old_std)

        # compute critic loss
        Q_obs0 = self.critic(normalize_obs0, actions)
        critic_loss = F.mse_loss(Q_obs0, critic_target)

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        mpi_util.average_gradients(self.critic_optimizer.param_groups)
        self.critic_optimizer.step()

        # compute actor loss
        actor_actions = self.actor(normalize_obs0)
        critic_with_actor = denormalize(torch.clamp(self.critic(normalize_obs0, actor_actions), self.return_range[0], self.return_range[1]), self.ret_rms)
        actor_loss = -critic_with_actor.mean()

        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        mpi_util.average_gradients(self.actor_optimizer.param_groups)
        self.actor_optimizer.step()

        return critic_loss.data.cpu().item(), actor_loss.data.cpu().item()

    def initialize(self):
        mpi_util.sync_from_root(self.actor.parameters())
        mpi_util.sync_from_root(self.critic.parameters())
        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic.to(torch_utils.device)
        self.actor.to(torch_utils.device)
        self.target_critic.to(torch_utils.device)
        self.target_actor.to(torch_utils.device)

    def update_target(self):
        # target critic
        critic_params = parameters_to_vector(self.critic.parameters())
        target_critic_params = parameters_to_vector(self.target_critic.parameters())
        target_critic_params = (1. - self.tau) * target_critic_params + self.tau * critic_params
        vector_to_parameters(target_critic_params, self.target_critic.parameters())

        # target actor
        actor_params = parameters_to_vector(self.actor.parameters())
        target_actor_params = parameters_to_vector(self.target_actor.parameters())
        target_actor_params = (1. - self.tau) * target_actor_params + self.tau * actor_params
        vector_to_parameters(target_actor_params, self.target_actor.parameters())

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.perturb_params(self.actor, self.pertubed_actor, self.param_noise.current_stddev)

    def save(self, path):
        torch.save(self.actor.state_dict(), path)
    
    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.actor.to(torch_utils.device)
