import torch
import torch.nn as nn
import numpy as np
import copy

from ptbaselines.common.mpi_running_mean_std import RunningMeanStd
from ptbaselines.algos.common.distributions import make_pdtype
from ptbaselines.algos.common.models import get_network_builder
from ptbaselines.algos.common import torch_utils

import gym

class PolicyWithValue(nn.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, latent, estimate_q=False, vf_latent=None, **tensors):
        super(PolicyWithValue, self).__init__()
        """
        Parameters:
        ----------
        env             RL environment
        latent          latent state from which policy distribution parameters should be inferred
        vf_latent       latent state from which value function should be inferred (if None, then latent is used)
        **tensors       tensorflow tensors for additional attributes such as state or mask
        """

        self.state = None
        self.initial_state = None
        self.__dict__.update(tensors)

        self.latent = latent
        self.vf_latent = vf_latent if vf_latent is not None else latent

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(self.latent.out_dim, env.action_space, init_scale = 0.01)

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.vf = nn.Linear(self.latent.out_dim, env.action_space.n)
            self.q = self.vf
        else:
            self.vf = nn.Linear(self.latent.out_dim, 1)

        # init weight
        torch_utils.init_weight(self.vf)

    def action(self, obs):
        l_out = self.latent(obs)
        pd, _ = self.pdtype.pdfromlatent(l_out)
        return pd.sample()

    def neglogp(self, obs, a):
        l_out = self.latent(obs)
        pd, _ = self.pdtype.pdfromlatent(l_out)
        neglogp = pd.neglogp(a)
        return neglogp, pd

    def step(self, obs, **extra_feed):
        """
        Compute next action(s) given the observation(s)
        Parameters:
        ----------
        observation     observation data (either single or a batch)
        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)
        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        l_out = self.latent(obs)
        pd, _ = self.pdtype.pdfromlatent(l_out)
        a = pd.sample()
        neglogp = pd.neglogp(a)

        vl_out = self.vf_latent(obs)
        v = self.vf(vl_out)

        state = None

        return a, v.squeeze(-1), state, neglogp

    def value(self, obs, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)
        Parameters:
        ----------
        observation     observation data (either single or a batch)
        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)
        Returns:
        -------
        value estimate
        """

        vl_out = self.vf_latent(obs)
        v = self.vf(vl_out)
        return v.squeeze(-1)

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))
        self.to(torch_utils.device)

def build_policy(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(env.observation_space.shape, **policy_kwargs)

    def policy_fn():
        ob_space = env.observation_space

        extra_tensors = {}

        # if normalize_observations and X.dtype == tf.float32:
            # encoded_x, rms = _normalize_clip_observation(X)
            # extra_tensors['rms'] = rms
        # else:
            # encoded_x = X

        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            _v_net = policy_network
        else:
            if _v_net == 'copy':
                _v_net = copy.deepcopy(policy_network)
            else:
                assert callable(_v_net)

        policy = PolicyWithValue(
            env=env,
            latent=policy_network,
            vf_latent=_v_net,
            estimate_q=estimate_q,
            **extra_tensors
        )
        policy.to(torch_utils.device)
        return policy

    return policy_fn

# def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
#     rms = RunningMeanStd(shape=x.shape[1:])  # mpi version
#     norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
#     return norm_x, rms
