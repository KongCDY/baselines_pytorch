import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from ptbaselines.algos.common.models import get_network_builder
from ptbaselines.algos.common.torch_utils import init_weight

class Actor(nn.Module):
    def __init__(self, env, network = 'mlp', **network_kwargs):
        super(Actor, self).__init__()
        self.num_actions = env.action_space.shape[-1]
        self.base_net = get_network_builder(network)(env.observation_space.shape, **network_kwargs)
        self.fc = nn.Linear(self.base_net.out_dim, self.num_actions)
        # init
        init.uniform_(self.fc.weight.data, -3e-3, 3e-3)
        init.uniform_(self.fc.bias.data, -3e-3, 3e-3)
    
    def forward(self, obs):
        latent = self.base_net(obs)
        latent = latent.view(obs.size(0), self.base_net.out_dim)
        action = torch.tanh(self.fc(latent))
        return action
    
class Critic(nn.Module):
    def __init__(self, env, network = 'mlp', **network_kwargs):
        super(Critic, self).__init__()
        self.num_actions = env.action_space.shape[-1]
        input_shape = list(env.observation_space.shape)
        input_shape[-1] += self.num_actions
        self.base_net = get_network_builder(network)(input_shape, **network_kwargs)
        self.fc = nn.Linear(self.base_net.out_dim, 1)
        # init
        init.uniform_(self.fc.weight.data, -3e-3, 3e-3)
        init.uniform_(self.fc.bias.data, -3e-3, 3e-3)
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim = -1)
        latent = self.base_net(x)
        latent = latent.view(obs.size(0), self.base_net.out_dim)
        value = self.fc(latent)
        return value
