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

class QNet(nn.Module):
    def __init__(self, env, network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
        super(QNet, self).__init__()
        self.dueling = dueling
        self.num_actions = env.action_space.n
        if isinstance(network, str):
            self.base_net = get_network_builder(network)(env.observation_space.shape, **network_kwargs)
        else:
            self.base_net = network
        action_layers = []
        out_dim = self.base_net.out_dim
        for hidden in hiddens:
            action_layers.append(nn.Linear(out_dim, hidden))
            if layer_norm:
                action_layers.append(nn.LayerNorm(hidden))
            action_layers.append(nn.ReLU())
            out_dim = hidden
        action_layers.append(nn.Linear(out_dim, self.num_actions))
        self.action_layers = nn.Sequential(*action_layers)
        # init
        for m in self.action_layers.modules():
            init_weight(m, init_scale = np.sqrt(2.0))

        if dueling:
            state_layers = []
            out_dim = self.base_net.out_dim
            for hidden in hiddens:
                state_layers.append(nn.Linear(out_dim, hidden))
                if layer_norm:
                    state_layers.append(nn.LayerNorm(hidden))
                state_layers.append(nn.ReLU())
                out_dim = hidden
            state_layers.append(nn.Linear(out_dim, 1))
            self.state_layers = nn.Sequential(*state_layers)
            # init
            for m in self.state_layers.modules():
                init_weight(m, init_scale = np.sqrt(2.0))

    def forward(self, obs):
        latent = self.base_net(obs)
        latent = latent.view(obs.size(0), self.base_net.out_dim)
        action_scores = self.action_layers(latent)
        if self.dueling:
            state_value = self.state_layers(latent)
            action_score_centered = action_scores - action_scores.mean(dim = 1, keepdim = True)
            return action_score_centered + state_value
        else:
            return action_scores
    
    def sample_actions(self, obs, eps, stochastic = True):
        """ Sample actions

        Parameters
        ----------
        obs: input observations
        Returns
        -------
        act: selected actions
        """
        q_values = self.forward(obs)
        deterministic_actions = torch.argmax(q_values, dim = 1)
        random_actions = torch.randint_like(deterministic_actions, 0, self.num_actions)
        choose_random = torch.rand_like(random_actions, dtype=torch.float32) < eps
        stochastic_actions = torch.where(choose_random, random_actions, deterministic_actions)

        if stochastic:
            return stochastic_actions
        else:
            return deterministic_actions

def default_param_noise_filter(m):
    # The remaining layers are likely conv or layer norm layers, which we do not wish to
    # perturb (in the former case because they only extract features, in the latter case because
    # we use them for normalization purposes). If you change your network, you will likely want
    # to re-consider which layers to perturb and which to keep untouched.
    return isinstance(m, nn.Linear)

class Model(object):
    def __init__(self, *, qnet, lr, grad_norm_clipping=None, gamma=1.0,
                double_q=True, param_noise=False, param_noise_filter_func=None):

        self.grad_norm_clipping = grad_norm_clipping
        self.gamma = gamma
        self.qnet = qnet
        self.target_qnet = copy.deepcopy(qnet)

        self.qnet.to(torch_utils.device)
        self.target_qnet.to(torch_utils.device)
        for param in self.target_qnet.parameters():
            param.requires_grad = False
        self.double_q = double_q
        self.param_noise = param_noise
        self.param_noise_filter_func = param_noise_filter_func

        self.optimizer = optim.Adam(self.qnet.parameters(), lr = lr)

        if param_noise:
            if self.param_noise_filter_func is None:
                self.param_noise_filter_func = default_param_noise_filter
            self.param_noise_scale = 0.01
            self.adaptive_qnet = copy.deepcopy(qnet)
            self.pertubed_qnet = copy.deepcopy(qnet)
            self.adaptive_qnet.to(torch_utils.device)
            self.pertubed_qnet.to(torch_utils.device)
            for param in self.adaptive_qnet.parameters():
                param.requires_grad = False
            for param in self.pertubed_qnet.parameters():
                param.requires_grad = False
 
    def actions(self, obs, eps = 0, stochastic = True):
        return self.qnet.sample_actions(obs, eps, stochastic)
    
    def actions_with_param_noise(self, obs, eps = 0, reset = False, stochastic = True):
        if reset:
            self.perturb_params(self.qnet, self.pertubed_qnet)
        return self.pertubed_qnet.sample_actions(obs, eps, stochastic)

    def perturb_params(self, src_net, dst_net):
        params = parameters_to_vector(src_net.parameters())
        vector_to_parameters(params, dst_net.parameters())
        for m in dst_net.modules():
            if self.param_noise_filter_func(m):
                for param in m.parameters():
                    param.data += torch.randn_like(param.data) * self.param_noise_scale

    def update_noise_scale(self, obs, param_noise_threshold = 0.05):
        # perturb adaptive qnet
        self.perturb_params(self.qnet, self.adaptive_qnet)
        with torch.no_grad():
            q_values = self.qnet(obs)
            q_values_adaptive = self.adaptive_qnet(obs)
        kl = torch.sum(F.softmax(q_values, -1) * (torch.log(F.softmax(q_values, -1)) - torch.log(F.softmax(q_values_adaptive, -1))), dim = -1)
        mean_kl = kl.mean()
        if mean_kl < param_noise_threshold:
            self.param_noise_scale *= 1.01
        else:
            self.param_noise_scale /= 1.01
    
    def update_target(self):
        params = parameters_to_vector(self.qnet.parameters())
        vector_to_parameters(params, self.target_qnet.parameters())
    
    def save(self, save_path):
        torch.save(self.qnet.state_dict(), save_path)

    def load(self, load_path):
        self.qnet.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))
        self.target_qnet = copy.deepcopy(self.qnet)
        self.qnet.to(torch_utils.device)
        self.target_qnet.to(torch_utils.device)
    
    def train(self, obs_t, act_t, rew_t, obs_pt1, done_mask, importance_weights_ph):
        # q network evaluation
        q_t = self.qnet(obs_t)

        # target q network evalution
        with torch.no_grad():
            q_tp1 = self.target_qnet(obs_pt1)

        # q scores for actions which we know were selected in the given state.
        q_t_selected = torch.gather(q_t, 1, act_t.unsqueeze(1)).squeeze()

        # compute estimate of best possible value starting from state at t + 1
        if self.double_q:
            q_tp1_using_online_net = self.qnet(obs_pt1).detach()
            q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1, keepdim=True)
            q_tp1_best = torch.gather(q_tp1, 1, q_tp1_best_using_online_net).squeeze()
        else:
            q_tp1_best = torch.max(q_tp1, 1)[0]
        q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = (rew_t + self.gamma * q_tp1_best_masked).detach()

        # compute the error (potentially clipped)
        td_error = q_t_selected - q_t_selected_target
        errors = torch_utils.huber_loss(td_error)
        weighted_error = torch.mean(importance_weights_ph * errors)

        # compute optimization op (potentially with gradient clipping)
        self.optimizer.zero_grad()
        weighted_error.backward()
        # average_gradients(self.optimizer.param_groups)
        if self.grad_norm_clipping is not None:
            clip_grad_norm_(self.qnet.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        td_error, q_values = torch_utils.toNumpy((td_error, q_t))
        return td_error, {'q_values': q_values}
