import time
import functools
from ptbaselines import logger
from collections import deque

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from ptbaselines.common import set_global_seeds, explained_variance
from ptbaselines.algos.common.policies import build_policy

from ptbaselines.algos.a2c.runner import Runner
from ptbaselines.algos.common import torch_utils
from ptbaselines.algos.common.torch_utils import safemean

class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model
        train():
        - Make the training part (feedforward and retropropagation of gradients)
        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        nenvs = env.num_envs
        nbatch = nenvs*nsteps

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.train_model = policy()
        self.step_model = self.train_model  # difference for recurrent net
        self.optimizer = optim.RMSprop(self.train_model.parameters(), lr = 0.001, alpha = alpha, eps = epsilon)

        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state

        self.save = self.train_model.save
        self.load = self.train_model.load

        self.lr = torch_utils.Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

    def train(self, obs, states, rewards, masks, actions, values):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # rewards = R + yV(s')
        advs = rewards - values
        for step in range(len(obs)):
            cur_lr = self.lr.value()

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        obs = obs.float()
        neglogpac, pd = self.train_model.neglogp(obs, actions)

        # L = A(s,a) * -logpi(a|s)
        policy_loss = torch.mean(advs* neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        policy_entropy = torch.mean(pd.entropy())

        # Value loss
        vpred = self.train_model.value(obs)
        value_loss = F.mse_loss(vpred, rewards)

        loss = policy_loss - policy_entropy * self.ent_coef + value_loss * self.vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        self.optimizer.zero_grad()
        loss.backward()
        # average_gradients(self.optimizer.param_groups)
        if self.max_grad_norm is not None:
            clip_grad_norm_(self.train_model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        outputs = torch_utils.toNumpy((policy_loss, value_loss, policy_entropy))
        return outputs

def learn(
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.
    Parameters:
    -----------
    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies
    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)
    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)
    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)
    total_timesteps:    int, total number of timesteps to train on (default: 80M)
    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)
    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)
    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)
    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)
    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output
    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)
    alpha:              float, RMSProp decay parameter (default: 0.99)
    gamma:              float, reward discounting parameter (default: 0.99)
    log_interval:       int, specifies how frequently the logs are printed out (default: 100)
    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.
    '''



    set_global_seeds(seed)

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)

    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # Start total timer
    tstart = time.time()

    for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values, epinfos = runner.run()
        obs, rewards, masks, actions, values = torch_utils.toTensor((obs, rewards, masks, actions, values))
        epinfobuf.extend(epinfos)

        model_outputs = model.train(obs, states, rewards, masks, actions, values)
        policy_loss, value_loss, policy_entropy = torch_utils.toNumpy(model_outputs)
        nseconds = time.time()-tstart

        # Calculate the fps (frame per second)
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(*torch_utils.toNumpy((values, rewards)))
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()

            # plot using visdom
            timesteps = update*nbatch
            logger.vizkv('eprewmean', timesteps, safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.vizkv('eplenmean', timesteps, safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.vizkv('policy_loss', timesteps, policy_loss)
            logger.vizkv('value_loss', timesteps, value_loss)
            logger.vizkv('policy_entropy', timesteps, policy_entropy)

    return model