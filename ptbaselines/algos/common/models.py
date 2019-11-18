import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np
from ptbaselines.algos.common.torch_utils import init_weight

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk

def oSize(size, ksize, stride = 1, pad = 0):
    return (size + 2*pad - ksize) // stride + 1

class nature_cnn(nn.Module):
    def __init__(self, input_size, **conv_kwargs):
        super(nature_cnn, self).__init__()
        """
        CNN from Nature paper.

        Args:
        input_size: (H, W, C)
        """
        in_dim = input_size[-1]
        self.convs = nn.Sequential(
                nn.Conv2d(in_dim, 32, kernel_size = 8, stride = 4),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
                nn.ReLU(),

                nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
                nn.ReLU(),
                )
        out_h = oSize(oSize(oSize(input_size[-3], 8, 4), 4, 2), 3)
        out_w = oSize(oSize(oSize(input_size[-2], 8, 4), 4, 2), 3)
        self.fc = nn.Sequential(
                nn.Linear(out_h*out_w*64, 512),
                nn.ReLU(),
                )
        self.out_dim = 512

        # init
        for m in self.modules():
            init_weight(m, **conv_kwargs)
        init_weight(self.convs[0], init_scale = np.sqrt(2.0))
        init_weight(self.fc[0], init_scale = np.sqrt(2.0))

    def forward(self, x):
        '''
        Parameters:
        ---------
        x: (batch, h, w, c)
        '''
        x = x.transpose(2, 3).transpose(1, 2)  # convert to NCHW
        conv_out = self.convs(x / 255.)
        return self.fc(conv_out.view(conv_out.size(0), -1))

class nature_mlp(nn.Module):
    def __init__(self, input_size, num_layers=2, num_hidden=64, activation=nn.Tanh, layer_norm=False):
        super(nature_mlp, self).__init__()
        """
        Stack of fully-connected layers to be used in a policy / q-function approximator
        Parameters:
        ----------
        input_size: (int, )             input size, use env.observation_space.shape
        num_layers: int                 number of fully-connected layers (default: 2)
        num_hidden: int                 size of fully-connected layers (default: 64)
        activation:                     activation function (default: nn.Tanh)

        Returns:
        -------
        fully connected network model
        """
        in_dim = input_size[0]
        self.out_dim = num_hidden
        layers = []

        layers.append(nn.Linear(in_dim, num_hidden))
        if layer_norm:
            layers.append(nn.LayerNorm(num_hidden))
        layers.append(activation())

        for i in range(1, num_layers):
            layers.append(nn.Linear(num_hidden, num_hidden))
            if layer_norm:
                layers.append(nn.LayerNorm(num_hidden))
            layers.append(activation())
        self.layers = nn.Sequential(*layers)

        # init
        for m in self.modules():
            init_weight(m, init_scale = np.sqrt(2.0))

    def forward(self, x):
        return self.layers(x)

@register("cnn")
def cnn(in_shape, **conv_kwargs):
    def network_fn():
        return nature_cnn(in_shape, **conv_kwargs)
    return network_fn()

@register("mlp")
def mlp(input_size, **mlp_kwargs):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator
    Parameters:
    ----------
    input_size: (int, )             input size, use env.observation_space.shape
    num_layers: int                 number of fully-connected layers (default: 2)
    num_hidden: int                 size of fully-connected layers (default: 64)
    activation:                     activation function (default: nn.Tanh)
    Returns:
    -------
    function that builds fully connected network
    """
    return nature_mlp(input_size, **mlp_kwargs)

def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:
    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn
    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
