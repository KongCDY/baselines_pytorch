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
        x = x.transpose(2, 3).transpose(1, 2)
        conv_out = self.convs(x / 255.)
        return self.fc(conv_out.view(conv_out.size(0), -1))

@register("cnn")
def cnn(in_shape, **conv_kwargs):
    def network_fn():
        return nature_cnn(in_shape, **conv_kwargs)
    return network_fn()

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
