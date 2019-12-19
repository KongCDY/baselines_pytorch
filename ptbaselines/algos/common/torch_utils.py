import torch
import torch.nn as nn
from torch.nn import init
from collections.abc import Generator
import numpy as np

device = 'cpu'

def init_weight(m, init_scale = 1.0, init_bias = 0.0):
    if isinstance(m, nn.Linear):
        init.orthogonal_(m.weight.data, init_scale)
        init.constant_(m.bias.data, init_bias)
    elif isinstance(m, nn.Conv2d):
        init.orthogonal_(m.weight.data, init_scale)
        init.constant_(m.bias.data, 0.0)

def toTensor(arrs):
    '''
    arrs: a list of numpy array
    '''
    if isinstance(arrs, (list, tuple, Generator)):
        return [torch.from_numpy(arr).to(device) for arr in arrs]
    else:
        return torch.from_numpy(arrs).to(device)

def toNumpy(tensors):
    '''
    tensors: a list of torch tensor
    '''
    if isinstance(tensors, (list, tuple, Generator)):
        return [tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor for tensor in tensors]
    else:
        return tensors.detach().cpu().numpy() if torch.is_tensor(tensors) else tensors

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

# learning rate schedules
def constant(p):
    return 1

def linear(p):
    return 1-p

def middle_drop(p):
    eps = 0.75
    if 1-p<eps:
        return eps*0.1
    return 1-p

def double_linear_con(p):
    p *= 2
    eps = 0.125
    if 1-p<eps:
        return eps
    return 1-p

def double_middle_drop(p):
    eps1 = 0.75
    eps2 = 0.25
    if 1-p<eps1:
        if 1-p<eps2:
            return eps2*0.5
        return eps1*0.1
    return 1-p

schedules = {
    'linear':linear,
    'constant':constant,
    'double_linear_con': double_linear_con,
    'middle_drop': middle_drop,
    'double_middle_drop': double_middle_drop
}

class Scheduler(object):

    def __init__(self, v, nvalues, schedule):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues
        self.schedule = schedules[schedule]

    def value(self):
        current_value = self.v*self.schedule(self.n/self.nvalues)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        return self.v*self.schedule(steps/self.nvalues)