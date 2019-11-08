import torch
import torch.nn as nn
from torch.nn import init
from collections.abc import Generator

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
        return [torch.from_numpy(arr) for arr in arrs]
    else:
        return torch.from_numpy(arrs)

def toNumpy(tensors):
    '''
    tensors: a list of torch tensor
    '''
    if isinstance(tensors, (list, tuple, Generator)):
        return [tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor for tensor in tensors]
    else:
        return tensors.detach().cpu().numpy() if torch.is_tensor(tensors) else tensors

