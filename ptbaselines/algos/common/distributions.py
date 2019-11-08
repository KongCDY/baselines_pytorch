import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as distributions
import numpy as np
from ptbaselines.algos.common.torch_utils import init_weight

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def logp(self, x):
        return - self.neglogp(x)
    def get_shape(self):
        return self.flatparam().shape
    @property
    def shape(self):
        return self.get_shape()
    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])

class PdType(nn.Module):
    """
    Parametrized family of probability distributions
    """
    def __init__(self):
        super(PdType, self).__init__()
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def pdfromlatent(self, latent_vector, init_scale, init_bias):
        raise NotImplementedError
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    # def __eq__(self, other):
        # return (type(self) == type(other)) and (self.__dict__ == other.__dict__)

class CategoricalPdType(PdType):
    def __init__(self, in_dim, ncat, init_scale = 1.0, init_bias = 0.0):
        super(CategoricalPdType, self).__init__()
        self.in_dim = in_dim
        self.ncat = ncat
        self.fc = nn.Linear(in_dim, ncat)
        init_weight(self.fc, init_scale, init_bias)

    def pdclass(self):
        return CategoricalPd
    def pdfromlatent(self, latent_vector):
        pdparam = self.fc(latent_vector)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return torch.int32

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self._m = distributions.Categorical(F.softmax(logits, -1))
    def flatparam(self):
        return self.logits
    def mode(self):
        return torch.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return F.softmax(self.logits)
    def neglogp(self, x):
        return -self._m.log_prob(x)
    def kl(self, other):
        a0 = self.logits - self.logits.max(dim=-1, keepdim=True)[0]
        a1 = other.logits - other.logits.max(dim=-1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = ea0.sum(dim=-1, keepdim=True)
        z1 = ea1.sum(dim=-1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=-1)
    def entropy(self):
        a0 = self.logits - self.logits.max(dim=-1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = ea0.sum(dim=-1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (torch.log(z0) - a0), dim=-1)
    def sample(self):
        return self._m.sample()
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def make_pdtype(in_dim, ac_space, init_scale = 1.0, init_bias = 0.0):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(in_dim, ac_space.n, init_scale, init_bias)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError
