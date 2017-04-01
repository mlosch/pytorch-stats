import torch
import numpy as np


class Normal(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, *args):
        x = args[0]
        mean = self.mean.expand(x.size())
        var = self.std.expand(x.size()) ** 2
        p = 1./torch.sqrt(2.0*np.pi*var) * torch.exp(- ((x-mean)**2) / (2.0 * var))
        return p
