import torch
import numpy as np


class Normal(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, *args):
        x = args[0]
        try:
            # is mean a function?
            mean = self.mean(args)
        except TypeError as e:
            mean = self.mean

        try:
            # is std a function?
            std = self.std(args)
        except TypeError:
            std = self.std

        if mean.numel() != x.numel():
            mean = mean.expand(x.size())
        if std.numel() != x.numel():
            std = std.expand(x.size())
        var = std ** 2
        p = 1./torch.sqrt(2.0*np.pi*var) * torch.exp(- ((x-mean)**2) / (2.0 * var))
        return p

