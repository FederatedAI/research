import random

import numpy as np
import torch
import torch.nn as nn


class AvgMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.n = 0
        self.avg = 0.

    def update(self, val, n=1):
        assert n > 0
        self.val += val
        self.n += n
        self.avg = self.val / self.n

    def get(self):
        return self.avg


class BestMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.n = -1

    def update(self, val, n):
        assert n > self.n
        if val > self.val:
            self.val = val
            self.n = n

    def get(self):
        return self.val, self.n


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def add_gaussian_noise(tensor, mean, std):
    return torch.randn(tensor.size()) * std + mean


def set_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
