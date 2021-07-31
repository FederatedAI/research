import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        # nn.init.xavier_uniform(m.weight)
        # m.bias.data.fill_(0.01)
