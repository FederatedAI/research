import torch.nn as nn
from torchvision import models
import torch


class Manual_A(nn.Module):

    def __init__(self, num_classes, layers, u_dim=64, k=2):
        self.num_classes = num_classes
        super(Manual_A, self).__init__()
        if layers == 18:
            self.net = models.resnet18(pretrained=False, num_classes=u_dim)
        elif layers == 50:
            self.net = models.resnet50(pretrained=False, num_classes=u_dim)
        elif layers == 101:
            self.net = models.resnet101(pretrained=False, num_classes=u_dim)
        elif layers == 19:
            self.net = models.mobilenet_v2(pretrained=False, num_classes=u_dim)
        else:
            raise ValueError("Wrong number of layers for resnet")
        for i in range(1, num_classes + 1):
            setattr(self, "fc_" + str(i), nn.Linear(u_dim * k, 1))

    def forward(self, input, U_B):
        out = self.net(input)
        if U_B is not None:
            out = torch.cat([out] + [U for U in U_B], dim=1)
        logits = list()
        for i in range(1, self.num_classes + 1):
            classifier = getattr(self, "fc_" + str(i))
            logit = classifier(out)
            logits.append(logit)
        return logits

class Manual_B(nn.Module):

    def __init__(self, layers, u_dim=64):
        super(Manual_B, self).__init__()
        if layers == 18:
            self.net = models.resnet18(pretrained=False, num_classes=u_dim)
        elif layers == 50:
            self.net = models.resnet50(pretrained=False, num_classes=u_dim)
        elif layers == 101:
            self.net = models.resnet101(pretrained=False, num_classes=u_dim)
        elif layers == 19:
            self.net = models.mobilenet_v2(pretrained=False, num_classes=u_dim)
        else:
            raise ValueError("Wrong number of layers for resnet")

    def forward(self, input):
        out = self.net(input)
        return out
