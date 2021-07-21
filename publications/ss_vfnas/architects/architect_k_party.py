import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect_A(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.k = args.k
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def update_alpha(self, input_valid, U_B_val_list, target_valid):
        if U_B_val_list is not None:
            U_B_val_list = [torch.autograd.Variable(U_B_valid, requires_grad=True).cuda() for U_B_valid in U_B_val_list]
        self.optimizer.zero_grad()
        loss, _ = self.model._loss(input_valid, U_B_val_list, target_valid)
        U_B_gradients_list = None
        if U_B_val_list is not None:
            U_B_gradients_list = [torch.autograd.grad(loss, U_B_valid, retain_graph=True) for U_B_valid in U_B_val_list]
        loss.backward()
        self.optimizer.step()
        return U_B_gradients_list

    def update_weights(self, trn_X, U_B_train_list, target_train, weights_optim, grad_clip):
        if U_B_train_list is not None:
            U_B_train_list = [torch.autograd.Variable(U_B_train_list[i], requires_grad=True).cuda() for i in
                              range(0, len(U_B_train_list))]
        weights_optim.zero_grad()
        loss, logits = self.model._loss(trn_X, U_B_train_list, target_train)
        U_B_gradients_list = None
        if U_B_train_list is not None:
            U_B_gradients_list = [torch.autograd.grad(loss, U_B_train, retain_graph=True) for U_B_train in
                                  U_B_train_list]
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        weights_optim.step()
        return U_B_gradients_list, logits, loss


class Architect_B(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def update_alpha(self, U_B_val, U_B_gradients):
        model_B_alpha_gradients = torch.autograd.grad(U_B_val, self.model.arch_parameters(), grad_outputs=U_B_gradients)
        self.optimizer.zero_grad()
        for w, g in zip(self.model.arch_parameters(), model_B_alpha_gradients):
            w.grad = g.detach()
        self.optimizer.step()

    def update_weights(self, U_B_train, U_B_gradients, weights_optim, grad_clip):
        model_B_weight_gradients = torch.autograd.grad(U_B_train, self.model.parameters(), grad_outputs=U_B_gradients)
        weights_optim.zero_grad()
        for w, g in zip(self.model.parameters(), model_B_weight_gradients):
            w.grad = g.detach()
        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        weights_optim.step()
