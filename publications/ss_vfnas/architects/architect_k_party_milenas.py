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
        self.val_lambda = 1.0
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def update(self, train_alpha_gradients, train_weights_gradients, val_alpha_gradients, w_optimizer, grad_clip):
        for alpha, val_gradient, trn_gradient in zip(self.model.arch_parameters(), val_alpha_gradients,
                                                     train_alpha_gradients):
            alpha.grad = self.val_lambda * val_gradient.detach() + trn_gradient.detach()
        self.optimizer.step()
        w_optimizer.zero_grad()
        for w, trn_gradient in zip(self.model.parameters(), train_weights_gradients):
            if trn_gradient is not None:
                w.grad = trn_gradient.detach()
        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        w_optimizer.step()

    def compute_grad(self, X, U_B_list, target, need_weight_grad):
        if U_B_list is not None:
            U_B_list = [torch.autograd.Variable(U_B_list[i], requires_grad=True).cuda() for i in
                        range(0, len(U_B_list))]
        loss, logits = self.model._loss(X, U_B_list, target)
        U_B_gradients_list = None
        if U_B_list is not None:
            U_B_gradients_list = [torch.autograd.grad(loss, U_B, retain_graph=True) for U_B in
                                  U_B_list]
        alpha_gradients = torch.autograd.grad(loss, self.model.arch_parameters(), retain_graph=True)
        if need_weight_grad:
            weights_gradients = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True, allow_unused=True)
            return U_B_gradients_list, alpha_gradients, weights_gradients, logits, loss
        else:
            return U_B_gradients_list, alpha_gradients


class Architect_B(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.val_lambda = 1.0
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def update_alpha(self, U_B_train, U_B_train_gradients, U_B_val, U_B_valid_gradients):
        self.optimizer.zero_grad()
        alpha_train_gradients = torch.autograd.grad(U_B_train, self.model.arch_parameters(),
                                                    grad_outputs=U_B_train_gradients,
                                                    retain_graph=True)
        alpha_valid_gradients = torch.autograd.grad(U_B_val, self.model.arch_parameters(),
                                                    grad_outputs=U_B_valid_gradients,
                                                    retain_graph=True)
        for alpha, val_gradient, trn_gradient in zip(self.model.arch_parameters(), alpha_valid_gradients,
                                                     alpha_train_gradients):
            alpha.grad = self.val_lambda * val_gradient.detach() + trn_gradient.detach()
        self.optimizer.step()

    def update_weights(self, U_B_train, U_B_gradients, weights_optim, grad_clip):
        model_B_weight_gradients = torch.autograd.grad(U_B_train, self.model.parameters(), grad_outputs=U_B_gradients,
                                                       retain_graph=True)
        weights_optim.zero_grad()
        for w, g in zip(self.model.parameters(), model_B_weight_gradients):
            w.grad = g.detach()
        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        weights_optim.step()
