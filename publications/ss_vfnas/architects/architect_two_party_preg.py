import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import copy


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect_A(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.alpha_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                                lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                                weight_decay=args.arch_weight_decay)
        self.n_local_update = args.n_local_update
        self.mu = 0.001

    def _mark_model(self):
        self.copy_model = copy.deepcopy(self.model)

    def compute_weights_proximal_regularization(self):

        weights_loss = 0
        for w, initial_w in zip(self.model.parameters(), self.copy_model.parameters()):
            weights_loss += torch.norm(w - initial_w.detach())
        return weights_loss

    def compute_alpha_proximal_regularization(self):
        alpha_loss = 0
        for alpha, initial_alpha in zip(self.model.arch_parameters(), self.copy_model.arch_parameters()):
            alpha_loss += torch.norm(alpha - initial_alpha.detach())
        return alpha_loss

    def update_alpha(self, input_valid, U_B_valid, target_valid):
        self._mark_model()
        U_B_valid = torch.autograd.Variable(U_B_valid, requires_grad=True).cuda()
        for _ in range(self.n_local_update):
            self.alpha_optimizer.zero_grad()
            model_loss, _ = self.model._loss(input_valid, U_B_valid, target_valid)
            preg_loss = self.compute_alpha_proximal_regularization()
            loss = model_loss  + self.mu * preg_loss
            loss.backward(retain_graph=True)
            self.alpha_optimizer.step()
        U_B_gradients = torch.autograd.grad(model_loss, U_B_valid, retain_graph=True)
        return U_B_gradients

    def update_weights(self, input_train, U_B_train, target_train, weights_optim, grad_clip):
        self._mark_model()
        U_B_train = torch.autograd.Variable(U_B_train, requires_grad=True).cuda()
        for _ in range(self.n_local_update):
            weights_optim.zero_grad()
            model_loss, logits = self.model._loss(input_train, U_B_train, target_train)
            preg_loss = self.compute_weights_proximal_regularization()
            loss = model_loss + self.mu * preg_loss
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            weights_optim.step()
        U_B_gradients = torch.autograd.grad(model_loss, U_B_train, retain_graph=True)
        return U_B_gradients, logits, model_loss


class Architect_B(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.alpha_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                                lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                                weight_decay=args.arch_weight_decay)
        self.weight_optimizer = torch.optim.SGD(self.model.parameters(), args.learning_rate,
                                                momentum=args.momentum, weight_decay=args.weight_decay)
        self.n_local_update = args.n_local_update
        self.mu = 0.001

    def _mark_model(self):
        self.copy_model = copy.deepcopy(self.model)

    def compute_weights_proximal_regularization(self):

        weights_loss = 0
        for w, initial_w in zip(self.model.parameters(), self.copy_model.parameters()):
            weights_loss +=  torch.norm(w - initial_w.detach())
        return weights_loss

    def compute_alpha_proximal_regularization(self):
        alpha_loss = 0
        for alpha, initial_alpha in zip(self.model.arch_parameters(), self.copy_model.arch_parameters()):
            alpha_loss += torch.norm(alpha - initial_alpha.detach())
        return alpha_loss

    def update_alpha(self, U_B_val, U_B_gradients):
        self._mark_model()
        model_B_alpha_gradients = torch.autograd.grad(U_B_val, self.model.arch_parameters(),
                                                      grad_outputs=U_B_gradients, retain_graph=True)
        self.alpha_optimizer.zero_grad()
        for w, g1 in zip(self.model.arch_parameters(), model_B_alpha_gradients):
            w.grad = g1
        self.alpha_optimizer.step()

    def update_weights(self, U_B_train, U_B_gradients, weights_optim, grad_clip):
        self._mark_model()
        model_B_weight_gradients = torch.autograd.grad(U_B_train, self.model.parameters(),
                                                       grad_outputs=U_B_gradients, retain_graph=True)
        weights_optim.zero_grad()
        for w, g1 in zip(self.model.parameters(), model_B_weight_gradients):
            w.grad = g1.detach()
        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        weights_optim.step()
