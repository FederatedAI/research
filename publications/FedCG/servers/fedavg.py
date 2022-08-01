import numpy as np
import torch
import torch.nn as nn

from config import args, logger, device
from models import Classifier, Extractor
from utils import set_seed


class Server():

    def __init__(self, clients):
        self.clients = clients

        self.weights = []
        for client in clients:
            self.weights.append(client.train_size)
        self.weights = np.array(self.weights) / np.sum(self.weights)
        logger.info("client weights: %s" % str(self.weights.tolist()))

        self.init_net()

    def get_params(self, models):
        params = []
        for model in models:
            params.append({"params": self.global_net[model].parameters()})
        return params

    def frozen_net(self, models, frozen):
        for model in models:
            for param in self.global_net[model].parameters():
                param.requires_grad = not frozen
            if frozen:
                self.global_net[model].eval()
            else:
                self.global_net[model].train()

    def init_net(self):
        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        set_seed(args.seed)
        self.global_net = nn.ModuleDict()

        self.global_net["extractor"] = Extractor()
        self.global_net["classifier"] = Classifier()
        self.frozen_net(["extractor", "classifier"], True)

        self.global_net.to(device)

        self.KL_criterion = nn.KLDivLoss(reduction="batchmean").to(device)
        self.CE_criterion = nn.CrossEntropyLoss().to(device)

    def load_client(self):
        for i in range(len(self.clients)):
            checkpoint = torch.load(args.checkpoint_dir + "/client" + str(i) + ".pkl")
            self.clients[i].net.load_state_dict(checkpoint["net"])
            self.clients[i].EC_optimizer.load_state_dict(checkpoint["EC_optimizer"])

    def save_client(self):
        for i in range(len(self.clients)):
            optim_dict = {
                "net": self.clients[i].net.state_dict(),
                "EC_optimizer": self.clients[i].EC_optimizer.state_dict(),
            }
            torch.save(optim_dict, args.checkpoint_dir + "/client" + str(i) + ".pkl")

    def receive(self, models):
        for model in models:
            avg_param = {}
            params = []
            for client in self.clients:
                params.append(client.net[model].state_dict())

            for key in params[0].keys():
                avg_param[key] = params[0][key] * self.weights[0]
                for idx in range(1, len(self.clients)):
                    avg_param[key] += params[idx][key] * self.weights[idx]
            self.global_net[model].load_state_dict(avg_param)

    def send(self, models):
        for model in models:
            global_param = self.global_net[model].state_dict()
            for client in self.clients:
                client.net[model].load_state_dict(global_param)
