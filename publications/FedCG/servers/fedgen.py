import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import args, logger, device
from models import Classifier, Generator
from utils import AvgMeter, set_seed


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

        self.global_net["generator"] = Generator()
        self.global_net["classifier"] = Classifier()
        self.frozen_net(["generator", "classifier"], True)
        self.G_optimizer = optim.Adam(self.get_params(["generator"]), lr=args.lr, weight_decay=args.weight_decay)

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

    def global_train(self):
        set_seed(args.seed)

        logger.info("Training Server's Network Start!")
        G_loss_meter = AvgMeter()
        G_div_meter = AvgMeter()

        self.frozen_net(["generator"], False)

        for epoch in range(args.global_epoch):
            G_loss_meter.reset()
            G_div_meter.reset()

            for batch in range(args.global_iter_per_epoch):
                y = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)
                z = torch.randn(args.batch_size, args.noise_dim, 1, 1).to(device)

                self.G_optimizer.zero_grad()

                # gen loss
                pred = 0
                G = self.global_net["generator"](z, y)
                for i, client in enumerate(self.clients):
                    GC = client.net["classifier"](G)
                    pred += self.weights[i] * GC
                G_loss = self.CE_criterion(pred, y)

                # div loss
                split_size = int(args.batch_size / 2)
                z1, z2 = torch.split(z, split_size, dim=0)
                G1, G2 = torch.split(G, split_size, dim=0)
                lz = torch.mean(torch.abs(G1 - G2)) / torch.mean(torch.abs(z1 - z2))
                eps = 1 * 1e-5
                G_div = 1 / (lz + eps)

                (G_loss + G_div).backward()
                self.G_optimizer.step()
                G_loss_meter.update(G_loss.item())
                G_div_meter.update(G_div.item())

            G_loss = G_loss_meter.get()
            G_div_meter.update(G_div.item())
            logger.info("Server Epoch:[%2d], G_loss:%2.6f, G_div:%2.6f" % (epoch, G_loss, G_div))

        self.frozen_net(["generator"], True)
