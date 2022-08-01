import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import args, logger, device
from models import Classifier, Generator
from utils import AvgMeter, add_gaussian_noise, set_seed


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
        self.GC_optimizer = optim.Adam(self.get_params(["generator", "classifier"]), lr=args.lr)

        self.global_net.to(device)

        self.KL_criterion = nn.KLDivLoss(reduction="batchmean").to(device)
        self.CE_criterion = nn.CrossEntropyLoss().to(device)

    def load_client(self):
        for i in range(len(self.clients)):
            checkpoint = torch.load(args.checkpoint_dir + "/client" + str(i) + ".pkl")
            self.clients[i].net.load_state_dict(checkpoint["net"])
            self.clients[i].EC_optimizer.load_state_dict(checkpoint["EC_optimizer"])
            self.clients[i].D_optimizer.load_state_dict(checkpoint["D_optimizer"])
            self.clients[i].G_optimizer.load_state_dict(checkpoint["G_optimizer"])

    def save_client(self):
        for i in range(len(self.clients)):
            optim_dict = {
                "net": self.clients[i].net.state_dict(),
                "EC_optimizer": self.clients[i].EC_optimizer.state_dict(),
                "D_optimizer": self.clients[i].D_optimizer.state_dict(),
                "G_optimizer": self.clients[i].G_optimizer.state_dict()
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

    def compute_aggregate_acc(self):
        correct, total = 0, 0
        with torch.no_grad():
            for client in self.clients:
                for x, y in client.valloader:
                    if args.add_noise:
                        x = add_gaussian_noise(x, mean=0., std=client.noise_std)
                    x = x.to(device)
                    y = y.to(device)
                    feat = client.net["extractor"](x)
                    pred = self.global_net["classifier"](feat)
                    correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                    total += x.size(0)
        return (correct / total).item()

    def global_train(self):
        set_seed(args.seed)

        logger.info("Training Server's Network Start!")
        distill_loss_meter = AvgMeter()

        after_average_acc = self.compute_aggregate_acc()
        logger.info("after average client acc:%2.6f" % after_average_acc)

        self.frozen_net(["generator", "classifier"], False)

        for epoch in range(args.global_epoch):
            distill_loss_meter.reset()

            for batch in range(args.global_iter_per_epoch):
                y = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)
                z = torch.randn(args.batch_size, args.noise_dim, 1, 1).to(device)

                self.GC_optimizer.zero_grad()

                global_feat = self.global_net["generator"](z, y)
                global_pred = self.global_net["classifier"](global_feat)
                q = torch.log_softmax(global_pred, -1)
                p = 0
                for i, client in enumerate(self.clients):
                    local_feat = client.net["generator"](z, y)
                    local_pred = client.net["classifier"](local_feat)
                    p += self.weights[i] * local_pred
                p = torch.softmax(p, -1).detach()
                distill_loss = self.KL_criterion(q, p)

                distill_loss.backward()
                self.GC_optimizer.step()
                distill_loss_meter.update(distill_loss.item())

            distill_loss = distill_loss_meter.get()
            logger.info("Server Epoch:[%2d], distill_loss:%2.6f" % (epoch, distill_loss))

        self.frozen_net(["generator", "classifier"], True)

        after_distill_acc = self.compute_aggregate_acc()
        logger.info("after distill client acc:%2.6f" % after_distill_acc)
