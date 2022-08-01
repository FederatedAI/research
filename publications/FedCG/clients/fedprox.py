import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import args, logger, device
from models import Extractor, Classifier
from utils import AvgMeter, set_seed, add_gaussian_noise


class Client():

    def __init__(self, client_id, trainset, valset, testset):
        set_seed(args.seed)
        self.id = client_id
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                      pin_memory=True)
        self.valloader = DataLoader(valset, batch_size=args.batch_size * 10, shuffle=False, num_workers=0,
                                    pin_memory=False)
        self.testloader = DataLoader(testset, batch_size=args.batch_size * 10, shuffle=False, num_workers=0,
                                     pin_memory=False)
        self.train_size = len(trainset)
        self.val_size = len(valset)
        self.test_size = len(testset)
        if args.add_noise:
            self.noise_std = args.noise_std * self.id / (args.n_clients - 1)
            logger.info("client:%2d, train_size:%4d, val_size:%4d, test_size:%4d, noise_std:%2.6f"
                        % (self.id, self.train_size, self.val_size, self.test_size, self.noise_std))
        else:
            logger.info("client:%2d, train_size:%4d, val_size:%4d, test_size:%4d"
                        % (self.id, self.train_size, self.val_size, self.test_size))
        self.init_net()

    def get_params(self, models):
        params = []
        for model in models:
            params.append({"params": self.net[model].parameters()})
        return params

    def frozen_net(self, models, frozen):
        for model in models:
            for param in self.net[model].parameters():
                param.requires_grad = not frozen
            if frozen:
                self.net[model].eval()
            else:
                self.net[model].train()

    def save_client(self):
        optim_dict = {
            "net": self.net.state_dict(),
            "EC_optimizer": self.EC_optimizer.state_dict()}
        torch.save(optim_dict, args.checkpoint_dir + "/client" + str(self.id) + ".pkl")

    def load_client(self):
        checkpoint = torch.load(args.checkpoint_dir + "/client" + str(self.id) + ".pkl")
        self.net.load_state_dict(checkpoint["net"])
        self.EC_optimizer.load_state_dict(checkpoint["EC_optimizer"])

    def init_net(self):
        set_seed(args.seed)
        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        self.net = nn.ModuleDict()

        self.net["extractor"] = Extractor()  # E
        self.net["classifier"] = Classifier()  # C
        self.frozen_net(["extractor", "classifier"], True)
        self.EC_optimizer = optim.Adam(self.get_params(["extractor", "classifier"]), lr=args.lr,
                                       weight_decay=args.weight_decay)

        self.net.to(device)

        self.BCE_criterion = nn.BCELoss().to(device)
        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        self.MSE_criterion = nn.MSELoss().to(device)
        self.COS_criterion = nn.CosineSimilarity().to(device)

    def local_test(self):
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.testloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item()

    def local_val(self):
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.valloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item()

    def local_train(self, current_round):
        set_seed(args.seed)
        logger.info("Training Client %2d's EC Network Start!" % self.id)
        EC_loss_meter = AvgMeter()
        prox_reg_meter = AvgMeter()
        global_weight_collector = copy.deepcopy(list(self.net.parameters()))

        for epoch in range(args.local_epoch):
            EC_loss_meter.reset()
            prox_reg_meter.reset()

            self.frozen_net(["extractor", "classifier"], False)

            for batch, (x, y) in enumerate(self.trainloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)

                self.EC_optimizer.zero_grad()

                E = self.net["extractor"](x)
                EC = self.net["classifier"](E)
                EC_loss = self.CE_criterion(EC, y)

                prox_reg = 0.0
                for param_index, param in enumerate(self.net.parameters()):
                    prox_reg += ((args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)

                (EC_loss + prox_reg).backward()
                self.EC_optimizer.step()
                EC_loss_meter.update(EC_loss.item())
                prox_reg_meter.update(prox_reg.item())

            self.frozen_net(["extractor", "classifier"], True)
            EC_loss = EC_loss_meter.get()
            prox_reg = prox_reg_meter.get()

            EC_acc = self.local_val()
            logger.info("Client:[%2d], Epoch:[%2d], EC_loss:%2.6f, prox_reg:%2.6f, EC_acc:%2.6f" % (
                self.id, epoch, EC_loss, prox_reg, EC_acc))
