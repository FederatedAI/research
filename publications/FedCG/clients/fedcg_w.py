import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import args, logger, device
from models import Extractor, Classifier, Generator, Discriminator
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
            "EC_optimizer": self.EC_optimizer.state_dict(),
            "D_optimizer": self.D_optimizer.state_dict(),
            "G_optimizer": self.G_optimizer.state_dict(),
        }
        torch.save(optim_dict, args.checkpoint_dir + "/client" + str(self.id) + ".pkl")

    def load_client(self):
        checkpoint = torch.load(args.checkpoint_dir + "/client" + str(self.id) + ".pkl")
        self.net.load_state_dict(checkpoint["net"])
        self.EC_optimizer.load_state_dict(checkpoint["EC_optimizer"])
        self.D_optimizer.load_state_dict(checkpoint["D_optimizer"])
        self.G_optimizer.load_state_dict(checkpoint["G_optimizer"])

    def init_net(self):
        set_seed(args.seed)
        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        self.net = nn.ModuleDict()

        self.net["extractor"] = Extractor()  # E
        self.net["classifier"] = Classifier()  # C
        self.net["generator"] = Generator()  # D
        self.net["discriminator"] = Discriminator()  # G
        self.frozen_net(["extractor", "classifier", "generator", "discriminator"], True)
        self.EC_optimizer = optim.Adam(self.get_params(["extractor", "classifier"]), lr=args.lr,
                                       weight_decay=args.weight_decay)
        self.D_optimizer = optim.Adam(self.get_params(["discriminator"]), lr=args.lr, betas=(0.5, 0.999))
        self.G_optimizer = optim.Adam(self.get_params(["generator"]), lr=args.lr, betas=(0.5, 0.999))

        self.net.to(device)

        self.BCE_criterion = nn.BCELoss().to(device)
        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        self.MSE_criterion = nn.MSELoss().to(device)
        self.COS_criterion = nn.CosineSimilarity().to(device)

    def compute_gan_acc(self):
        correct, total = 0, 0
        with torch.no_grad():
            for batch in range(100):
                y = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)
                z = torch.randn(args.batch_size, args.noise_dim, 1, 1).to(device)
                feat = self.net["generator"](z, y)
                pred = self.net["classifier"](feat)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += args.batch_size
        return (correct / total).item()

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
        EG_distance_meter = AvgMeter()

        for epoch in range(args.local_epoch):
            EC_loss_meter.reset()
            EG_distance_meter.reset()

            self.frozen_net(["extractor", "classifier"], False)

            for batch, (x, y) in enumerate(self.trainloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                z = torch.randn(x.size(0), args.noise_dim, 1, 1).to(device)

                self.EC_optimizer.zero_grad()

                E = self.net["extractor"](x)
                EC = self.net["classifier"](E)
                EC_loss = self.CE_criterion(EC, y)

                G = self.net["generator"](z, y).detach()
                if args.distance == "mse":
                    EG_distance = self.MSE_criterion(E, G)
                elif args.distance == "cos":
                    EG_distance = 1 - self.COS_criterion(E, G).mean()
                elif args.distance == "none":
                    EG_distance = 0
                p = min(current_round / 50, 1.)
                gamma = 2 / (1 + np.exp(-10 * p)) - 1
                EG_distance = gamma * EG_distance

                (EC_loss + EG_distance).backward()
                self.EC_optimizer.step()
                EC_loss_meter.update(EC_loss.item())
                EG_distance_meter.update(EG_distance.item())

            self.frozen_net(["extractor", "classifier"], True)

            EC_loss = EC_loss_meter.get()
            EG_distance = EG_distance_meter.get()
            EC_acc = self.local_val()
            logger.info("Client:[%2d], Epoch:[%2d], EC_loss:%2.6f, EG_distance:%2.6f, EC_acc:%2.6f" % (
                self.id, epoch, EC_loss, EG_distance, EC_acc))

        logger.info("Training Client %2d's DG Network Start!" % self.id)
        ED_loss_meter = AvgMeter()
        GD_loss_meter = AvgMeter()
        G_loss_meter = AvgMeter()
        G_div_meter = AvgMeter()

        for epoch in range(args.gan_epoch):
            ED_loss_meter.reset()
            GD_loss_meter.reset()
            G_loss_meter.reset()
            G_div_meter.reset()

            self.frozen_net(["generator", "discriminator"], False)

            for batch, (x, y) in enumerate(self.trainloader):
                if x.size(0) % 2 == 1:
                    x = torch.cat([x, x[:1]], dim=0)
                    y = torch.cat([y, y[:1]], dim=0)
                split_size = int(x.size(0) / 2)

                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                z = torch.randn(x.size(0), args.noise_dim, 1, 1).to(device)
                ones = torch.ones((x.size(0), 1)).to(device)
                zeros = torch.zeros((x.size(0), 1)).to(device)

                # train discriminator
                def train_discriminator(x, y, z):
                    self.D_optimizer.zero_grad()

                    E = self.net["extractor"](x)
                    ED = self.net["discriminator"](E.detach(), y)
                    ED_loss = self.BCE_criterion(ED, ones)

                    G = self.net["generator"](z, y)
                    GD = self.net["discriminator"](G.detach(), y)
                    GD_loss = self.BCE_criterion(GD, zeros)

                    D_loss = ED_loss + GD_loss
                    D_loss.backward()
                    self.D_optimizer.step()
                    ED_loss_meter.update(ED_loss.item())
                    GD_loss_meter.update(GD_loss.item())

                # train generator with diversity
                def train_generator_with_diversity(y, z):
                    self.G_optimizer.zero_grad()

                    # gen loss
                    G = self.net["generator"](z, y)
                    GD = self.net["discriminator"](G, y)
                    G_loss = self.BCE_criterion(GD, ones)

                    # div loss
                    z1, z2 = torch.split(z, split_size, dim=0)
                    G1, G2 = torch.split(G, split_size, dim=0)
                    lz = torch.mean(torch.abs(G1 - G2)) / torch.mean(torch.abs(z1 - z2))
                    eps = 1 * 1e-5
                    G_div = 0.2 / (lz + eps)

                    (G_loss + G_div).backward()
                    self.G_optimizer.step()
                    G_loss_meter.update(G_loss.item())
                    G_div_meter.update(G_div.item())

                train_discriminator(x, y, z)
                train_generator_with_diversity(y, z)

            self.frozen_net(["generator", "discriminator"], True)

            ED_loss = ED_loss_meter.get()
            GD_loss = GD_loss_meter.get()
            G_loss = G_loss_meter.get()
            G_div = G_div_meter.get()
            GC_acc = self.compute_gan_acc()
            logger.info(
                "Client:[%2d], Epoch:[%2d], ED_loss:%2.6f, GD_loss:%2.6f, G_loss:%2.6f, G_div:%2.6f, GC_acc:%2.6f"
                % (self.id, epoch, ED_loss, GD_loss, G_loss, G_div, GC_acc))
