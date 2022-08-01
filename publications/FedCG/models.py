import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision.models import resnet18

from config import args
from utils import weights_init

if args.model == "lenet5":

    class Extractor(nn.Module):

        def __init__(self):
            super(Extractor, self).__init__()
            self.extractor = nn.Sequential(
                nn.Conv2d(args.image_channel, 6, 5),
                nn.AvgPool2d(2, 2),
                nn.Sigmoid(),
                nn.Conv2d(6, 16, 5),
                nn.AvgPool2d(2, 2),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.extractor(x)
            return x


    class Classifier(nn.Module):

        def __init__(self, class_num=10):
            super(Classifier, self).__init__()
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.Sigmoid(),
                nn.Linear(120, 84),
                nn.Sigmoid(),
                nn.Linear(84, class_num),
            )

        def forward(self, x):
            x = self.classifier(x)
            return x


    class Generator(nn.Module):

        def __init__(self):
            super(Generator, self).__init__()
            self.embedding = nn.Embedding(args.num_classes, args.num_classes)
            self.generator = nn.Sequential(
                nn.ConvTranspose2d(args.noise_dim + args.num_classes, 512, 2, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(512, 256, 2, 1, 0, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 128, 2, 1, 0, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, args.feature_num * 1, 2, 1, 0, bias=False),
                nn.Sigmoid(),
            )
            self.apply(weights_init)

        def forward(self, z, y):
            y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
            zy = torch.cat([z, y], 1)
            return self.generator(zy)


    class Discriminator(nn.Module):

        def __init__(self):
            super(Discriminator, self).__init__()
            self.embedding = nn.Embedding(args.num_classes, args.num_classes)
            self.discriminator = nn.Sequential(
                spectral_norm(nn.Conv2d(args.feature_num + args.num_classes, 128, 2, 1, 0, bias=False)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(128, 256, 2, 1, 0, bias=False)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(256, 512, 2, 1, 0, bias=False)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(512, 1, 2, 1, 0, bias=False)),
                nn.Sigmoid(),
            )
            self.apply(weights_init)

        def forward(self, f, y):
            y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
            y = y.expand(y.size(0), args.num_classes, args.feature_size, args.feature_size)
            fy = torch.cat([f, y], 1)
            return self.discriminator(fy).squeeze(-1).squeeze(-1)


elif args.model == "alexnet":

    class Extractor(nn.Module):

        def __init__(self):
            super(Extractor, self).__init__()
            self.extractor = nn.Sequential(
                nn.Conv2d(args.image_channel, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )

        def forward(self, x):
            x = self.extractor(x)
            return x


    class Classifier(nn.Module):

        def __init__(self, num_classes=10):
            super(Classifier, self).__init__()
            self.classifier = nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.AdaptiveAvgPool2d((6, 6)),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.classifier(x)
            return x


    class Generator(nn.Module):

        def __init__(self):
            super(Generator, self).__init__()
            self.embedding = nn.Embedding(args.num_classes, args.num_classes)
            self.generator = nn.Sequential(
                nn.ConvTranspose2d(args.noise_dim + args.num_classes, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, args.feature_num * 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            )
            self.apply(weights_init)

        def forward(self, z, y):
            y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
            zy = torch.cat([z, y], 1)
            return self.generator(zy)


    class Discriminator(nn.Module):

        def __init__(self):
            super(Discriminator, self).__init__()
            self.embedding = nn.Embedding(args.num_classes, args.num_classes)
            self.discriminator = nn.Sequential(
                spectral_norm(nn.Conv2d(args.feature_num + args.num_classes, 128, 4, 1, 0, bias=False)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(128, 256, 4, 1, 0, bias=False)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(256, 512, 4, 1, 0, bias=False)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
                nn.Sigmoid(),
            )
            self.apply(weights_init)

        def forward(self, f, y):
            y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
            y = y.expand(y.size(0), args.num_classes, args.feature_size, args.feature_size)
            fy = torch.cat([f, y], 1)
            return self.discriminator(fy).squeeze(-1).squeeze(-1)



elif args.model == "resnet18":

    class Extractor(nn.Module):

        def __init__(self):
            super(Extractor, self).__init__()
            model = resnet18(pretrained=False)
            self.conv1 = model.conv1
            self.bn1 = model.bn1
            self.relu = model.relu
            self.maxpool = model.maxpool
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)  # [64, 64, 56, 56]
            x = self.layer1(x)  # [64, 64, 56, 56]
            x = self.layer2(x)  # [64, 128, 28, 28]
            x = self.sigmoid(x)
            return x


    class Classifier(nn.Module):

        def __init__(self):
            super(Classifier, self).__init__()
            model = resnet18(pretrained=False, num_classes=args.num_classes)
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.avgpool = model.avgpool
            self.fc = model.fc

        def forward(self, x):
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x


    class Generator(nn.Module):

        def __init__(self):
            super(Generator, self).__init__()
            self.embedding = nn.Embedding(args.num_classes, args.num_classes)
            self.generator = nn.Sequential(
                nn.ConvTranspose2d(args.noise_dim + args.num_classes, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, args.feature_num * 1, 4, 2, 1, bias=False),
                nn.Sigmoid(),
            )
            self.apply(weights_init)

        def forward(self, z, y):
            y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
            zy = torch.cat([z, y], 1)
            return self.generator(zy)


    class Discriminator(nn.Module):

        def __init__(self):
            super(Discriminator, self).__init__()
            self.embedding = nn.Embedding(args.num_classes, args.num_classes)
            self.discriminator = nn.Sequential(
                spectral_norm(nn.Conv2d(args.feature_num + args.num_classes, 128, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(256, 512, 4, 1, 0, bias=False)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
                nn.Sigmoid(),
            )
            self.apply(weights_init)

        def forward(self, f, y):
            y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
            y = y.expand(y.size(0), args.num_classes, args.feature_size, args.feature_size)
            fy = torch.cat([f, y], 1)
            return self.discriminator(fy).squeeze(-1).squeeze(-1)

if __name__ == "__main__":
    x = torch.rand(64, 3, 224, 224)
    y = torch.randint(0, 10, (64,))
    z = torch.rand(64, 100, 1, 1)

    extractor = Extractor()
    classifier = Classifier()
    generator = Generator()
    discriminator = Discriminator()

    E = extractor(x)
    print("E", E.shape)
    G = generator(z, y)
    print("G", G.shape)
    ED = discriminator(E, y)
    print("ED", ED.shape)
    GD = discriminator(G, y)
    print("GD", GD.shape)
    EC = classifier(E)
    print("EC", EC.shape)
