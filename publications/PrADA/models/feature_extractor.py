import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import init_weights


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


activation_fn = nn.LeakyReLU()


# activation_fn = Mish()


class CensusRegionFeatureExtractorDense(nn.Module):
    def __init__(self, input_dims):
        super(CensusRegionFeatureExtractorDense, self).__init__()
        if len(input_dims) == 6:
            self.extractor = nn.Sequential(
                nn.Linear(in_features=input_dims[0], out_features=input_dims[1]),
                nn.BatchNorm1d(input_dims[1]),
                activation_fn,
                nn.Linear(in_features=input_dims[1], out_features=input_dims[2]),
                nn.BatchNorm1d(input_dims[2]),
                activation_fn,
                nn.Linear(in_features=input_dims[2], out_features=input_dims[3]),
                nn.BatchNorm1d(input_dims[3]),
                activation_fn,
                nn.Linear(in_features=input_dims[3], out_features=input_dims[4]),
                nn.BatchNorm1d(input_dims[4]),
                activation_fn,
                nn.Linear(in_features=input_dims[4], out_features=input_dims[5]),
                nn.BatchNorm1d(input_dims[5]),
                activation_fn
            )
        elif len(input_dims) == 5:
            self.extractor = nn.Sequential(
                nn.Linear(in_features=input_dims[0], out_features=input_dims[1]),
                nn.BatchNorm1d(input_dims[1]),
                activation_fn,
                nn.Linear(in_features=input_dims[1], out_features=input_dims[2]),
                nn.BatchNorm1d(input_dims[2]),
                activation_fn,
                nn.Linear(in_features=input_dims[2], out_features=input_dims[3]),
                nn.BatchNorm1d(input_dims[3]),
                activation_fn,
                nn.Linear(in_features=input_dims[3], out_features=input_dims[4]),
                nn.BatchNorm1d(input_dims[4]),
                activation_fn
            )
        elif len(input_dims) == 4:
            self.extractor = nn.Sequential(
                nn.Linear(in_features=input_dims[0], out_features=input_dims[1]),
                nn.BatchNorm1d(input_dims[1]),
                nn.Dropout(p=0.1),
                activation_fn,
                nn.Linear(in_features=input_dims[1], out_features=input_dims[2]),
                nn.BatchNorm1d(input_dims[2]),
                nn.Dropout(p=0.1),
                activation_fn,
                nn.Linear(in_features=input_dims[2], out_features=input_dims[3]),
                nn.BatchNorm1d(input_dims[3]),
                activation_fn
            )
        elif len(input_dims) == 3:
            self.extractor = nn.Sequential(
                nn.Linear(in_features=input_dims[0], out_features=input_dims[1]),
                nn.BatchNorm1d(input_dims[1]),
                activation_fn,
                nn.Linear(in_features=input_dims[1], out_features=input_dims[2]),
                nn.BatchNorm1d(input_dims[2]),
                activation_fn
            )
        elif len(input_dims) == 2:
            self.extractor = nn.Sequential(
                nn.Linear(in_features=input_dims[0], out_features=input_dims[1]),
                nn.BatchNorm1d(input_dims[1]),
                activation_fn
            )
        else:
            raise RuntimeError(f"Currently does not support input_dims of layers {input_dims}")

        # self.extractor.apply(init_weights)

    def forward(self, x):
        x = self.extractor(x)
        x = F.normalize(x, dim=1, p=2)
        return x


class CensusRegionFeatureExtractorV3(nn.Module):
    def __init__(self, input_dim):
        super(CensusRegionFeatureExtractorV3, self).__init__()
        self.extractor = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=60),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.extractor(x)
        print("x shape:", x.shape)
        # x = x.view(x.shape[0], -1)
        return x


class CensusExtractorV2(nn.Module):
    def __init__(self):
        super(CensusExtractorV2, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.shape[0], -1)
        return x


class MNISTExtractor(nn.Module):
    def __init__(self):
        super(MNISTExtractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.shape[0], -1)
        return x


class MNISTRegionExtractor(nn.Module):
    def __init__(self):
        super(MNISTRegionExtractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3),
            nn.BatchNorm2d(24),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.shape[0], -1)
        return x


class MNISTExpandInputExtractor(MNISTExtractor):

    def __init__(self):
        super(MNISTExpandInputExtractor, self).__init__()

    def forward(self, x):
        x = x.expand(x.data_to_split.shape[0], 3, 28, 28)
        return super().forward(x)


class MNISTRegionExpandInputExtractor(MNISTRegionExtractor):

    def __init__(self):
        super(MNISTRegionExpandInputExtractor, self).__init__()

    def forward(self, x):
        x = x.expand(x.data_to_split.shape[0], 3, 14, 14)
        return super().forward(x)
