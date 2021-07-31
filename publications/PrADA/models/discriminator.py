import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models.model_utils import init_weights


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

    # @staticmethod
    # def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
    #     ctx.coeff = coeff
    #     output = input * 1.0
    #     return output
    #
    # @staticmethod
    # def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
    #     return grad_output.neg() * ctx.coeff, None


class RegionDiscriminator(nn.Module):

    def __init__(self, input_dim):
        super(RegionDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=40),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=2)
        )

        # self.discriminator.apply(init_weights)

    def apply_discriminator(self, x):
        return self.discriminator(x)

    def forward(self, input, alpha):
        reversed_input = GradientReverseFunction.apply(input, alpha)
        x = self.apply_discriminator(reversed_input)
        return F.softmax(x, dim=1)


activation_fn = nn.LeakyReLU()


class GlobalDiscriminator(nn.Module):

    def __init__(self, input_dim):
        super(GlobalDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_features=input_dim, out_features=12)),
            nn.BatchNorm1d(12),
            activation_fn,
            nn.Linear(in_features=12, out_features=6),
            nn.BatchNorm1d(6),
            activation_fn,
            nn.Linear(in_features=6, out_features=2),
        )

        # self.discriminator.apply(init_weights)

    def apply_discriminator(self, x):
        return self.discriminator(x)

    def forward(self, input, alpha):
        reversed_input = GradientReverseFunction.apply(input, alpha)
        x = self.apply_discriminator(reversed_input)
        return x


class CensusRegionDiscriminator(nn.Module):

    def __init__(self, input_dim):
        super(CensusRegionDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_features=input_dim, out_features=24)),
            nn.BatchNorm1d(24),
            activation_fn,
            nn.Linear(in_features=24, out_features=10),
            nn.BatchNorm1d(10),
            activation_fn,
            nn.Linear(in_features=10, out_features=2),
        )

    def apply_discriminator(self, x):
        return self.discriminator(x)

    def forward(self, input, alpha):
        reversed_input = GradientReverseFunction.apply(input, alpha)
        x = self.apply_discriminator(reversed_input)
        return x


class LendingRegionDiscriminator(nn.Module):

    def __init__(self, input_dim, hidden_dim=50):
        super(LendingRegionDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            activation_fn,
            nn.Linear(in_features=hidden_dim, out_features=10),
            # nn.BatchNorm1d(10),
            activation_fn,
            nn.Linear(in_features=10, out_features=2)
        )

    def apply_discriminator(self, x):
        return self.discriminator(x)

    def forward(self, input, alpha):
        reversed_input = GradientReverseFunction.apply(input, alpha)
        x = self.apply_discriminator(reversed_input)
        return x
        # return F.softmax(x, dim=1)


class CellNoFeatureGroupDiscriminator(nn.Module):

    def __init__(self, input_dim):
        super(CellNoFeatureGroupDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=200),
            nn.BatchNorm1d(200),
            activation_fn,
            nn.Linear(in_features=200, out_features=60),
            nn.BatchNorm1d(60),
            activation_fn,
            nn.Linear(in_features=60, out_features=10),
            nn.BatchNorm1d(10),
            activation_fn,
            nn.Linear(in_features=10, out_features=2)
        )

    def apply_discriminator(self, x):
        return self.discriminator(x)

    def forward(self, input, alpha):
        reversed_input = GradientReverseFunction.apply(input, alpha)
        x = self.apply_discriminator(reversed_input)
        return x


class IncomeDegreeDiscriminator(nn.Module):

    def __init__(self, input_dim):
        super(IncomeDegreeDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=36),
            nn.BatchNorm1d(36),
            activation_fn,
            nn.Linear(in_features=36, out_features=10),
            nn.BatchNorm1d(10),
            activation_fn,
            nn.Linear(in_features=10, out_features=2)
        )

    def apply_discriminator(self, x):
        return self.discriminator(x)

    def forward(self, input, alpha):
        reversed_input = GradientReverseFunction.apply(input, alpha)
        x = self.apply_discriminator(reversed_input)
        return x


class AdultCensusRegionDiscriminator(nn.Module):

    def __init__(self, input_dim):
        super(AdultCensusRegionDiscriminator, self).__init__()
        # hidden_dim = int(input_dim * 1.5)
        # self.discriminator = nn.Sequential(
        #     nn.Linear(in_features=input_dim, out_features=hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=hidden_dim, out_features=input_dim),
        #     nn.BatchNorm1d(input_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=input_dim, out_features=2)
        # )
        # self.discriminator = nn.Sequential(
        #     nn.Linear(in_features=input_dim, out_features=30),
        #     nn.BatchNorm1d(30),
        #     activation_fn,
        #     nn.Linear(in_features=30, out_features=10),
        #     nn.BatchNorm1d(10),
        #     activation_fn,
        #     nn.Linear(in_features=10, out_features=2)
        # )
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=30),
            nn.BatchNorm1d(30),
            activation_fn,
            nn.Linear(in_features=30, out_features=15),
            nn.BatchNorm1d(15),
            activation_fn,
            nn.Linear(in_features=15, out_features=6),
            nn.BatchNorm1d(6),
            activation_fn,
            nn.Linear(in_features=6, out_features=2)
        )

    def apply_discriminator(self, x):
        return self.discriminator(x)

    def forward(self, input, alpha):
        reversed_input = GradientReverseFunction.apply(input, alpha)
        x = self.apply_discriminator(reversed_input)
        return x
        # return F.softmax(x, dim=1)
