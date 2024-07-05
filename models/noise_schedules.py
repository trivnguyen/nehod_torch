
import torch
import torch.nn as nn
from torch.nn import functional as F


class DenseMonotone(nn.Module):
    def __init__(
        self, in_features, out_features, kernel_init, bias_init=None,
        use_bias=True, decreasing=True):
        super().__init__()
        self.use_bias = use_bias
        self.decreasing = decreasing

        # initialize weights and biases
        self.weight = nn.Parameter(
            kernel_init(torch.empty(out_features, in_features)))
        if self.use_bias:
            if bias_init is None:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.bias = nn.Parameter(bias_init(torch.empty(out_features)))

    def forward(self, x):
        weight = torch.abs(self.weight)
        if self.decreasing:
            weight = -weight
        out = F.linear(x, weight)
        if self.use_bias:
            out += self.bias
        return out


class NoiseScheduleNet(nn.Module):
    def __init__(
        self, gamma_min=-6.0, gamma_max=7.0, n_features=1024, nonlinear=True,
        scale_non_linear_init=False):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.n_features = n_features
        self.nonlinear = nonlinear
        self.scale_non_linear_init = scale_non_linear_init

        init_bias = self.gamma_max
        init_scale = self.gamma_min - init_bias

        # Initialize layers
        self.l1 = DenseMonotone(
            1, 1, kernel_init=lambda x: nn.init.constant_(x, init_scale),
            bias_init=lambda x: nn.init.constant_(x, init_bias))

        if self.nonlinear:
            if self.scale_non_linear_init:
                stddev_l2 = init_scale
                stddev_l3 = init_scale
            else:
                stddev_l2 = stddev_l3 = 0.01

            self.l2 = DenseMonotone(
                1, self.n_features,
                kernel_init=lambda x: nn.init.normal_(x, std=stddev_l2))
            self.l3 = DenseMonotone(
                self.n_features, 1, kernel_init=lambda x: nn.init.normal_(x, std=stddev_l3),
                use_bias=False, decreasing=False)

    def forward(self, t):
        if torch.is_tensor(t):
            if t.dim() == 0 or t.dim() == 1:
                t = t.view(-1, 1)
        elif isinstance(t, (int, float)):
            t = torch.tensor([[t]], dtype=torch.float32)

        t = torch.tensor(t, dtype=torch.float32)

        h = self.l1(t)
        if self.nonlinear:
            _h = 2.0 * (t - 0.5)  # Scale input to [-1, +1]
            _h = self.l2(_h)
            _h = 2 * (torch.sigmoid(_h) - 0.5)
            _h = self.l3(_h) / self.n_features
            h += _h

        return torch.squeeze(h, dim=-1)


class NoiseScheduleFixedLinear(nn.Module):
    """ Linear schedule for diffusion variance """
    def __init__(self, gamma_min=-6.0, gamma_max=6.0):
        super().__init__()
        self.gamma_min = torch.tensor(gamma_min, dtype=torch.float32)
        self.gamma_max = torch.tensor(gamma_max, dtype=torch.float32)

    def forward(self, t):
        return self.gamma_max + (self.gamma_min - self.gamma_max) * t


class NoiseScheduleScalar(nn.Module):
    def __init__(self, gamma_min=-6.0, gamma_max=7.0):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        init_bias = self.gamma_max
        init_scale = self.gamma_min - self.gamma_max
        self.w = nn.Parameter(torch.tensor([init_scale], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    def __call__(self, t):
        # gamma = self.gamma_max - |self.gamma_min - self.gamma_max| * t
        return self.b + - torch.abs(self.w) * t
