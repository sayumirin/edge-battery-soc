import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid
from torch import nn


def coulomb_count(time_s, current_a, initial_soc, capacity_ah):
    t, i = np.asarray(time_s), np.asarray(current_a)
    if capacity_ah <= 0 or len(t) != len(i) or (np.diff(t) <= 0).any():
        raise ValueError("Invalid integration inputs")
    return initial_soc - cumulative_trapezoid(i, t, initial=0) / (3600 * capacity_ah)


class GRU(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.input = nn.Linear(3, 3)
        with torch.no_grad():
            self.input.weight.copy_(torch.eye(3))
            self.input.bias.zero_()
        self.gru = nn.GRU(3, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(self.input(x))
        return self.head(out[:, -1]).squeeze(-1)

    def step(self, x, state=None):
        out, state = self.gru(self.input(x), state)
        return self.head(out[:, -1]).squeeze(-1), state


class CNN(nn.Module):
    """Small dilated convolution baseline; final endpoint has 15-step receptive field."""

    def __init__(self, hidden=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, hidden, 3, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, dilation=4),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.features(x.transpose(1, 2))[:, :, -1]).squeeze(-1)
