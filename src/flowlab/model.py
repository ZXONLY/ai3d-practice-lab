"""A velocity network. Input: (x,y,t). Output: (dx/dt,dy/dt), not a point."""
import torch
from torch import nn


class VelocityMLP(nn.Module):
    def __init__(self, hidden_dim=128, hidden_layers=3):
        super().__init__()
        layers = []
        size = 3
        for _ in range(hidden_layers):
            layers += [nn.Linear(size, hidden_dim), nn.SiLU()]
            size = hidden_dim
        layers += [nn.Linear(size, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, t):
        if z.ndim != 2 or z.shape[1] != 2 or t.shape != (z.shape[0], 1):
            raise ValueError("Expected z=(batch,2), t=(batch,1)")
        return self.net(torch.cat([z, t], dim=1))
