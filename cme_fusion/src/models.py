from __future__ import annotations

import torch
from torch import nn


class SmallBranch(nn.Module):
    def __init__(self, in_ch: int, emb: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, emb, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class FusionCNN(nn.Module):
    def __init__(self, sdo_channels: int):
        super().__init__()
        self.lasco = SmallBranch(1, emb=32)
        self.sdo = SmallBranch(sdo_channels, emb=64)
        self.head = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_lasco, x_sdo):
        a = self.lasco(x_lasco)
        b = self.sdo(x_sdo)
        return self.head(torch.cat([a, b], dim=1)).squeeze(1)
