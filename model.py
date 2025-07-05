# model.py
import torch
from torch import nn

# MLP模型
class PRPredictor(nn.Module):
    def __init__(self, T=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, T)
        )

    def forward(self, x):
        return self.model(x)

