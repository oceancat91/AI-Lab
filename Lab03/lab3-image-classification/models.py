"""Two controlled small CNN architectures, trained from scratch."""
import torch
from torch import nn
from config import CHANNELS


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, residual):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.skip = (nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                  nn.BatchNorm2d(out_channels)) if residual else None)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        if self.skip is not None:
            y = y + self.skip(x)
        return self.pool(self.relu(y))


class SmallCNN(nn.Module):
    def __init__(self, architecture='plain'):
        super().__init__()
        if architecture not in ['plain', 'residual']:
            raise ValueError(architecture)
        ch = (3,) + CHANNELS
        self.features = nn.Sequential(*(Block(ch[i], ch[i + 1], architecture == 'residual') for i in range(3)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(CHANNELS[-1], 10)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def make_model(architecture):
    return SmallCNN(architecture).to(memory_format=torch.channels_last)
