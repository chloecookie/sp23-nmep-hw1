import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn

import math

# pulled from Dr. Karpathy's minGPT implementation
class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut_flag, stride=1):
        super(ResNetBlock, self).__init__()

        self.shortcut_flag = shortcut_flag
        self.gelu = GELU()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            self.gelu,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        if shortcut_flag:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        out = self.features(x)

        if not self.shortcut_flag:
            out += x
        else:
            shortcut = self.shortcut(x)
            out += shortcut
        return self.gelu(out)


class ResNet34(nn.Module):
    def __init__(self, num_classes=200):
        # Read the following, and uncomment it when you understand it, no need to add more code
        num_classes = num_classes
        super(ResNet34, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=7,
                                stride=1,
                                padding="same",
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gelu = GELU()
        self.pool = nn.MaxPool2d(2)

        self.layer1 = self.make_block(out_channels=64, num_layers=3, no_shortcut=True)
        self.layer2 = self.make_block(out_channels=128, num_layers=4)
        self.layer3 = self.make_block(out_channels=256, num_layers=5)
        self.layer4 = self.make_block(out_channels=512, num_layers=3)

        # number of out channels is the number of filters and they have to increase because you need to get more accurate as the network goes on
        self.linear = nn.Linear(512, num_classes)


    def make_block(self, out_channels, num_layers, no_shortcut=False):
        # Read the following, and uncomment it when you understand it, no need to add more code
        layers = []
        for i, layer in enumerate(range(num_layers)):
            if i == 0 and not no_shortcut:
                layers.append(ResNetBlock(self.in_channels, out_channels, True, 2))
            elif i == 0 and no_shortcut:
                layers.append(ResNetBlock(self.in_channels, out_channels, False, 1))
            else:
                layers.append(ResNetBlock(self.in_channels, out_channels, False, 1))
            self.in_channels = out_channels
        return nn.Sequential(*layers)


    def forward(self, x):
        # Read the following, and uncomment it when you understand it, no need to add more code
        x = self.gelu(self.pool(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x
