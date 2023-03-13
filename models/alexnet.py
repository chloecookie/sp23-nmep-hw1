import torch
from torch import nn


class AlexNet(nn.Module):
    """Alexnet"""

    def __init__(self, num_classes: int = 200) -> None:
        super().__init__(3,3)
        self.features = nn.Sequential(
            nn.Conv2d(11, 11, kernel_size=64, stride=4, padding=2),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(5,5, kernel_size=192, stride=4, padding=2),
            nn.MaxPool2d(3,3, stride=2),
            nn.Conv2d(3,3, kernel_size=384, stride=4, padding=1),
            nn.Conv2d(3,3, kernel_size=256, stride=4, padding=1),
            nn.MaxPool2d(3,3, stride=2),
            nn.AdaptiveAvgPool2d(6,6, kernel_size=2, stride=2),
            nn.Flatten(), # this needs to be changed
            nn.Dropout(p=0.5),
            nn.Linear(out_features=4096),
            nn.Dropout(p=0.5),
            nn.Linear(out_features=4096),
            nn.Linear(out_features=num_classes)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
