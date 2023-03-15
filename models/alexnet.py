import torch
from torch import nn


class AlexNet(nn.Module):
    """Fake LeNet with 32x32 color images and 200 classes"""
    def __init__(self, num_classes: int = 200) -> None:
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            #nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.lin = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        out = self.lin(x)
        return out

# class AlexNet(nn.Module):
#     """Alexnet"""

#     def __init__(self, num_classes: int = 200) -> None:
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(11, 11, kernel_size=64, stride=4, padding=2),
#             nn.MaxPool2d(3, stride=2),
#             nn.Conv2d(5,5, kernel_size=192, stride=4, padding=2),
#             nn.MaxPool2d(3,3, stride=2),
#             nn.Conv2d(3,3, kernel_size=384, stride=4, padding=1),
#             nn.Conv2d(3,3, kernel_size=256, stride=4, padding=1),
#             nn.MaxPool2d(3,3, stride=2),
#             nn.AdaptiveAvgPool2d(6,6, kernel_size=2, stride=2),
#             nn.Flatten(), # this needs to be changed
#             nn.Dropout(p=0.5),
#             nn.Linear(out_features=4096),
#             nn.Dropout(p=0.5),
#             nn.Linear(out_features=4096),
#             nn.Linear(out_features=num_classes)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(16 * 6 * 6, 120),
#             nn.Sigmoid(),
#             nn.Linear(120, 84),
#             nn.Sigmoid(),
#             nn.Linear(84, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
