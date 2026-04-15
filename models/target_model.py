"""
Neural network architectures for adversarial detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectorNet(nn.Module):
    """
    Binary classifier to detect adversarial examples.
    Takes feature vectors from model.get_features() as input.
    Outputs 2 classes: [clean, adversarial].

    Default input_dim=512 matches ResNet-18/34 feature dimensionality.
    """

    def __init__(self, input_dim=512):
        super(DetectorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
