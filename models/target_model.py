"""
Neural network architectures for Traffic Sign classification and adversarial detection.
Adapted from MNIST to Indian Traffic Sign Recognition dataset (58 classes, RGB 32x32).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 58
IMG_SIZE = 32  # We resize all traffic sign images to 32x32
IN_CHANNELS = 3  # RGB


class TrafficNet(nn.Module):
    """
    CNN for Traffic Sign classification.
    Architecture: 4 conv layers + 2 FC layers with BatchNorm and Dropout.
    Input: 3×32×32 RGB images | Output: 58 classes
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(TrafficNet, self).__init__()
        # Block 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(IN_CHANNELS, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 2: 32 -> 64 channels
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)

        # After 2 pools: 32->16->8, channels=64 → 64*8*8=4096
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Classifier
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

    def get_features(self, x):
        """Extract feature vector before classifier (for detection network)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_with_temperature(self, x, temperature=1.0):
        """Forward pass with temperature scaling for distillation."""
        logits = self.forward(x)
        return logits / temperature


class DetectorNet(nn.Module):
    """
    Binary classifier to detect adversarial examples.
    Takes feature vectors from TrafficNet.get_features() as input.
    Outputs 2 classes: [clean, adversarial].
    """

    def __init__(self, input_dim=64 * 8 * 8):
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


# Backward-compatible alias
MNISTNet = TrafficNet
