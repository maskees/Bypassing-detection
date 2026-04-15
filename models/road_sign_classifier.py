import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RoadSignClassifier(nn.Module):
    """Transfer-learning classifier for cropped road-sign images."""

    def __init__(self, num_classes=4, backbone="resnet34", pretrained=False):
        super().__init__()
        self.backbone = backbone

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            self._feature_dim = model.fc.in_features
            model.fc = nn.Linear(self._feature_dim, num_classes)
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
            self._feature_dim = model.fc.in_features
            model.fc = nn.Linear(self._feature_dim, num_classes)
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            self._feature_dim = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(self._feature_dim, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.model = model

    @property
    def feature_dim(self):
        """Dimensionality of the feature vector before the classifier head."""
        return self._feature_dim

    def get_features(self, x):
        """Extract feature vector before the final classifier layer."""
        if self.backbone in ("resnet18", "resnet34"):
            m = self.model
            x = m.conv1(x)
            x = m.bn1(x)
            x = m.relu(x)
            x = m.maxpool(x)
            x = m.layer1(x)
            x = m.layer2(x)
            x = m.layer3(x)
            x = m.layer4(x)
            x = m.avgpool(x)
            return torch.flatten(x, 1)
        elif self.backbone == "efficientnet_b0":
            m = self.model
            x = m.features(x)
            x = m.avgpool(x)
            return torch.flatten(x, 1)

    def forward(self, x):
        return self.model(x)


class NormalizedModel(nn.Module):
    """Accept display tensors in [0, 1] and normalize for the backbone."""

    def __init__(self, model, mean=None, std=None):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(mean or IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std or IMAGENET_STD).view(1, 3, 1, 1)
        )

    def _normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x):
        return self.model(self._normalize(x))

    def get_features(self, x):
        return self.model.get_features(self._normalize(x))

    def forward_with_bbox(self, x):
        return self.model.forward_with_bbox(self._normalize(x))


def load_road_sign_classifier_checkpoint(path, device="cpu", strict=True):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = RoadSignClassifier(
        num_classes=config.get("num_classes", 4),
        backbone=config.get("backbone", "resnet34"),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    model.to(device)
    model.eval()
    return model, checkpoint
