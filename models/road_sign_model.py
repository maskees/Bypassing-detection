import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class RoadSignResNet(nn.Module):
    """ResNet road-sign classifier with an optional bounding-box head."""

    def __init__(self, num_classes=4, backbone="resnet34", pretrained=False):
        super().__init__()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        feature_dim = resnet.fc.in_features
        self.backbone_name = backbone
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.bbox_head = nn.Linear(feature_dim, 4)

    def get_features(self, x):
        features = self.features(x)
        return torch.flatten(features, 1)

    def forward_with_bbox(self, x):
        features = self.get_features(x)
        logits = self.classifier(features)
        bbox = torch.sigmoid(self.bbox_head(features))
        return logits, bbox

    def forward(self, x):
        logits, _ = self.forward_with_bbox(x)
        return logits


def load_road_sign_checkpoint(path, device="cpu", strict=True):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = RoadSignResNet(
        num_classes=config.get("num_classes", 4),
        backbone=config.get("backbone", "resnet34"),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    model.to(device)
    model.eval()
    return model, checkpoint


def road_sign_loss(logits, pred_bbox, labels, true_bbox, bbox_weight=5.0, class_weights=None):
    class_loss = F.cross_entropy(logits, labels, weight=class_weights)
    bbox_loss = F.smooth_l1_loss(pred_bbox, true_bbox)
    return class_loss + bbox_weight * bbox_loss, class_loss, bbox_loss
