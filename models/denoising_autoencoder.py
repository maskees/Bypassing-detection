"""
Denoising Autoencoder — Defense against adversarial perturbations.

Architecture: U-Net style encoder-decoder with skip connections.
Trained unsupervised on clean road-sign images with injected noise to
learn a projection onto the manifold of "natural" clean images.

At inference time, adversarial inputs are passed through the autoencoder
BEFORE the classifier. The reconstruction strips high-frequency adversarial
perturbations while preserving semantic content.

Operates in [0, 1] display-image space. The downstream classifier should
be a NormalizedModel that handles ImageNet normalization internally.
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Two stacked Conv-BN-ReLU layers."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DenoisingAutoencoder(nn.Module):
    """
    U-Net denoising autoencoder for 224x224 RGB images in [0, 1].

    Args:
        base_channels: Number of channels in the first encoder block.
                       Channels double at each downsampling step.
    """

    def __init__(self, base_channels=32):
        super().__init__()
        c = base_channels

        # ── Encoder ──
        self.enc1 = _ConvBlock(3, c)            # 224x224
        self.pool1 = nn.MaxPool2d(2)            # -> 112
        self.enc2 = _ConvBlock(c, c * 2)        # 112x112
        self.pool2 = nn.MaxPool2d(2)            # -> 56
        self.enc3 = _ConvBlock(c * 2, c * 4)    # 56x56
        self.pool3 = nn.MaxPool2d(2)            # -> 28

        # ── Bottleneck ──
        self.bottleneck = _ConvBlock(c * 4, c * 8)  # 28x28

        # ── Decoder ──
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(c * 8, c * 4)    # concat skip: c*4 + c*4 = c*8

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(c * 4, c * 2)    # concat skip: c*2 + c*2 = c*4

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(c * 2, c)        # concat skip: c + c = c*2

        self.out_conv = nn.Conv2d(c, 3, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder path with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Sigmoid bounds output to [0, 1]
        return torch.sigmoid(self.out_conv(d1))


def load_autoencoder_checkpoint(checkpoint_path, device='cuda', base_channels=32):
    """Load a trained autoencoder from checkpoint.

    Returns:
        model: DenoisingAutoencoder in eval mode on the given device.
        checkpoint: The raw checkpoint dict (may contain metrics).
    """
    model = DenoisingAutoencoder(base_channels=base_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Accept either raw state_dict or {'state_dict': ..., 'metrics': ...}
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint
