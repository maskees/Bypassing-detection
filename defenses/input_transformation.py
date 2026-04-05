"""
Input Transformation Defense — Reactive defense.

Pre-processes input images before classification to remove adversarial
perturbations. Techniques applied:
  1. Gaussian Smoothing — blurs high-frequency noise
  2. Bit-Depth Reduction — quantizes pixel values
  3. JPEG-like Compression — removes fine-grained perturbations

Advantage: Can be applied to any pre-trained model without retraining.
Weakness: Adaptive attackers aware of the transformation can bypass it.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import io


def gaussian_smooth(images, kernel_size=3, sigma=1.0):
    """Apply Gaussian smoothing to a batch of images."""
    # Create Gaussian kernel
    channels = images.shape[1]
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    kernel = kernel.to(images.device)

    padding = kernel_size // 2
    smoothed = F.conv2d(images, kernel, padding=padding, groups=channels)
    return torch.clamp(smoothed, 0.0, 1.0)


def bit_depth_reduction(images, bits=4):
    """Reduce bit depth of pixel values."""
    levels = 2 ** bits
    reduced = torch.round(images * (levels - 1)) / (levels - 1)
    return torch.clamp(reduced, 0.0, 1.0)


def jpeg_compression(images, quality=75):
    """Apply JPEG-like compression/decompression to images."""
    device = images.device
    batch_size = images.shape[0]
    result = torch.zeros_like(images)

    for i in range(batch_size):
        img = images[i].cpu()
        # Convert to PIL Image
        if img.shape[0] == 1:
            img_np = (img.squeeze(0).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='L')
        else:
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

        # Compress and decompress via JPEG
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)

        # Convert back to tensor
        comp_np = np.array(compressed).astype(np.float32) / 255.0
        if comp_np.ndim == 2:
            comp_np = comp_np[np.newaxis, ...]
        else:
            comp_np = comp_np.transpose(2, 0, 1)
        result[i] = torch.tensor(comp_np)

    return result.to(device)


def apply_input_transforms(images, methods=None):
    """
    Apply a pipeline of input transformations.

    Args:
        images: Input image batch (N, C, H, W)
        methods: List of transforms to apply.
                 Options: 'gaussian', 'bitdepth', 'jpeg'
                 Default: all three.

    Returns:
        Transformed images
    """
    if methods is None:
        methods = ['gaussian', 'bitdepth', 'jpeg']

    result = images.clone()

    for method in methods:
        if method == 'gaussian':
            result = gaussian_smooth(result, kernel_size=3, sigma=1.0)
        elif method == 'bitdepth':
            result = bit_depth_reduction(result, bits=4)
        elif method == 'jpeg':
            result = jpeg_compression(result, quality=75)

    return result


def transform_and_predict(model, images, methods=None, device='cuda'):
    """
    Apply input transformations and then classify.

    Args:
        model: Classifier model
        images: Input images
        methods: Transform methods to apply
        device: Computation device

    Returns:
        predictions: Predicted labels
        probabilities: Class probabilities
    """
    model.eval()
    images = images.to(device)
    transformed = apply_input_transforms(images, methods)

    with torch.no_grad():
        outputs = model(transformed)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

    return preds, probs
