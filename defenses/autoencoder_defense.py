"""
Autoencoder Defense — Reactive defense via learned reconstruction.

Uses a denoising autoencoder trained on clean images to project potentially
adversarial inputs back onto the clean-image manifold. High-frequency
adversarial perturbations are stripped during reconstruction, while the
semantic content of the road sign is preserved by the U-Net skip connections.

Pipeline:
    adversarial image [0,1] → Autoencoder → reconstructed [0,1] → classifier

Advantage over Input Transformation (hand-crafted filters):
    - Learned from data rather than fixed filter parameters
    - Adapts to the distribution of clean images
    - Can be stacked on top of any pre-trained classifier
"""

import torch
import torch.nn.functional as F


def apply_autoencoder_defense(images, autoencoder):
    """
    Pass images through the autoencoder to strip adversarial perturbations.

    Args:
        images: Input batch (N, C, H, W) in [0, 1].
        autoencoder: Trained DenoisingAutoencoder in eval mode.

    Returns:
        Reconstructed images (N, C, H, W) in [0, 1].
    """
    autoencoder.eval()
    with torch.no_grad():
        reconstructed = autoencoder(images)
    return reconstructed.clamp(0.0, 1.0)


def autoencoder_and_predict(classifier, autoencoder, images, device='cuda'):
    """
    Denoise with the autoencoder then classify.

    Args:
        classifier: Main classifier (typically NormalizedModel-wrapped).
        autoencoder: Trained DenoisingAutoencoder.
        images: Input batch in [0, 1].
        device: Computation device.

    Returns:
        predictions: Predicted class indices.
        probabilities: Softmax probabilities.
        reconstructed: The denoised images (useful for visualization).
    """
    classifier.eval()
    autoencoder.eval()
    images = images.to(device)

    with torch.no_grad():
        reconstructed = autoencoder(images).clamp(0.0, 1.0)
        outputs = classifier(reconstructed)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

    return preds, probs, reconstructed
