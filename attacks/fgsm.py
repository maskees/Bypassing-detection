"""
FGSM (Fast Gradient Sign Method) Attack — ML-based, gradient-based.

Single-step adversarial attack that perturbs inputs in the direction
of the gradient of the loss function.

Formula: x_adv = x + ε * sign(∇_x L(θ, x, y))
"""

import torch
import torch.nn.functional as F


def fgsm_attack(model, images, labels, epsilon, device='cuda'):
    """
    Generate adversarial examples using FGSM.
    Uses targeted mode at higher epsilon for diverse misclassification.

    Args:
        model: Target classifier (nn.Module)
        images: Clean input images (batch), values in [0, 1]
        labels: True labels
        epsilon: Perturbation magnitude (L∞ bound)
        device: Computation device

    Returns:
        adv_images: Adversarial images (batch)
        perturbations: Applied perturbations
        success_mask: Boolean mask of successful attacks
    """
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # At higher epsilon, use targeted attack toward a random wrong class
    # to get diverse misclassification instead of collapsing to one class
    use_targeted = epsilon > 0.05
    if use_targeted:
        num_classes = model(images[:1]).shape[1]
        # Pick a random class that is NOT the true label
        target_labels = torch.zeros_like(labels)
        for i in range(labels.size(0)):
            choices = [c for c in range(num_classes) if c != labels[i].item()]
            target_labels[i] = choices[torch.randint(len(choices), (1,)).item()]

    images.requires_grad_(True)

    outputs = model(images)
    if use_targeted:
        # Minimize loss toward target (negative CE)
        loss = -F.cross_entropy(outputs, target_labels)
    else:
        loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.data.sign()
    perturbations = epsilon * grad_sign

    adv_images = torch.clamp(images.data + perturbations, 0.0, 1.0)
    perturbations = adv_images - images.data

    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_preds = adv_outputs.argmax(dim=1)
        success_mask = (adv_preds != labels)

    return adv_images.detach(), perturbations.detach(), success_mask.detach()


def fgsm_attack_single(model, image, label, epsilon, device='cuda'):
    """
    FGSM attack on a single image. Returns detailed results for visualization.
    """
    model.eval()
    image = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
    label = torch.tensor([label], device=device) if not isinstance(label, torch.Tensor) else label.unsqueeze(0).to(device)

    adv_images, perturbations, success_mask = fgsm_attack(model, image, label, epsilon, device)

    with torch.no_grad():
        orig_output = model(image)
        adv_output = model(adv_images)
        orig_probs = F.softmax(orig_output, dim=1)[0]
        adv_probs = F.softmax(adv_output, dim=1)[0]

    return {
        'original': image.squeeze(0).detach().cpu(),
        'adversarial': adv_images.squeeze(0).detach().cpu(),
        'perturbation': perturbations.squeeze(0).detach().cpu(),
        'orig_pred': orig_output.argmax(1).item(),
        'adv_pred': adv_output.argmax(1).item(),
        'orig_probs': orig_probs.detach().cpu().numpy(),
        'adv_probs': adv_probs.detach().cpu().numpy(),
        'success': success_mask[0].item(),
        'true_label': label.item() if label.dim() == 1 else label.squeeze().item(),
        'l_inf': perturbations.abs().max().item(),
        'l2': perturbations.norm(2).item(),
    }
