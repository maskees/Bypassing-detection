"""
PGD (Projected Gradient Descent) Attack — ML-based, iterative gradient-based.

Multi-step adversarial attack that iteratively applies small FGSM steps
within an ε-ball around the original input. Stronger than FGSM.

Algorithm:
  1. x' = x + uniform(-ε, ε)  [random start]
  2. For each step:
     x' = x' + α * sign(∇_x' L(θ, x', y))
     x' = clip(x' , x - ε, x + ε)
     x' = clip(x', 0, 1)
"""

import torch
import torch.nn.functional as F


def pgd_attack(model, images, labels, epsilon, alpha=None, steps=40, device='cuda'):
    """
    Generate adversarial examples using PGD.

    Args:
        model: Target classifier
        images: Clean input images, values in [0, 1]
        labels: True labels
        epsilon: L∞ perturbation bound
        alpha: Step size per iteration (default: epsilon/4)
        steps: Number of PGD iterations
        device: Computation device

    Returns:
        adv_images, perturbations, success_mask
    """
    if alpha is None:
        alpha = epsilon / 4.0

    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # At higher epsilon, use targeted attack toward a random wrong class
    use_targeted = epsilon > 0.05
    if use_targeted:
        num_classes = model(images[:1]).shape[1]
        target_labels = torch.zeros_like(labels)
        for i in range(labels.size(0)):
            choices = [c for c in range(num_classes) if c != labels[i].item()]
            target_labels[i] = choices[torch.randint(len(choices), (1,)).item()]

    # Random initialization within ε-ball
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)

        outputs = model(adv_images)
        if use_targeted:
            loss = -F.cross_entropy(outputs, target_labels)
        else:
            loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Gradient step
        grad = adv_images.grad.data
        adv_images = adv_images.detach() + alpha * grad.sign()

        # Project back to ε-ball
        delta = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

    perturbations = adv_images - images

    # Check attack success
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_preds = adv_outputs.argmax(dim=1)
        success_mask = (adv_preds != labels)

    return adv_images, perturbations, success_mask


def pgd_attack_single(model, image, label, epsilon, alpha=None, steps=40, device='cuda'):
    """
    PGD attack on a single image. Returns detailed results for visualization.
    """
    model.eval()
    image = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
    label_tensor = torch.tensor([label], device=device) if not isinstance(label, torch.Tensor) else label.unsqueeze(0).to(device)

    adv_images, perturbations, success_mask = pgd_attack(
        model, image, label_tensor, epsilon, alpha, steps, device
    )

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
        'true_label': label if isinstance(label, int) else label.item(),
        'l_inf': perturbations.abs().max().item(),
        'l2': perturbations.norm(2).item(),
    }
