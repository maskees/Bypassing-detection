"""
Adversarial Training Defense — Proactive defense.

Retrains the model with PGD-generated adversarial examples mixed into
the training data. The most effective known defense against white-box attacks.

Algorithm:
  For each training batch:
    1. Generate adversarial examples using PGD
    2. Compute loss on adversarial examples
    3. Update model weights to minimize adversarial loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy


def pgd_for_training(model, images, labels, epsilon, alpha, steps):
    """Generate PGD adversarial examples during training (model in train mode)."""
    adv = images.clone().detach()
    adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)

    for _ in range(steps):
        adv.requires_grad_(True)
        outputs = model(adv)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        grad = adv.grad.data

        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - images, -epsilon, epsilon)
        adv = torch.clamp(images + delta, 0.0, 1.0).detach()

    return adv


def train_adversarial_model(model_factory, train_loader, epsilon=0.3,
                            alpha=0.01, pgd_steps=7, epochs=10,
                            lr=0.01, device='cuda', weight_decay=1e-4,
                            class_weights=None):
    """
    Train a model using adversarial training (Madry et al. approach).

    Args:
        model_factory: Callable that returns a fresh model instance
        train_loader: Training data loader yielding (images, labels) tuples
        epsilon: Perturbation bound for PGD (in input space)
        alpha: PGD step size
        pgd_steps: Number of PGD steps per training batch
        epochs: Training epochs
        lr: Learning rate
        device: Computation device
        weight_decay: AdamW weight decay
        class_weights: Optional class weights for cross-entropy loss

    Returns:
        Trained adversarially robust model
    """
    model = model_factory().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n[Defense] Adversarial Training")
    print(f"  eps={epsilon}, alpha={alpha}, PGD steps={pgd_steps}, epochs={epochs}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples
            adv_images = pgd_for_training(model, images, labels,
                                          epsilon, alpha, pgd_steps)

            # Train on adversarial examples
            optimizer.zero_grad()
            outputs = model(adv_images)
            loss = F.cross_entropy(outputs, labels, weight=class_weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, "
              f"Robust Acc: {acc:.2f}%")

    return model
