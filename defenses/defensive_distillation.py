"""
Defensive Distillation Defense — Proactive defense.

Uses knowledge distillation with temperature scaling to create a model
with smoother decision boundaries that are harder to exploit.

Algorithm:
  1. Train a "teacher" model normally (or use pre-trained model)
  2. Generate soft labels: softmax(logits / T) with high temperature T
  3. Train "student" model on these soft labels at temperature T
  4. At inference, student uses temperature T=1

The high temperature produces softer probability distributions,
smoothing the loss surface and reducing gradient magnitudes that
attackers rely on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy


def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=20.0, alpha=0.7):
    """
    Combined distillation loss.

    Loss = α * KL(soft_student || soft_teacher) * T² + (1-α) * CE(student, labels)

    Args:
        student_logits: Raw logits from student model
        teacher_logits: Raw logits from teacher model
        labels: True hard labels
        temperature: Distillation temperature
        alpha: Weight for soft label loss vs hard label loss
    """
    # Soft targets from teacher
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)

    # KL divergence loss (scaled by T²)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # Hard label cross-entropy loss
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    total_loss = alpha * kl_loss + (1 - alpha) * hard_loss
    return total_loss


def train_distilled_model(teacher_model, model_factory, train_loader,
                          temperature=20.0, alpha=0.7, epochs=10,
                          lr=0.01, device='cuda', weight_decay=1e-4):
    """
    Train a student model using defensive distillation.

    Args:
        teacher_model: Pre-trained teacher model (frozen)
        model_factory: Callable that returns a fresh student model instance
        train_loader: Training data loader yielding (images, labels) tuples
        temperature: Distillation temperature (higher = softer labels)
        alpha: Weight for distillation loss vs standard CE loss
        epochs: Training epochs
        lr: Learning rate
        device: Computation device
        weight_decay: AdamW weight decay

    Returns:
        Trained distilled student model
    """
    teacher_model.eval()
    teacher_model = teacher_model.to(device)

    student_model = model_factory().to(device)
    optimizer = optim.AdamW(student_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n[Defense] Defensive Distillation")
    print(f"  Temperature={temperature}, alpha={alpha}, epochs={epochs}")

    for epoch in range(epochs):
        student_model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Get teacher soft predictions
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            # Student forward pass
            student_logits = student_model(images)

            # Distillation loss
            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                temperature=temperature, alpha=alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, "
              f"Clean Acc: {acc:.2f}%")

    return student_model
