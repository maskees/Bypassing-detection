"""
Master Training Script — Trains all models and runs evaluation.

Pipeline:
  1. Download MNIST dataset
  2. Train base CNN model (~99% accuracy)
  3. Train adversarially robust model (with PGD adversarial training)
  4. Train distilled model (teacher→student with temperature scaling)
  5. Train detection network (binary classifier on features)
  6. Run full evaluation (4 attacks × 5 defenses)
  7. Save all models + results

Usage: python train_all.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import json
import time

from models.target_model import MNISTNet, DetectorNet
from defenses.adversarial_training import train_adversarial_model
from defenses.defensive_distillation import train_distilled_model
from defenses.detection_network import train_detector
from evaluation.evaluator import run_full_evaluation


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU (training will be slower)")
    return device


def get_data_loaders(batch_size=128):
    """Download MNIST and create data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"MNIST loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_loader, test_loader


def train_base_model(train_loader, test_loader, epochs=5, lr=0.001, device='cuda'):
    """Train the base MNIST classifier."""
    print("\n" + "=" * 50)
    print("STEP 1: Training Base Model")
    print("=" * 50)

    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        avg_loss = total_loss / len(train_loader)

        print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, "
              f"Train: {train_acc:.2f}%, Test: {test_acc:.2f}%")

    print(f"  ✓ Base model trained — Test accuracy: {test_acc:.2f}%")
    return model


def evaluate_model(model, test_loader, device='cuda', name="Model"):
    """Quick evaluation of a model on clean test data."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    print(f"  {name} clean accuracy: {acc:.2f}%")
    return acc


def main():
    start_time = time.time()
    device = get_device()

    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ── Load Data ──
    train_loader, test_loader = get_data_loaders()

    # ── Step 1: Train Base Model ──
    base_model = train_base_model(train_loader, test_loader, epochs=5, device=device)
    torch.save(base_model.state_dict(), 'saved_models/base_model.pth')
    print("  ✓ Saved: saved_models/base_model.pth")

    # ── Step 2: Adversarial Training ──
    print("\n" + "=" * 50)
    print("STEP 2: Adversarial Training Defense")
    print("=" * 50)
    adv_model = train_adversarial_model(
        MNISTNet, train_loader, epsilon=0.3, alpha=0.01,
        pgd_steps=7, epochs=10, lr=0.001, device=device
    )
    torch.save(adv_model.state_dict(), 'saved_models/adv_trained_model.pth')
    print("  ✓ Saved: saved_models/adv_trained_model.pth")
    evaluate_model(adv_model, test_loader, device, "Adversarial Model")

    # ── Step 3: Defensive Distillation ──
    print("\n" + "=" * 50)
    print("STEP 3: Defensive Distillation")
    print("=" * 50)
    distilled_model = train_distilled_model(
        teacher_model=base_model,
        model_class=MNISTNet,
        train_loader=train_loader,
        temperature=20.0,
        alpha=0.7,
        epochs=10,
        lr=0.001,
        device=device
    )
    torch.save(distilled_model.state_dict(), 'saved_models/distilled_model.pth')
    print("  ✓ Saved: saved_models/distilled_model.pth")
    evaluate_model(distilled_model, test_loader, device, "Distilled Model")

    # ── Step 4: Detection Network ──
    print("\n" + "=" * 50)
    print("STEP 4: Detection Network")
    print("=" * 50)
    detector = DetectorNet()
    detector = train_detector(
        target_model=base_model,
        detector_model=detector,
        train_loader=train_loader,
        epsilon=0.3,
        epochs=10,
        lr=0.001,
        device=device
    )
    torch.save(detector.state_dict(), 'saved_models/detector_model.pth')
    print("  ✓ Saved: saved_models/detector_model.pth")

    # ── Step 5: Full Evaluation ──
    print("\n" + "=" * 50)
    print("STEP 5: Full Evaluation (4 attacks × 5 defenses)")
    print("=" * 50)
    print("  Note: EC attacks (GA, DE) are slower — evaluating on 50 samples")

    models_dict = {
        'base': base_model,
        'adv_trained': adv_model,
        'distilled': distilled_model,
        'detector': detector,
    }

    results = run_full_evaluation(
        models_dict=models_dict,
        test_loader=test_loader,
        epsilon=0.3,
        num_samples=50,
        device=device,
        save_path='results/evaluation_results.json'
    )

    # ── Summary ──
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"TRAINING COMPLETE — Total time: {elapsed/60:.1f} minutes")
    print("=" * 50)
    print("\nSaved files:")
    print("  • saved_models/base_model.pth")
    print("  • saved_models/adv_trained_model.pth")
    print("  • saved_models/distilled_model.pth")
    print("  • saved_models/detector_model.pth")
    print("  • results/evaluation_results.json")
    print("\nRun the web interface: python app.py")


if __name__ == '__main__':
    main()
