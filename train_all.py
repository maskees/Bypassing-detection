"""
train_all.py — Canonical training script for the Road Sign Adversarial Robustness Pipeline.

Trains all 4 models and runs the full evaluation matrix.
Designed to run headless (no matplotlib) on an RTX 3050 or better.

Usage:
    python train_all.py
    python train_all.py --base-epochs 20 --eval-samples 100

Estimated time on RTX 3050 Laptop:
    Base model (15 epochs)      ~2 min
    Adversarial training (10)   ~5 min
    Distillation (10)           ~2 min
    Detector (10)               ~3 min
    Full evaluation             ~5 min
    Total                       ~17 min
"""

import os
import sys
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── Project imports ──
from models.target_model import TrafficNet, DetectorNet
from models.data_utils import get_data_loaders, get_train_loader, get_test_loader
from defenses.adversarial_training import train_adversarial_model
from defenses.defensive_distillation import train_distilled_model
from defenses.detection_network import train_detector
from evaluation.evaluator import run_full_evaluation

# ── Output directories ──
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train all models for road sign adversarial robustness.')
    parser.add_argument('--base-epochs', type=int, default=15, help='Epochs for base model training (default: 15)')
    parser.add_argument('--adv-epochs', type=int, default=10, help='Epochs for adversarial training (default: 10)')
    parser.add_argument('--distill-epochs', type=int, default=10, help='Epochs for distillation (default: 10)')
    parser.add_argument('--detector-epochs', type=int, default=10, help='Epochs for detector training (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--eval-samples', type=int, default=50, help='Samples for evaluation matrix (default: 50)')
    parser.add_argument('--epsilon', type=float, default=0.03, help='L-inf perturbation bound (default: 0.03)')
    parser.add_argument('--num-workers', type=int, default=None, help='DataLoader workers (default: auto)')
    parser.add_argument('--skip-eval', action='store_true', help='Skip the evaluation step')
    parser.add_argument('--base-only', action='store_true', help='Train only the base model')
    return parser.parse_args()


def print_banner(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def evaluate_model(model, test_loader, device, name="Model"):
    """Evaluate model accuracy on clean test data."""
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
    acc = 100.0 * correct / total
    print(f"  {name}: {acc:.2f}% ({correct}/{total})")
    return acc


def train_base_model(train_loader, test_loader, epochs, device):
    """Train the base TrafficNet classifier."""
    print_banner("STEP 1/4: Training Base Model (TrafficNet)")
    print(f"  Epochs: {epochs}  |  Device: {device}")

    model = TrafficNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Cosine annealing gives smoother convergence than StepLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_acc = 0.0
    start = time.time()

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

        train_acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate_model(model, test_loader, device, "Test")

        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch + 1}/{epochs} — "
              f"Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | "
              f"Test: {test_acc:.2f}% | LR: {lr:.6f}")

        if test_acc > best_acc:
            best_acc = test_acc

    elapsed = time.time() - start
    print(f"\n  ✅ Base model trained in {elapsed:.1f}s — Best test accuracy: {best_acc:.2f}%")

    torch.save(model.state_dict(), 'saved_models/base_model.pth')
    print("  💾 Saved: saved_models/base_model.pth")

    return model


def train_adversarial(train_loader, test_loader, epochs, epsilon, device):
    """Train the adversarially hardened model."""
    print_banner("STEP 2/4: Adversarial Training Defense")
    print(f"  Epochs: {epochs} | ε={epsilon} | α=0.01 | PGD steps=7")

    start = time.time()
    adv_model = train_adversarial_model(
        TrafficNet, train_loader,
        epsilon=epsilon,
        alpha=0.01,
        pgd_steps=7,
        epochs=epochs,
        lr=0.001,
        device=device,
    )
    elapsed = time.time() - start

    adv_acc = evaluate_model(adv_model, test_loader, device, "Adv Model (clean)")
    print(f"\n  ✅ Adversarial training completed in {elapsed:.1f}s")

    torch.save(adv_model.state_dict(), 'saved_models/adv_trained_model.pth')
    print("  💾 Saved: saved_models/adv_trained_model.pth")

    return adv_model


def train_distilled(base_model, train_loader, test_loader, epochs, device):
    """Train the distilled model."""
    print_banner("STEP 3/4: Defensive Distillation")
    print(f"  Epochs: {epochs} | Temperature=20.0 | α=0.7")

    start = time.time()
    distilled_model = train_distilled_model(
        teacher_model=base_model,
        model_class=TrafficNet,
        train_loader=train_loader,
        temperature=20.0,
        alpha=0.7,
        epochs=epochs,
        lr=0.001,
        device=device,
    )
    elapsed = time.time() - start

    dist_acc = evaluate_model(distilled_model, test_loader, device, "Distilled (clean)")
    print(f"\n  ✅ Distillation completed in {elapsed:.1f}s")

    torch.save(distilled_model.state_dict(), 'saved_models/distilled_model.pth')
    print("  💾 Saved: saved_models/distilled_model.pth")

    return distilled_model


def train_detector_net(base_model, train_loader, epochs, epsilon, device):
    """Train the adversarial detector network."""
    print_banner("STEP 4/4: Detection Network")
    print(f"  Epochs: {epochs} | ε={epsilon}")

    start = time.time()
    detector = DetectorNet()
    detector = train_detector(
        target_model=base_model,
        detector_model=detector,
        train_loader=train_loader,
        epsilon=epsilon,
        epochs=epochs,
        lr=0.001,
        device=device,
    )
    elapsed = time.time() - start

    print(f"\n  ✅ Detector trained in {elapsed:.1f}s")

    torch.save(detector.state_dict(), 'saved_models/detector_model.pth')
    print("  💾 Saved: saved_models/detector_model.pth")

    return detector


def run_evaluation(models_dict, test_loader, epsilon, num_samples, device):
    """Run the full 4-attack × 5-defense evaluation matrix."""
    print_banner("EVALUATION: 4 Attacks × 5 Defenses")
    print(f"  Epsilon: {epsilon} | Samples: {num_samples}")

    start = time.time()
    eval_results = run_full_evaluation(
        models_dict=models_dict,
        test_loader=test_loader,
        epsilon=epsilon,
        num_samples=num_samples,
        device=device,
        save_path='results/evaluation_results.json',
    )
    elapsed = time.time() - start

    print(f"\n  ✅ Evaluation completed in {elapsed / 60:.1f} minutes")
    print("  💾 Saved: results/evaluation_results.json")

    # Print summary table
    print_results_table(eval_results)

    return eval_results


def print_results_table(results):
    """Print the attack success rate table to stdout."""
    attack_display = results['attack_names']
    defense_display = results['defense_names']
    matrix = results['results']

    print(f"\n{'─' * 90}")
    print(f"  ATTACK SUCCESS RATE (%) — Lower = better defense")
    print(f"{'─' * 90}")

    header = f"{'Attack':<28}"
    for dk in defense_display:
        header += f"{defense_display[dk]:>14}"
    print(header)
    print("─" * 90)

    for ak in attack_display:
        row = f"{attack_display[ak]:<28}"
        for dk in defense_display:
            val = matrix.get(ak, {}).get(dk, {})
            asr = val.get('attack_success_rate', 'N/A')
            if isinstance(asr, (int, float)):
                row += f"{asr:>13.1f}%"
            else:
                row += f"{'N/A':>14}"
        print(row)

    print(f"{'─' * 90}\n")


def main():
    args = parse_args()

    # ── Device selection ──
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        print("⚡ CUDA optimizations enabled: TF32 + cuDNN benchmark")
    else:
        device = 'cpu'
        print("⚠️  Using CPU — training will be slower")

    print(f"🔧 PyTorch {torch.__version__} | Device: {device}")

    # ── Load data ──
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ── Step 1: Base model ──
    total_start = time.time()
    base_model = train_base_model(train_loader, test_loader, args.base_epochs, device)

    if args.base_only:
        print("\n✅ Base model only — done.")
        return

    # ── Step 2: Adversarial training ──
    adv_model = train_adversarial(train_loader, test_loader, args.adv_epochs, args.epsilon, device)

    # ── Step 3: Distillation ──
    distilled_model = train_distilled(base_model, train_loader, test_loader, args.distill_epochs, device)

    # ── Step 4: Detector ──
    detector = train_detector_net(base_model, train_loader, args.detector_epochs, args.epsilon, device)

    # ── Summary ──
    print_banner("MODEL SUMMARY")
    evaluate_model(base_model, test_loader, device, "Base CNN")
    evaluate_model(adv_model, test_loader, device, "Adversarially Trained")
    evaluate_model(distilled_model, test_loader, device, "Distilled")

    total_elapsed = time.time() - total_start
    print(f"\n  Total training time: {total_elapsed / 60:.1f} minutes")

    for f in ['base_model.pth', 'adv_trained_model.pth', 'distilled_model.pth', 'detector_model.pth']:
        path = f'saved_models/{f}'
        size = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        print(f"  💾 {path:<42} ({size:.0f} KB)")

    # ── Step 5: Evaluation ──
    if not args.skip_eval:
        models_dict = {
            'base': base_model,
            'adv_trained': adv_model,
            'distilled': distilled_model,
            'detector': detector,
        }
        run_evaluation(models_dict, test_loader, args.epsilon, args.eval_samples, device)
    else:
        print("\n⏭️  Evaluation skipped (--skip-eval)")

    print("\n🚀 All done! Launch the dashboard with:")
    print("   python app.py")
    print("   Open: http://localhost:5000")


if __name__ == '__main__':
    main()
