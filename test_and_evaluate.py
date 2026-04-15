"""
test_and_evaluate.py — Standalone evaluation script.

Loads pre-trained models from saved_models/ and runs the full
4-attack × 5-defense evaluation matrix. Use this to re-evaluate
without retraining.

Usage:
    python test_and_evaluate.py
    python test_and_evaluate.py --samples 100 --epsilon 0.1
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn.functional as F

from models.target_model import TrafficNet, DetectorNet
from models.data_utils import get_test_loader
from evaluation.evaluator import run_full_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate adversarial robustness of trained models.')
    parser.add_argument('--samples', type=int, default=50, help='Number of test samples (default: 50)')
    parser.add_argument('--epsilon', type=float, default=0.3, help='L-inf perturbation bound (default: 0.3)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    return parser.parse_args()


def load_model(cls, path, device):
    """Load a model from a checkpoint file."""
    if not os.path.exists(path):
        print(f"  ❌ NOT FOUND: {path}")
        return None
    model = cls().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✅ {path:<42} ({size_kb:.0f} KB)")
    return model


def compute_accuracy(model, test_loader, device):
    """Compute clean accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── Load models ──
    print(f"\n📦 Loading models...")
    models = {}

    base = load_model(TrafficNet, 'saved_models/base_model.pth', device)
    adv = load_model(TrafficNet, 'saved_models/adv_trained_model.pth', device)
    distilled = load_model(TrafficNet, 'saved_models/distilled_model.pth', device)
    detector = load_model(DetectorNet, 'saved_models/detector_model.pth', device)

    if any(m is None for m in [base, adv, distilled, detector]):
        print("\n❌ Some models are missing. Run 'python train_all.py' first.")
        sys.exit(1)

    models = {
        'base': base,
        'adv_trained': adv,
        'distilled': distilled,
        'detector': detector,
    }

    # ── Load test data ──
    test_loader, test_dataset = get_test_loader(batch_size=args.batch_size, num_workers=2)
    print(f"\n📊 Test dataset: {len(test_dataset)} images")

    # ── Clean accuracy ──
    print(f"\n{'─' * 50}")
    print(f"  CLEAN ACCURACY (no attack)")
    print(f"{'─' * 50}")
    for name, model in [('Base CNN', base), ('Adversarially Trained', adv), ('Distilled', distilled)]:
        acc = compute_accuracy(model, test_loader, device)
        print(f"  {name:<25} {acc:.2f}%")

    # ── Full evaluation ──
    print(f"\n🔬 Running full evaluation ({args.samples} samples, ε={args.epsilon})...")
    start = time.time()

    results = run_full_evaluation(
        models_dict=models,
        test_loader=test_loader,
        epsilon=args.epsilon,
        num_samples=args.samples,
        device=device,
        save_path='results/evaluation_results.json',
    )

    elapsed = time.time() - start

    # ── Print results ──
    attack_display = results['attack_names']
    defense_display = results['defense_names']
    matrix = results['results']

    print(f"\n{'═' * 90}")
    print(f"  ATTACK SUCCESS RATE (%) — Lower = better defense")
    print(f"{'═' * 90}")

    header = f"{'Attack':<28}"
    for dk in defense_display:
        header += f"{defense_display[dk]:>14}"
    print(header)
    print("─" * 90)

    for ak in attack_display:
        row = f"{attack_display[ak]:<28}"
        for dk in defense_display:
            val = matrix.get(ak, {}).get(dk, {})
            asr = val.get('attack_success_rate', None)
            if asr is not None:
                row += f"{asr:>13.1f}%"
            else:
                row += f"{'ERR':>14}"
        print(row)

    print(f"{'═' * 90}")

    print(f"\n{'═' * 90}")
    print(f"  ROBUST ACCURACY (%) — Higher = better defense")
    print(f"{'═' * 90}")

    header = f"{'Attack':<28}"
    for dk in defense_display:
        header += f"{defense_display[dk]:>14}"
    print(header)
    print("─" * 90)

    for ak in attack_display:
        row = f"{attack_display[ak]:<28}"
        for dk in defense_display:
            val = matrix.get(ak, {}).get(dk, {})
            rob = val.get('robust_accuracy', None)
            if rob is not None:
                row += f"{rob:>13.1f}%"
            else:
                row += f"{'ERR':>14}"
        print(row)

    print(f"{'═' * 90}")
    print(f"\n✅ Evaluation completed in {elapsed / 60:.1f} minutes")
    print(f"💾 Results saved to: results/evaluation_results.json")
    print(f"\n🚀 Launch the dashboard:")
    print(f"   python app.py")


if __name__ == '__main__':
    main()
