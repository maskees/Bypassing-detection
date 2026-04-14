"""
Test trained models and run evaluation to generate results for the web interface.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.target_model import MNISTNet, DetectorNet


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠ Using CPU")
    return device


def test_model_loading(device):
    """Test that all 4 models load correctly."""
    print("\n" + "=" * 50)
    print("STEP 1: Testing Model Loading")
    print("=" * 50)

    models = {}
    model_files = {
        'base': ('saved_models/base_model.pth', MNISTNet),
        'adv_trained': ('saved_models/adv_trained_model.pth', MNISTNet),
        'distilled': ('saved_models/distilled_model.pth', MNISTNet),
        'detector': ('saved_models/detector_model.pth', DetectorNet),
    }

    for name, (path, cls) in model_files.items():
        if not os.path.exists(path):
            print(f"  ✗ {name}: File not found at {path}")
            continue

        try:
            model = cls(in_channels=3, num_classes=4).to(device)
            state_dict = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            models[name] = model

            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {name}: Loaded ({params:,} params, {size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ✗ {name}: Failed to load — {e}")
            # Try to diagnose the issue
            try:
                state_dict = torch.load(path, map_location=device, weights_only=True)
                print(f"    State dict keys: {list(state_dict.keys())[:5]}...")
            except Exception as e2:
                print(f"    Cannot read file: {e2}")

    return models


def test_model_accuracy(models, test_loader, device):
    """Test clean accuracy of each model."""
    print("\n" + "=" * 50)
    print("STEP 2: Testing Clean Accuracy")
    print("=" * 50)

    for name, model in models.items():
        if name == 'detector':
            continue  # Detector is binary, not digit classifier

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
        print(f"  {name}: {acc:.2f}% ({correct}/{total})")


def test_attacks(models, test_loader, device):
    """Quick test of each attack method."""
    print("\n" + "=" * 50)
    print("STEP 3: Testing Attack Methods")
    print("=" * 50)

    base_model = models.get('base')
    if not base_model:
        print("  ✗ No base model — skipping attack tests")
        return

    # Get a small batch
    images, labels = next(iter(test_loader))
    images, labels = images[:10].to(device), labels[:10].to(device)
    epsilon = 0.3

    # Test FGSM
    try:
        from attacks.fgsm import fgsm_attack
        adv, pert, success = fgsm_attack(base_model, images, labels, epsilon, device)
        asr = success.float().mean().item() * 100
        print(f"  ✓ FGSM: ASR={asr:.0f}%, L∞={pert.abs().max():.4f}")
    except Exception as e:
        print(f"  ✗ FGSM failed: {e}")

    # Test PGD
    try:
        from attacks.pgd import pgd_attack
        adv, pert, success = pgd_attack(base_model, images, labels, epsilon, steps=10, device=device)
        asr = success.float().mean().item() * 100
        print(f"  ✓ PGD: ASR={asr:.0f}%, L∞={pert.abs().max():.4f}")
    except Exception as e:
        print(f"  ✗ PGD failed: {e}")

    # Test Genetic Algorithm (single image)
    try:
        from attacks.genetic_attack import genetic_attack
        img, lbl = images[0], labels[0].item()
        result = genetic_attack(base_model, img, lbl, epsilon,
                                pop_size=20, generations=20, device=device)
        print(f"  ✓ GA: Success={result['success']}, "
              f"L∞={result['l_inf']:.4f}, Time={result['time']:.2f}s")
    except Exception as e:
        print(f"  ✗ GA failed: {e}")

    # Test Differential Evolution (single image)
    try:
        from attacks.differential_evolution_attack import de_attack
        img, lbl = images[0], labels[0].item()
        result = de_attack(base_model, img, lbl, epsilon,
                           maxiter=20, device=device)
        print(f"  ✓ DE: Success={result['success']}, "
              f"L∞={result['l_inf']:.4f}, Time={result['time']:.2f}s")
    except Exception as e:
        print(f"  ✗ DE failed: {e}")


def run_evaluation(models, test_loader, device):
    """Run full evaluation and save results."""
    print("\n" + "=" * 50)
    print("STEP 4: Running Full Evaluation")
    print("=" * 50)
    print("  (4 attacks × 5 defenses on 50 images — EC attacks are slower)")

    try:
        from evaluation.evaluator import run_full_evaluation
        results = run_full_evaluation(
            models_dict=models,
            test_loader=test_loader,
            epsilon=0.3,
            num_samples=50,
            device=device,
            save_path='results/evaluation_results.json'
        )
        print("\n  ✓ Evaluation complete! Results saved to results/evaluation_results.json")
        return results
    except Exception as e:
        print(f"\n  ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    start_time = time.time()
    device = get_device()

    # Download MNIST data
    print("\nDownloading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.ImageFolder(root='./data/RoadSigns/test', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    print(f"✓ MNIST test set: {len(test_dataset)} images")

    # Step 1: Test model loading
    models = test_model_loading(device)

    if not models:
        print("\n✗ No models could be loaded. Check saved_models/ directory.")
        return

    # Step 2: Test accuracy
    test_model_accuracy(models, test_loader, device)

    # Step 3: Test attacks
    test_attacks(models, test_loader, device)

    # Step 4: Run evaluation
    run_evaluation(models, test_loader, device)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"ALL TESTS COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 50}")
    print("\nNext: Run 'python app.py' to start the web interface")


if __name__ == '__main__':
    main()
