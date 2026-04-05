"""
Evaluation Engine — Runs all attack-defense combinations and collects metrics.

Evaluates 4 attacks × 5 defenses (including no defense baseline) = 20 combinations.
Metrics collected per combination:
  - Clean Accuracy: Model accuracy on unperturbed images
  - Attack Success Rate: % of correctly classified images that become misclassified
  - Robust Accuracy: Model accuracy on adversarial images
  - Average L∞ perturbation
  - Average L2 perturbation
  - Average attack time
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import os

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.genetic_attack import genetic_attack
from attacks.differential_evolution_attack import de_attack
from defenses.input_transformation import apply_input_transforms
from defenses.detection_network import detect_and_predict


def evaluate_single_combination(attack_fn, model, images, labels, epsilon,
                                defense_type=None, defense_extras=None,
                                device='cuda'):
    """Evaluate a single attack against a single defense."""
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    results = {
        'attack_success_count': 0,
        'total_correct_clean': 0,
        'total_correct_adv': 0,
        'total_samples': 0,
        'l_inf_values': [],
        'l2_values': [],
        'times': [],
    }

    # ── Clean accuracy ──
    with torch.no_grad():
        if defense_type == 'input_transform':
            transformed = apply_input_transforms(images)
            clean_out = model(transformed)
        elif defense_type == 'detection':
            detector = defense_extras['detector']
            clean_preds, _, _ = detect_and_predict(model, detector, images, device)
            clean_correct = (clean_preds == labels).sum().item()
            results['total_correct_clean'] = clean_correct
            results['total_samples'] = labels.size(0)
        else:
            clean_out = model(images)

        if defense_type != 'detection':
            clean_preds = clean_out.argmax(dim=1)
            clean_correct = (clean_preds == labels).sum().item()
            results['total_correct_clean'] = clean_correct
            results['total_samples'] = labels.size(0)

    # ── Generate adversarial examples ──
    start = time.time()

    if attack_fn in ['fgsm', 'pgd']:
        # Gradient-based attacks (batched)
        if attack_fn == 'fgsm':
            adv_images, perturbations, _ = fgsm_attack(model, images, labels, epsilon, device)
        else:
            adv_images, perturbations, _ = pgd_attack(model, images, labels, epsilon, device=device)

        elapsed = time.time() - start
        results['times'].append(elapsed)
        results['l_inf_values'].append(perturbations.abs().max().item())
        results['l2_values'].append(perturbations.norm(2, dim=[1, 2, 3]).mean().item())
    else:
        # EC attacks (per-image, collect perturbations)
        adv_list = []
        for i in range(images.size(0)):
            img = images[i]
            lbl = labels[i].item()
            t0 = time.time()

            if attack_fn == 'genetic':
                res = genetic_attack(model, img, lbl, epsilon,
                                     pop_size=30, generations=50, device=device)
            else:
                res = de_attack(model, img, lbl, epsilon,
                                maxiter=50, device=device)

            adv_list.append(res['adversarial'])
            results['l_inf_values'].append(res['l_inf'])
            results['l2_values'].append(res['l2'])
            results['times'].append(time.time() - t0)

        adv_images = torch.stack(adv_list).to(device)

    # ── Evaluate defense on adversarial examples ──
    with torch.no_grad():
        if defense_type == 'input_transform':
            transformed_adv = apply_input_transforms(adv_images)
            adv_out = model(transformed_adv)
            adv_preds = adv_out.argmax(dim=1)
        elif defense_type == 'detection':
            detector = defense_extras['detector']
            adv_preds, is_detected, _ = detect_and_predict(
                model, detector, adv_images, device
            )
            # Detected adversarial = successfully defended
            # adv_preds == -1 means rejected
        else:
            adv_out = model(adv_images)
            adv_preds = adv_out.argmax(dim=1)

    # Count successes: attacks that changed prediction from correct to incorrect
    with torch.no_grad():
        if defense_type != 'detection':
            # Re-evaluate clean predictions with this model/defense
            if defense_type == 'input_transform':
                clean_out2 = model(apply_input_transforms(images))
            else:
                clean_out2 = model(images)
            clean_preds2 = clean_out2.argmax(dim=1)
        else:
            clean_preds2, _, _ = detect_and_predict(model, defense_extras['detector'], images, device)

    # For detection defense: -1 predictions count as "defended" (not misclassified)
    for i in range(labels.size(0)):
        true_lbl = labels[i].item()
        clean_pred = clean_preds2[i].item()
        adv_pred = adv_preds[i].item()

        if clean_pred == true_lbl:  # Only count images that were correctly classified
            if defense_type == 'detection' and adv_pred == -1:
                results['total_correct_adv'] += 1  # Successfully defended (rejected)
            elif adv_pred == true_lbl:
                results['total_correct_adv'] += 1  # Still correct
            else:
                results['attack_success_count'] += 1  # Attack succeeded

    return results


def run_full_evaluation(models_dict, test_loader, epsilon=0.3,
                        num_samples=100, device='cuda',
                        save_path='results/evaluation_results.json'):
    """
    Run comprehensive evaluation of all attack-defense combinations.

    Args:
        models_dict: Dictionary with keys:
            'base': Base model
            'adv_trained': Adversarially trained model
            'distilled': Distilled model
            'detector': Detection network
        test_loader: Test data loader
        epsilon: Attack epsilon
        num_samples: Number of test samples to evaluate
        device: Computation device
        save_path: Path to save results JSON

    Returns:
        results_matrix: Nested dict of results
    """
    print("\n" + "=" * 60)
    print("RUNNING FULL EVALUATION")
    print("=" * 60)

    attack_names = ['fgsm', 'pgd', 'genetic', 'de']
    attack_display = {
        'fgsm': 'FGSM (ML)',
        'pgd': 'PGD (ML)',
        'genetic': 'Genetic Algorithm (EC)',
        'de': 'Differential Evolution (EC)',
    }

    defense_configs = [
        ('none', 'No Defense', models_dict['base'], None),
        ('adv_training', 'Adversarial Training', models_dict['adv_trained'], None),
        ('input_transform', 'Input Transformation', models_dict['base'],
         {'type': 'input_transform'}),
        ('detection', 'Detection Network', models_dict['base'],
         {'type': 'detection', 'detector': models_dict['detector']}),
        ('distillation', 'Defensive Distillation', models_dict['distilled'], None),
    ]

    # Collect test samples
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
        if sum(x.size(0) for x in all_images) >= num_samples:
            break
    all_images = torch.cat(all_images)[:num_samples]
    all_labels = torch.cat(all_labels)[:num_samples]

    results_matrix = {}

    for atk_name in attack_names:
        results_matrix[atk_name] = {}
        print(f"\n── Attack: {attack_display[atk_name]} ──")

        for def_key, def_display, model, extras in defense_configs:
            print(f"  vs {def_display}...", end=" ", flush=True)

            defense_type = None
            defense_extras = None
            if extras:
                defense_type = extras.get('type')
                defense_extras = extras

            try:
                res = evaluate_single_combination(
                    atk_name, model, all_images, all_labels, epsilon,
                    defense_type=defense_type,
                    defense_extras=defense_extras,
                    device=device
                )

                total = res['total_samples']
                clean_correct = res['total_correct_clean']
                adv_correct = res['total_correct_adv']
                attack_success = res['attack_success_count']

                clean_acc = clean_correct / total * 100 if total > 0 else 0
                robust_acc = adv_correct / total * 100 if total > 0 else 0
                success_rate = attack_success / clean_correct * 100 if clean_correct > 0 else 0

                entry = {
                    'clean_accuracy': round(clean_acc, 2),
                    'robust_accuracy': round(robust_acc, 2),
                    'attack_success_rate': round(success_rate, 2),
                    'avg_l_inf': round(np.mean(res['l_inf_values']), 4) if res['l_inf_values'] else 0,
                    'avg_l2': round(np.mean(res['l2_values']), 4) if res['l2_values'] else 0,
                    'avg_time': round(np.mean(res['times']), 4) if res['times'] else 0,
                    'total_samples': total,
                }

                results_matrix[atk_name][def_key] = entry
                print(f"ASR={success_rate:.1f}%, Robust={robust_acc:.1f}%")

            except Exception as e:
                print(f"ERROR: {e}")
                results_matrix[atk_name][def_key] = {
                    'clean_accuracy': 0, 'robust_accuracy': 0,
                    'attack_success_rate': 0, 'error': str(e),
                }

    # ── Save results ──
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    output = {
        'epsilon': epsilon,
        'num_samples': num_samples,
        'attack_names': attack_display,
        'defense_names': {k: v for k, v, _, _ in defense_configs},
        'results': results_matrix,
    }

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {save_path}")
    return output
