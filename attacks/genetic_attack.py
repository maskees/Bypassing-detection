"""
Genetic Algorithm Attack — Evolutionary Computation, gradient-free.

A population-based, black-box adversarial attack that evolves perturbation
vectors using selection, crossover, and mutation operators.

Key Advantage: Does NOT require gradient access — treats the model as
a black box and only queries its output predictions/confidence scores.
This can bypass gradient-masking defenses.

Operators:
  - Fitness: 1 - P(true_class) — maximize misclassification confidence
  - Selection: Tournament selection (size 3)
  - Crossover: Uniform crossover (rate 0.7)
  - Mutation: Gaussian noise on random pixels (rate 0.3)
"""

import numpy as np
import torch
import torch.nn.functional as F
import time


def genetic_attack(model, image, label, epsilon, pop_size=50, generations=100,
                   crossover_rate=0.7, mutation_rate=0.3, tournament_size=3,
                   device='cuda'):
    """
    Generate an adversarial example using a Genetic Algorithm.

    Args:
        model: Target classifier (used as black-box — only output queried)
        image: Single clean image tensor (C, H, W), values in [0, 1]
        label: True label (int)
        epsilon: L∞ perturbation bound
        pop_size: Population size
        generations: Max number of generations
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation per individual
        tournament_size: Tournament selection size
        device: Computation device

    Returns:
        dict with adversarial image, success flag, generation count, etc.
    """
    model.eval()
    start_time = time.time()

    image = image.to(device)
    flat_size = image.numel()  # 1*28*28 = 784 for MNIST
    img_shape = image.shape

    # ── Initialize population: random perturbations in [-ε, ε] ──
    population = np.random.uniform(-epsilon, epsilon,
                                   (pop_size, flat_size)).astype(np.float32)

    best_adv = None
    best_fitness = -1.0
    queries = 0

    def evaluate_fitness(pop_array):
        """Evaluate fitness for entire population (batched for GPU speed)."""
        nonlocal queries
        pert_tensor = torch.tensor(pop_array, device=device).view(-1, *img_shape)
        adv_batch = torch.clamp(image.unsqueeze(0) + pert_tensor, 0.0, 1.0)

        with torch.no_grad():
            outputs = model(adv_batch)
            probs = F.softmax(outputs, dim=1)
            # Fitness = 1 - P(true_class). Higher = better attack.
            correct_probs = probs[:, label].cpu().numpy()
            fitness = 1.0 - correct_probs
            preds = outputs.argmax(dim=1).cpu().numpy()

        queries += len(pop_array)
        return fitness, preds

    for gen in range(generations):
        fitness, preds = evaluate_fitness(population)

        # ── Track best individual ──
        gen_best_idx = np.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_pert = population[gen_best_idx].copy()

        # ── Early stopping if attack succeeded ──
        success_indices = np.where(preds != label)[0]
        if len(success_indices) > 0:
            # Pick the one with highest fitness among successful
            best_success = success_indices[np.argmax(fitness[success_indices])]
            best_pert = population[best_success]
            elapsed = time.time() - start_time
            adv_tensor = torch.clamp(
                image + torch.tensor(best_pert, device=device).view(img_shape),
                0.0, 1.0
            )
            return _build_result(model, image, adv_tensor, label, True,
                                 gen + 1, queries, elapsed, device)

        # ── Selection: Tournament ──
        new_pop = []
        for _ in range(pop_size):
            candidates = np.random.choice(pop_size, tournament_size, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            new_pop.append(population[winner].copy())

        # ── Crossover: Uniform ──
        for i in range(0, pop_size - 1, 2):
            if np.random.random() < crossover_rate:
                mask = np.random.random(flat_size) < 0.5
                child1 = new_pop[i].copy()
                child2 = new_pop[i + 1].copy()
                child1[mask] = new_pop[i + 1][mask]
                child2[mask] = new_pop[i][mask]
                new_pop[i] = child1
                new_pop[i + 1] = child2

        # ── Mutation: Gaussian noise on random pixels ──
        for i in range(pop_size):
            if np.random.random() < mutation_rate:
                # Mutate ~10% of pixels
                pixel_mask = np.random.random(flat_size) < 0.1
                noise = np.random.normal(0, epsilon * 0.3, flat_size).astype(np.float32)
                new_pop[i][pixel_mask] += noise[pixel_mask]
                # Clip to epsilon bound
                new_pop[i] = np.clip(new_pop[i], -epsilon, epsilon)

        # ── Elitism: Keep the best individual ──
        new_pop[0] = best_pert.copy()
        population = np.array(new_pop)

    # ── Return best result after all generations ──
    elapsed = time.time() - start_time
    adv_tensor = torch.clamp(
        image + torch.tensor(best_pert, device=device).view(img_shape),
        0.0, 1.0
    )
    with torch.no_grad():
        pred = model(adv_tensor.unsqueeze(0)).argmax(1).item()
    success = pred != label

    return _build_result(model, image, adv_tensor, label, success,
                         generations, queries, elapsed, device)


def _build_result(model, original, adversarial, label, success,
                  generations_used, queries, elapsed, device):
    """Build result dictionary with all info needed for visualization."""
    perturbation = adversarial - original
    with torch.no_grad():
        orig_output = model(original.unsqueeze(0))
        adv_output = model(adversarial.unsqueeze(0))
        orig_probs = F.softmax(orig_output, dim=1)[0]
        adv_probs = F.softmax(adv_output, dim=1)[0]

    return {
        'original': original.detach().cpu(),
        'adversarial': adversarial.detach().cpu(),
        'perturbation': perturbation.detach().cpu(),
        'orig_pred': orig_output.argmax(1).item(),
        'adv_pred': adv_output.argmax(1).item(),
        'orig_probs': orig_probs.detach().cpu().numpy(),
        'adv_probs': adv_probs.detach().cpu().numpy(),
        'success': success,
        'true_label': label,
        'l_inf': perturbation.abs().max().item(),
        'l2': perturbation.norm(2).item(),
        'generations': generations_used,
        'queries': queries,
        'time': elapsed,
        'attack_type': 'Genetic Algorithm',
    }
