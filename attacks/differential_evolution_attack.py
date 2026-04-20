"""
Differential Evolution Attack — Evolutionary Computation, gradient-free.

Uses scipy's Differential Evolution optimizer to find adversarial perturbations
in a reduced-dimensional space. The image is divided into blocks, and a single
perturbation value is optimized per block, then upscaled to full resolution.

Key Advantage: Continuous optimization, gradient-free. Effective against
gradient-masking defenses and non-differentiable models.

Dimensionality Reduction:
  - 28×28 MNIST image → 7×7 = 49 optimization variables
  - Each variable controls perturbation for a 4×4 pixel block
  - Final perturbation is upscaled via nearest-neighbor interpolation
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import differential_evolution
import time


def de_attack(model, image, label, epsilon, maxiter=100, popsize=15,
              reduce_size=7, device='cuda'):
    """
    Generate an adversarial example using Differential Evolution.

    Args:
        model: Target classifier (black-box)
        image: Single clean image tensor (C, H, W), values in [0, 1]
        label: True label (int)
        epsilon: L∞ perturbation bound
        maxiter: Max DE iterations
        popsize: Population size multiplier for DE
        reduce_size: Size of reduced perturbation grid (reduce_size × reduce_size)
        device: Computation device

    Returns:
        dict with adversarial image, success flag, etc.
    """
    model.eval()
    start_time = time.time()

    image_dev = image.to(device)
    img_shape = image.shape  
    C, H, W = img_shape
    queries = [0]  # Use list for mutability in closure

    # Scale factor for upsampling
    scale_h = H // reduce_size
    scale_w = W // reduce_size

    def make_perturbation(z):
        """Convert reduced-dim vector to full perturbation."""
        z_2d = z.reshape(C, reduce_size, reduce_size)
        # Upscale using nearest neighbor (repeat pixels)
        pert = np.repeat(np.repeat(z_2d, scale_h, axis=1), scale_w, axis=2)
        # Handle any size mismatch
        pert = pert[:, :H, :W]
        return pert

    def objective(z):
        """Objective: minimize probability of correct class."""
        queries[0] += 1
        pert = make_perturbation(z)
        adv = np.clip(image.cpu().numpy() + pert, 0.0, 1.0)
        adv_tensor = torch.tensor(adv, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            output = model(adv_tensor)
            probs = F.softmax(output, dim=1)
            # Minimize P(true_class) → the DE minimizes the objective
            return probs[0, label].item()

    # ── Run Differential Evolution ──
    num_vars = C * reduce_size * reduce_size  # 1 * 7 * 7 = 49 for MNIST
    bounds = [(-epsilon, epsilon)] * num_vars

    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        seed=42,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=False,
    )

    elapsed = time.time() - start_time

    # ── Reconstruct best adversarial example ──
    best_pert = make_perturbation(result.x)
    adv_np = np.clip(image.cpu().numpy() + best_pert, 0.0, 1.0)
    adv_tensor = torch.tensor(adv_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        orig_output = model(image_dev.unsqueeze(0))
        adv_output = model(adv_tensor.unsqueeze(0))
        orig_probs = F.softmax(orig_output, dim=1)[0]
        adv_probs = F.softmax(adv_output, dim=1)[0]
        adv_pred = adv_output.argmax(1).item()

    perturbation = adv_tensor - image_dev
    success = adv_pred != label

    return {
        'original': image.detach().cpu(),
        'adversarial': adv_tensor.detach().cpu(),
        'perturbation': perturbation.detach().cpu(),
        'orig_pred': orig_output.argmax(1).item(),
        'adv_pred': adv_pred,
        'orig_probs': orig_probs.detach().cpu().numpy(),
        'adv_probs': adv_probs.detach().cpu().numpy(),
        'success': success,
        'true_label': label,
        'l_inf': perturbation.abs().max().item(),
        'l2': perturbation.norm(2).item(),
        'de_result_fun': float(result.fun),
        'queries': queries[0],
        'time': elapsed,
        'attack_type': 'Differential Evolution',
    }
