"""
Train the Denoising Autoencoder defense.

Unsupervised training on clean road-sign images. During training, each
image is corrupted with synthetic adversarial-like noise (Gaussian +
uniform L_inf perturbation), and the autoencoder learns to reconstruct
the clean original.

Usage:
    python train_autoencoder.py
    python train_autoencoder.py --epochs 20 --batch-size 32 --lr 1e-3

Output:
    saved_models/road_sign_crop_autoencoder.pth
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.denoising_autoencoder import DenoisingAutoencoder
from road_sign_data import (
    DisplayTensorDataset,
    RoadSignCropDataset,
    load_records,
    load_records_imagefolder,
    stratified_split,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inject_noise(images, epsilon_max=0.15, gaussian_sigma=0.05):
    """
    Add adversarial-like noise to clean images.

    Mix of:
      1. Uniform L_inf perturbation up to a random epsilon in [0, epsilon_max]
      2. Gaussian noise with std up to gaussian_sigma
    The mix exposes the autoencoder to perturbations of varying strength
    so it generalizes across attack budgets at inference time.
    """
    batch_size = images.size(0)
    device = images.device

    # Per-sample random epsilon in [0, epsilon_max]
    eps = torch.rand(batch_size, 1, 1, 1, device=device) * epsilon_max
    uniform_noise = (torch.rand_like(images) * 2 - 1) * eps

    # Gaussian noise on top
    sigma = torch.rand(batch_size, 1, 1, 1, device=device) * gaussian_sigma
    gauss_noise = torch.randn_like(images) * sigma

    noisy = images + uniform_noise + gauss_noise
    return noisy.clamp(0.0, 1.0)


def build_data_loaders(batch_size, num_workers=2):
    """Build train/val data loaders of display-space images in [0,1]."""
    gtsrb_dir = "data/GTSRB_mapped"
    image_size = 224

    if os.path.exists(gtsrb_dir):
        print(f"Using GTSRB mapped dataset: {gtsrb_dir}")
        train_records = load_records_imagefolder(gtsrb_dir, split="train")
        val_records = load_records_imagefolder(gtsrb_dir, split="test")
    else:
        print("GTSRB not found — using original cropped dataset (annotations/images)")
        records = load_records()
        train_records, val_records = stratified_split(records, val_ratio=0.2, seed=42)

    train_ds = RoadSignCropDataset(train_records, image_size=image_size,
                                    augment=True, return_display=True)
    val_ds = RoadSignCropDataset(val_records, image_size=image_size,
                                  augment=False, return_display=True)

    train_ds = DisplayTensorDataset(train_ds)
    val_ds = DisplayTensorDataset(val_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}", flush=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, device, epsilon_max=0.15,
                    clean_weight=0.2):
    """One epoch of denoising training.

    Total loss = MSE(recon(noisy), clean) + clean_weight * MSE(recon(clean), clean)
    The clean reconstruction term is a regularizer so the model doesn't
    distort inputs that arrive clean.
    """
    model.train()
    total_loss = 0.0
    count = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)

        noisy = inject_noise(images, epsilon_max=epsilon_max)

        optimizer.zero_grad()
        recon_noisy = model(noisy)
        loss_denoise = F.mse_loss(recon_noisy, images)

        if clean_weight > 0:
            recon_clean = model(images)
            loss_identity = F.mse_loss(recon_clean, images)
            loss = loss_denoise + clean_weight * loss_identity
        else:
            loss = loss_denoise

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        count += images.size(0)

    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device, epsilon_max=0.15):
    """Return (clean_mse, noisy_mse, psnr_noisy) on the val set."""
    model.eval()
    clean_mse = 0.0
    noisy_mse = 0.0
    count = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        noisy = inject_noise(images, epsilon_max=epsilon_max)

        recon_clean = model(images)
        recon_noisy = model(noisy)

        clean_mse += F.mse_loss(recon_clean, images, reduction='sum').item()
        noisy_mse += F.mse_loss(recon_noisy, images, reduction='sum').item()
        count += images.numel()

    clean_mse /= max(count, 1)
    noisy_mse /= max(count, 1)
    # PSNR on [0,1] images: 10 * log10(1 / MSE)
    psnr = 10.0 * np.log10(1.0 / max(noisy_mse, 1e-12))
    return clean_mse, noisy_mse, psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--epsilon-max', type=float, default=0.15,
                        help="Max L_inf noise injected during training")
    parser.add_argument('--clean-weight', type=float, default=0.0,
                        help="Weight on clean-identity reconstruction loss (0=off, saves VRAM)")
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str,
                        default='saved_models/road_sign_crop_autoencoder.pth')
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)

    train_loader, val_loader = build_data_loaders(args.batch_size, args.num_workers)

    model = DenoisingAutoencoder(base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DenoisingAutoencoder(base_channels={args.base_channels}) "
          f"— {n_params / 1e6:.2f}M params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = -float('inf')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("\n[Defense] Training Denoising Autoencoder")
    print(f"  epsilon_max={args.epsilon_max}, clean_weight={args.clean_weight}, "
          f"epochs={args.epochs}, lr={args.lr}", flush=True)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            epsilon_max=args.epsilon_max, clean_weight=args.clean_weight,
        )
        clean_mse, noisy_mse, psnr = evaluate(
            model, val_loader, device, epsilon_max=args.epsilon_max
        )
        scheduler.step()

        print(f"  Epoch {epoch+1:02d}/{args.epochs} — "
              f"train_loss={train_loss:.5f} | "
              f"val clean_mse={clean_mse:.5f} noisy_mse={noisy_mse:.5f} PSNR={psnr:.2f}dB",
              flush=True)

        if psnr > best_psnr:
            best_psnr = psnr
            torch.save({
                'state_dict': model.state_dict(),
                'base_channels': args.base_channels,
                'epsilon_max_train': args.epsilon_max,
                'metrics': {
                    'val_clean_mse': clean_mse,
                    'val_noisy_mse': noisy_mse,
                    'val_psnr': psnr,
                    'train_loss': train_loss,
                    'epoch': epoch + 1,
                },
            }, args.output)
            print(f"    Saved new best → {args.output}", flush=True)

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB", flush=True)
    print(f"Checkpoint: {args.output}", flush=True)


if __name__ == '__main__':
    main()
