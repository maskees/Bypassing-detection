import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.road_sign_model import RoadSignResNet, road_sign_loss
from road_sign_data import CLASS_NAMES, make_road_sign_datasets


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def collate_batch(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "bbox": torch.stack([item["bbox"] for item in batch]),
    }


def accuracy_from_logits(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def class_counts(dataset):
    counts = {idx: 0 for idx in range(len(CLASS_NAMES))}
    for record in dataset.records:
        counts[record.label] += 1
    return counts


def make_class_weights(counts, device):
    total = sum(counts.values())
    weights = []
    for idx in range(len(CLASS_NAMES)):
        weights.append(total / max(counts[idx], 1))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights / weights.mean()


def make_balanced_sampler(dataset):
    counts = class_counts(dataset)
    sample_weights = [1.0 / max(counts[record.label], 1) for record in dataset.records]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def run_epoch(model, loader, optimizer, device, bbox_weight, class_weights, train):
    model.train(train)
    total_loss = 0.0
    total_class_loss = 0.0
    total_bbox_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            boxes = batch["bbox"].to(device)

            logits, pred_boxes = model.forward_with_bbox(images)
            loss, class_loss, bbox_loss = road_sign_loss(
                logits,
                pred_boxes,
                labels,
                boxes,
                bbox_weight=bbox_weight,
                class_weights=class_weights,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_class_loss += class_loss.item() * batch_size
            total_bbox_loss += bbox_loss.item() * batch_size
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "class_loss": total_class_loss / total_count,
        "bbox_loss": total_bbox_loss / total_count,
        "accuracy": total_correct / total_count,
    }


def save_checkpoint(path, model, epoch, metrics, args):
    os.makedirs(Path(path).parent, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "class_names": CLASS_NAMES,
            "config": {
                "num_classes": len(CLASS_NAMES),
                "backbone": args.backbone,
                "image_size": args.image_size,
                "bbox_weight": args.bbox_weight,
            },
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train the road-sign ResNet classifier/bbox model.")
    parser.add_argument("--annotations-dir", default="annotations")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--output", default="saved_models/road_sign_resnet34.pth")
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet34")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bbox-weight", type=float, default=5.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds, val_ds = make_road_sign_datasets(
        annotations_dir=args.annotations_dir,
        images_dir=args.images_dir,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Classes: {CLASS_NAMES}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_ds),
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device == "cuda",
    )

    model = RoadSignResNet(
        num_classes=len(CLASS_NAMES),
        backbone=args.backbone,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    weights = make_class_weights(class_counts(train_ds), device)
    print(f"Class counts: {class_counts(train_ds)}")
    print(f"Class weights: {[round(x, 3) for x in weights.detach().cpu().tolist()]}")

    best_acc = -1.0
    best_metrics = None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            bbox_weight=args.bbox_weight,
            class_weights=weights,
            train=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            bbox_weight=args.bbox_weight,
            class_weights=weights,
            train=False,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_acc={train_metrics['accuracy'] * 100:5.1f}% "
            f"val_acc={val_metrics['accuracy'] * 100:5.1f}% "
            f"val_loss={val_metrics['loss']:.4f} "
            f"bbox={val_metrics['bbox_loss']:.4f}"
        )

        if val_metrics["accuracy"] >= best_acc:
            best_acc = val_metrics["accuracy"]
            best_metrics = {
                "train": train_metrics,
                "validation": val_metrics,
            }
            save_checkpoint(args.output, model, epoch, best_metrics, args)
            print(f"  saved best checkpoint -> {args.output}")

    elapsed = time.time() - start
    metrics_path = Path("results") / "road_sign_training_metrics.json"
    metrics_path.parent.mkdir(exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(
            {
                "best_accuracy": best_acc,
                "best_metrics": best_metrics,
                "class_names": CLASS_NAMES,
                "output": args.output,
                "elapsed_seconds": elapsed,
                "args": vars(args),
            },
            f,
            indent=2,
        )

    print(f"Best validation accuracy: {best_acc * 100:.2f}%")
    print(f"Training metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
