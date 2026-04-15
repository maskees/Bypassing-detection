import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.road_sign_classifier import load_road_sign_classifier_checkpoint
from road_sign_data import CLASS_NAMES, make_road_sign_crop_datasets
from train_crop_classifier import collate_batch


def main():
    parser = argparse.ArgumentParser(description="Evaluate a cropped road-sign classifier.")
    parser.add_argument("--checkpoint", default="saved_models/road_sign_crop_resnet34.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="results/road_sign_crop_eval.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_ds = make_road_sign_crop_datasets(return_display=False)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    model, checkpoint = load_road_sign_classifier_checkpoint(args.checkpoint, device=device)
    model.eval()

    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.long)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
                confusion[true_label, pred_label] += 1

    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        class_total = confusion[idx].sum().item()
        class_correct = confusion[idx, idx].item()
        per_class[name] = class_correct / class_total if class_total else 0.0

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "class_names": CLASS_NAMES,
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
        "checkpoint_metrics": checkpoint.get("metrics"),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(f"Accuracy: {result['accuracy'] * 100:.2f}% ({correct}/{total})")
    for name, acc in per_class.items():
        print(f"  {name}: {acc * 100:.2f}%")
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
