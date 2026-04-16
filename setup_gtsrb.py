"""
Download GTSRB (German Traffic Sign Recognition Benchmark) and map to 4 classes:
    crosswalk, speedlimit, stop, trafficlight

GTSRB has 43 classes with ~39,000 train + ~12,600 test images.
We map relevant classes and save as ImageFolder structure.

Usage:
    python setup_gtsrb.py

Output:
    data/GTSRB_mapped/train/{crosswalk,speedlimit,stop,trafficlight}/
    data/GTSRB_mapped/test/{crosswalk,speedlimit,stop,trafficlight}/
"""

import os
import shutil
from pathlib import Path
from collections import Counter

from torchvision.datasets import GTSRB
from PIL import Image

# ── GTSRB class → our 4 classes ──
# Mapping based on visual similarity to our road sign categories
GTSRB_MAPPING = {
    # speedlimit: all speed limit signs
    0: "speedlimit",   # Speed limit (20km/h)
    1: "speedlimit",   # Speed limit (30km/h)
    2: "speedlimit",   # Speed limit (50km/h)
    3: "speedlimit",   # Speed limit (60km/h)
    4: "speedlimit",   # Speed limit (70km/h)
    5: "speedlimit",   # Speed limit (80km/h)
    7: "speedlimit",   # Speed limit (100km/h)
    8: "speedlimit",   # Speed limit (120km/h)

    # stop
    14: "stop",         # Stop sign

    # trafficlight
    26: "trafficlight", # Traffic signals

    # crosswalk / pedestrian signs
    27: "crosswalk",    # Pedestrians
    28: "crosswalk",    # Children crossing
}

OUTPUT_DIR = Path("data/GTSRB_mapped")
IMAGE_SIZE = 224
MAX_PER_CLASS = 2000  # Cap to balance classes


def setup():
    print("Downloading GTSRB dataset (this may take a few minutes)...")

    # Download train and test splits
    train_gtsrb = GTSRB(root="data/gtsrb_raw", split="train", download=True)
    test_gtsrb = GTSRB(root="data/gtsrb_raw", split="test", download=True)

    for split_name, dataset in [("train", train_gtsrb), ("test", test_gtsrb)]:
        print(f"\nProcessing {split_name} split ({len(dataset)} total images)...")

        # Count per class for balancing
        class_counts = Counter()
        saved = 0
        skipped = 0

        for idx in range(len(dataset)):
            img, gtsrb_label = dataset[idx]

            # Skip classes we don't map
            if gtsrb_label not in GTSRB_MAPPING:
                skipped += 1
                continue

            our_class = GTSRB_MAPPING[gtsrb_label]

            # Cap per class to keep balanced
            if class_counts[our_class] >= MAX_PER_CLASS:
                continue

            # Save image
            out_dir = OUTPUT_DIR / split_name / our_class
            out_dir.mkdir(parents=True, exist_ok=True)

            # Resize to 224x224
            if isinstance(img, Image.Image):
                img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            
            img.save(out_dir / f"{our_class}_{gtsrb_label}_{idx:06d}.png")
            class_counts[our_class] += 1
            saved += 1

            if saved % 500 == 0:
                print(f"  Saved {saved} images...")

        print(f"  {split_name}: saved {saved}, skipped {skipped}")
        for cls, cnt in sorted(class_counts.items()):
            print(f"    {cls}: {cnt}")

    print(f"\nDataset ready at: {OUTPUT_DIR}")
    print(f"Total size: {sum(1 for _ in OUTPUT_DIR.rglob('*.png'))} images")
    print("\nTo use in training, change train_models.ipynb cell 4:")
    print('  records = load_records_imagefolder("data/GTSRB_mapped")')


if __name__ == "__main__":
    setup()
