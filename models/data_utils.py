"""
Data loading utilities for Indian Traffic Sign dataset.

Dataset structure:
  data/traffic_Data/DATA/{class_id}/  — training images (ImageFolder)
  data/traffic_Data/TEST/             — test images (flat, filenames encode class)
  data/labels.csv                     — class ID → name mapping
"""

import os
import re
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

IMG_SIZE = 32

# Standard transforms: resize to 32x32, convert to tensor
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class TrafficTestDataset(Dataset):
    """
    Custom dataset for the flat TEST directory.
    Filenames are like: {classId:03d}_{imageId}_j.png
    Skip duplicates with '_1_' pattern.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or TEST_TRANSFORM
        self.samples = []  # list of (filepath, label)

        # Pattern: 3-digit class ID, then underscore, then image number
        # Skip files with '_1_' which are duplicates
        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith('.png') and not fname.lower().endswith('.jpg'):
                continue
            if '_1_' in fname:
                continue  # Skip duplicates

            # Extract class ID from first 3 chars
            match = re.match(r'^(\d{3})_', fname)
            if match:
                class_id = int(match.group(1))
                self.samples.append((os.path.join(root_dir, fname), class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def load_label_names(csv_path='data/labels.csv'):
    """Load class ID to name mapping from CSV."""
    names = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row['ClassId'])
            names[cid] = row['Name'].strip()
    return names


def get_train_loader(batch_size=64, num_workers=0):
    """Get training data loader using ImageFolder on DATA/."""
    train_dataset = datasets.ImageFolder(
        root='data/traffic_Data/DATA',
        transform=TRAIN_TRANSFORM,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, train_dataset


def get_test_loader(batch_size=64, num_workers=0):
    """Get test data loader from flat TEST directory."""
    test_dataset = TrafficTestDataset(
        root_dir='data/traffic_Data/TEST',
        transform=TEST_TRANSFORM,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return test_loader, test_dataset


def get_data_loaders(batch_size=64, num_workers=0):
    """Get both train and test loaders."""
    train_loader, train_dataset = get_train_loader(batch_size, num_workers)
    test_loader, test_dataset = get_test_loader(batch_size, num_workers)
    print(f"✅ Traffic Sign dataset loaded")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Test samples:     {len(test_dataset):,}")
    print(f"   Batch size:       {batch_size}")
    return train_loader, test_loader, train_dataset, test_dataset
