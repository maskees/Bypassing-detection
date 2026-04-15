import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


CLASS_NAMES = ["crosswalk", "speedlimit", "stop", "trafficlight"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class RoadSignRecord:
    image_path: Path
    label: int
    bbox: tuple[float, float, float, float]


def read_annotation(xml_path: Path, images_dir: Path) -> RoadSignRecord | None:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    class_name = root.findtext("./object/name")
    if not filename or class_name not in CLASS_TO_IDX:
        return None

    image_path = images_dir / filename
    if not image_path.exists():
        return None

    box = root.find("./object/bndbox")
    if box is None:
        return None

    xmin = float(box.findtext("xmin", "0"))
    ymin = float(box.findtext("ymin", "0"))
    xmax = float(box.findtext("xmax", "0"))
    ymax = float(box.findtext("ymax", "0"))
    if xmax <= xmin or ymax <= ymin:
        return None

    return RoadSignRecord(
        image_path=image_path,
        label=CLASS_TO_IDX[class_name],
        bbox=(xmin, ymin, xmax, ymax),
    )


def load_records(annotations_dir="annotations", images_dir="images"):
    annotations_path = Path(annotations_dir)
    images_path = Path(images_dir)
    records = []
    for xml_path in sorted(annotations_path.glob("*.xml")):
        record = read_annotation(xml_path, images_path)
        if record is not None:
            records.append(record)
    return records


def stratified_split(records, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    by_label: dict[int, list[RoadSignRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)

    train_records = []
    val_records = []
    for label_records in by_label.values():
        label_records = label_records[:]
        rng.shuffle(label_records)
        val_count = max(1, int(round(len(label_records) * val_ratio)))
        val_records.extend(label_records[:val_count])
        train_records.extend(label_records[val_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def _resize_bbox(bbox, original_size, image_size):
    width, height = original_size
    xmin, ymin, xmax, ymax = bbox
    scale_x = image_size / width
    scale_y = image_size / height
    return torch.tensor(
        [xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y],
        dtype=torch.float32,
    )


def _normalize_bbox_pixels(bbox, image_size):
    return bbox / float(image_size)


class RoadSignFullImageDataset(Dataset):
    def __init__(self, records, image_size=224, augment=False, return_display=False):
        self.records = list(records)
        self.image_size = image_size
        self.augment = augment
        self.return_display = return_display
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        original_size = image.size
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        bbox = _resize_bbox(record.bbox, original_size, self.image_size)

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                xmin, ymin, xmax, ymax = bbox.tolist()
                bbox = torch.tensor(
                    [self.image_size - xmax, ymin, self.image_size - xmin, ymax],
                    dtype=torch.float32,
                )
            angle = random.uniform(-12, 12)
            image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)

        display_tensor = TF.to_tensor(image)
        model_tensor = self.normalize(display_tensor)
        bbox = _normalize_bbox_pixels(bbox, self.image_size)

        result = {
            "image": model_tensor,
            "label": torch.tensor(record.label, dtype=torch.long),
            "bbox": bbox,
            "display_image": display_tensor,
            "path": str(record.image_path),
        }
        if not self.return_display:
            result.pop("display_image")
        return result


class RoadSignTensorDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        return item["image"], item["label"]


class DisplayTensorDataset(Dataset):
    """Return (display_image, label) tuples with images in [0, 1] range.

    Used for defense training where adversarial attacks operate on
    unnormalized display images and the model is wrapped in NormalizedModel.
    Requires the base dataset to be created with ``return_display=True``.
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        return item["display_image"], item["label"]


def padded_crop_box(bbox, image_size, padding_ratio=0.2):
    width, height = image_size
    xmin, ymin, xmax, ymax = bbox
    box_width = xmax - xmin
    box_height = ymax - ymin
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    return (
        max(0, int(round(xmin - pad_x))),
        max(0, int(round(ymin - pad_y))),
        min(width, int(round(xmax + pad_x))),
        min(height, int(round(ymax + pad_y))),
    )


class RoadSignCropDataset(Dataset):
    """Classification dataset that crops signs from full images using XML boxes."""

    def __init__(
        self,
        records,
        image_size=224,
        padding_ratio=0.2,
        augment=False,
        return_display=False,
    ):
        self.records = list(records)
        self.image_size = image_size
        self.padding_ratio = padding_ratio
        self.augment = augment
        self.return_display = return_display
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        crop_box = padded_crop_box(record.bbox, image.size, self.padding_ratio)
        image = image.crop(crop_box)

        if self.augment:
            image = transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.2,
                hue=0.03,
            )(image)
            angle = random.uniform(-12, 12)
            image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)

        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        display_tensor = TF.to_tensor(image)
        model_tensor = self.normalize(display_tensor)

        result = {
            "image": model_tensor,
            "label": torch.tensor(record.label, dtype=torch.long),
            "display_image": display_tensor,
            "path": str(record.image_path),
            "crop_box": torch.tensor(crop_box, dtype=torch.float32),
        }
        if not self.return_display:
            result.pop("display_image")
        return result


def make_road_sign_datasets(
    annotations_dir="annotations",
    images_dir="images",
    image_size=224,
    val_ratio=0.2,
    seed=42,
    return_display=False,
):
    records = load_records(annotations_dir, images_dir)
    train_records, val_records = stratified_split(records, val_ratio=val_ratio, seed=seed)
    train_ds = RoadSignFullImageDataset(train_records, image_size=image_size, augment=True)
    val_ds = RoadSignFullImageDataset(
        val_records,
        image_size=image_size,
        augment=False,
        return_display=return_display,
    )
    return train_ds, val_ds


def make_road_sign_crop_datasets(
    annotations_dir="annotations",
    images_dir="images",
    image_size=224,
    padding_ratio=0.2,
    val_ratio=0.2,
    seed=42,
    return_display=False,
):
    records = load_records(annotations_dir, images_dir)
    train_records, val_records = stratified_split(records, val_ratio=val_ratio, seed=seed)
    train_ds = RoadSignCropDataset(
        train_records,
        image_size=image_size,
        padding_ratio=padding_ratio,
        augment=True,
    )
    val_ds = RoadSignCropDataset(
        val_records,
        image_size=image_size,
        padding_ratio=padding_ratio,
        augment=False,
        return_display=return_display,
    )
    return train_ds, val_ds
