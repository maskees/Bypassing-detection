"""Fast training script — runs all 4 models + eval in ~10 min on RTX 3050."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

from models.road_sign_classifier import RoadSignClassifier, NormalizedModel, load_road_sign_classifier_checkpoint
from models.target_model import DetectorNet
from road_sign_data import CLASS_NAMES, load_records, stratified_split, RoadSignCropDataset, DisplayTensorDataset
from defenses.adversarial_training import train_adversarial_model
from defenses.defensive_distillation import train_distilled_model
from defenses.detection_network import train_detector

os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Config ──
BACKBONE = "resnet34"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = len(CLASS_NAMES)
BASE_CHECKPOINT = "saved_models/road_sign_crop_resnet34.pth"

# ── Dataset ──
print("\n[1/6] Loading dataset...")
records = load_records("annotations", "images")
train_records, val_records = stratified_split(records, val_ratio=0.2, seed=42)
print(f"  Train: {len(train_records)}  Val: {len(val_records)}")

train_ds = RoadSignCropDataset(train_records, image_size=IMAGE_SIZE, augment=True, return_display=True)
val_ds = RoadSignCropDataset(val_records, image_size=IMAGE_SIZE, augment=False, return_display=True)
train_display_ds = DisplayTensorDataset(train_ds)
val_display_ds = DisplayTensorDataset(val_ds)

# Class weights
counts = Counter(r.label for r in train_records)
total_samples = sum(counts.values())
cw = [total_samples / max(counts[i], 1) for i in range(NUM_CLASSES)]
cw_tensor = torch.tensor(cw, dtype=torch.float32, device=device)
class_weights = cw_tensor / cw_tensor.mean()

defense_loader = DataLoader(train_display_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_display_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

T_TOTAL = time.time()

# ── Step 1: Base model ──
if os.path.exists(BASE_CHECKPOINT):
    print(f"\n[2/6] Base model exists — loading {BASE_CHECKPOINT}")
    base_clf, ckpt = load_road_sign_classifier_checkpoint(BASE_CHECKPOINT, device=device)
    acc = ckpt.get("metrics", {}).get("validation", {}).get("accuracy", 0) * 100
    print(f"  Val accuracy: {acc:.1f}%")
else:
    print("\n[2/6] Training base classifier (10 epochs)...")
    def collate_batch(batch):
        return {"image": torch.stack([b["image"] for b in batch]),
                "label": torch.stack([b["label"] for b in batch])}

    sw = [1.0 / max(counts[r.label], 1) for r in train_records]
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, collate_fn=collate_batch)
    vl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_batch)

    base_clf = RoadSignClassifier(num_classes=NUM_CLASSES, backbone=BACKBONE, pretrained=True).to(device)
    opt = optim.AdamW(base_clf.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    best = -1.0
    t0 = time.time()
    for ep in range(1, 11):
        base_clf.train()
        for batch in train_loader:
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(base_clf(imgs), lbls, weight=class_weights)
            loss.backward(); opt.step()
        sched.step()
        base_clf.eval()
        c = t = 0
        with torch.no_grad():
            for batch in vl:
                imgs, lbls = batch["image"].to(device), batch["label"].to(device)
                c += (base_clf(imgs).argmax(1) == lbls).sum().item(); t += lbls.size(0)
        vacc = 100.0 * c / t
        if vacc >= best:
            best = vacc
            torch.save({"model_state": base_clf.state_dict(), "epoch": ep,
                        "metrics": {"train": {}, "validation": {"accuracy": vacc / 100}},
                        "class_names": CLASS_NAMES,
                        "config": {"num_classes": NUM_CLASSES, "backbone": BACKBONE,
                                   "image_size": IMAGE_SIZE, "padding_ratio": 0.2}},
                       BASE_CHECKPOINT)
        print(f"  Epoch {ep}/10  Val={vacc:.1f}%{'  *' if vacc >= best else ''}")
    print(f"  Done in {time.time()-t0:.0f}s  Best: {best:.1f}%")
    base_clf, _ = load_road_sign_classifier_checkpoint(BASE_CHECKPOINT, device=device)

# ── Step 2: Adversarial Training ──
print("\n[3/6] Adversarial training (3 epochs, 3 PGD steps)...")
def adv_factory():
    return NormalizedModel(RoadSignClassifier(num_classes=NUM_CLASSES, backbone=BACKBONE, pretrained=True))

t0 = time.time()
adv_model = train_adversarial_model(
    model_factory=adv_factory, train_loader=defense_loader,
    epsilon=8/255, alpha=2/255, pgd_steps=3, epochs=3,
    lr=1e-4, device=device, class_weights=class_weights)
inner = adv_model.model
torch.save({"model_state": inner.state_dict(), "epoch": 3, "metrics": {},
            "class_names": CLASS_NAMES,
            "config": {"num_classes": NUM_CLASSES, "backbone": BACKBONE,
                       "image_size": IMAGE_SIZE, "padding_ratio": 0.2}},
           "saved_models/road_sign_crop_adv_trained.pth")
print(f"  Saved in {time.time()-t0:.0f}s")

# ── Step 3: Distillation ──
print("\n[4/6] Defensive distillation (3 epochs)...")
teacher = NormalizedModel(base_clf).to(device); teacher.eval()
def dist_factory():
    return NormalizedModel(RoadSignClassifier(num_classes=NUM_CLASSES, backbone=BACKBONE, pretrained=True))

t0 = time.time()
distilled_model = train_distilled_model(
    teacher_model=teacher, model_factory=dist_factory, train_loader=defense_loader,
    temperature=20.0, alpha=0.7, epochs=3, lr=1e-4, device=device)
inner = distilled_model.model
torch.save({"model_state": inner.state_dict(), "epoch": 3, "metrics": {},
            "class_names": CLASS_NAMES,
            "config": {"num_classes": NUM_CLASSES, "backbone": BACKBONE,
                       "image_size": IMAGE_SIZE, "padding_ratio": 0.2}},
           "saved_models/road_sign_crop_distilled.pth")
print(f"  Saved in {time.time()-t0:.0f}s")

# ── Step 4: Detection Network ──
print("\n[5/6] Detection network (3 epochs)...")
base_norm = NormalizedModel(base_clf).to(device); base_norm.eval()
detector = DetectorNet(input_dim=base_clf.feature_dim)
t0 = time.time()
detector = train_detector(
    target_model=base_norm, detector_model=detector, train_loader=defense_loader,
    epsilon=8/255, epochs=3, lr=1e-3, device=device)
torch.save(detector.state_dict(), "saved_models/road_sign_crop_detector.pth")
print(f"  Saved in {time.time()-t0:.0f}s")

# ── Step 5: Clean accuracy ──
print("\n[6/6] Clean accuracy check...")
models_check = {
    "Base": base_norm,
    "Adv Trained": adv_model,
    "Distilled": distilled_model,
}
for name, model in models_check.items():
    model.eval()
    c = t = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            c += (model(imgs).argmax(1) == lbls).sum().item(); t += lbls.size(0)
    print(f"  {name:15s} {100.*c/t:.1f}%")

elapsed = time.time() - T_TOTAL
print(f"\n{'='*50}")
print(f"ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*50}")
print("Saved models:")
for f in ["road_sign_crop_resnet34.pth", "road_sign_crop_adv_trained.pth",
          "road_sign_crop_distilled.pth", "road_sign_crop_detector.pth"]:
    p = f"saved_models/{f}"
    sz = os.path.getsize(p) / 1024 / 1024 if os.path.exists(p) else 0
    print(f"  {p} ({sz:.1f} MB)")
print("\nRun: python app.py")
