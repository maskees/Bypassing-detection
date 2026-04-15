"""Fast test script — evaluates all models in ~5 min on RTX 3050."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from models.road_sign_classifier import NormalizedModel, load_road_sign_classifier_checkpoint
from models.target_model import DetectorNet
from road_sign_data import CLASS_NAMES, load_records, stratified_split, RoadSignCropDataset, DisplayTensorDataset
from attacks.fgsm import fgsm_attack, fgsm_attack_single
from attacks.pgd import pgd_attack, pgd_attack_single
from defenses.input_transformation import apply_input_transforms
from defenses.detection_network import detect_and_predict

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

NUM_CLASSES = len(CLASS_NAMES)
EPSILON = 8 / 255

# ── Load models ──
print("\n[1/7] Loading models...")
models = {}
for name, path in [("base", "saved_models/road_sign_crop_resnet34.pth"),
                    ("adv_trained", "saved_models/road_sign_crop_adv_trained.pth"),
                    ("distilled", "saved_models/road_sign_crop_distilled.pth")]:
    clf, _ = load_road_sign_classifier_checkpoint(path, device=device)
    models[name] = NormalizedModel(clf).to(device)
    models[name].eval()
    print(f"  Loaded {name}")

det_path = "saved_models/road_sign_crop_detector.pth"
detector = DetectorNet(input_dim=512).to(device)
detector.load_state_dict(torch.load(det_path, map_location=device, weights_only=True))
detector.eval()
models["detector"] = detector
print(f"  Loaded detector")

# ── Load data ──
print("\n[2/7] Loading validation data...")
records = load_records("annotations", "images")
_, val_records = stratified_split(records, val_ratio=0.2, seed=42)
val_ds = RoadSignCropDataset(val_records, image_size=224, augment=False, return_display=True)
val_display_ds = DisplayTensorDataset(val_ds)
test_loader = DataLoader(val_display_ds, batch_size=64, shuffle=False, num_workers=0)
print(f"  {len(val_display_ds)} images")

T_TOTAL = time.time()

# ── Clean accuracy ──
print("\n[3/7] Clean accuracy...")
for name in ["base", "adv_trained", "distilled"]:
    model = models[name]
    c = t = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            c += (model(imgs).argmax(1) == lbls).sum().item(); t += lbls.size(0)
    print(f"  {name:15s} {100.*c/t:.1f}%")

# ── Confusion matrix (base) ──
print("\n[4/7] Confusion matrix (base model)...")
cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
with torch.no_grad():
    for imgs, lbls in test_loader:
        preds = models["base"](imgs.to(device)).argmax(1).cpu()
        for t_val, p in zip(lbls, preds):
            cm[t_val, p] += 1
print("  Predicted ->  ", "  ".join(f"{n:>12}" for n in CLASS_NAMES))
for i, name in enumerate(CLASS_NAMES):
    row = "  ".join(f"{cm[i,j].item():>12}" for j in range(NUM_CLASSES))
    acc = cm[i,i].item() / max(cm[i].sum().item(), 1) * 100
    print(f"  {name:>12}  {row}   ({acc:.0f}%)")

# ── Batch attacks ──
print("\n[5/7] Batch attack comparison (20 samples)...")
test_images, test_labels = [], []
for imgs, lbls in test_loader:
    test_images.append(imgs); test_labels.append(lbls)
    if sum(x.size(0) for x in test_images) >= 20:
        break
test_images = torch.cat(test_images)[:20].to(device)
test_labels = torch.cat(test_labels)[:20].to(device)

def robust_accuracy(model, adv_images, labels):
    model.eval()
    with torch.no_grad():
        return (model(adv_images).argmax(1) == labels).float().mean().item() * 100

adv_fgsm, _, _ = fgsm_attack(models["base"], test_images, test_labels, EPSILON, device)
adv_pgd, _, _ = pgd_attack(models["base"], test_images, test_labels, EPSILON, device=device)

print(f"  {'Attack':<10} {'Defense':<22} {'Robust Acc':>10}")
print(f"  {'-'*45}")
for atk_name, adv_imgs in [("FGSM", adv_fgsm), ("PGD", adv_pgd)]:
    for def_name, def_model in [("No Defense", models["base"]),
                                 ("Adv Training", models["adv_trained"]),
                                 ("Distillation", models["distilled"])]:
        acc = robust_accuracy(def_model, adv_imgs, test_labels)
        print(f"  {atk_name:<10} {def_name:<22} {acc:>9.1f}%")

    with torch.no_grad():
        transformed = apply_input_transforms(adv_imgs)
        acc = robust_accuracy(models["base"], transformed, test_labels)
    print(f"  {atk_name:<10} {'Input Transform':<22} {acc:>9.1f}%")

    preds, detected, _ = detect_and_predict(models["base"], models["detector"], adv_imgs, device)
    det_rate = detected.float().mean().item() * 100
    print(f"  {atk_name:<10} {'Detection (det rate)':<22} {det_rate:>9.1f}%")
    print()

# ── Detector scores ──
print("[6/7] Detector score distribution...")
with torch.no_grad():
    clean_feats = models["base"].get_features(test_images)
    clean_scores = F.softmax(models["detector"](clean_feats), dim=1)[:, 1].cpu().numpy()
    adv_feats = models["base"].get_features(adv_pgd)
    adv_scores = F.softmax(models["detector"](adv_feats), dim=1)[:, 1].cpu().numpy()

tp = (adv_scores > 0.5).sum(); fn = (adv_scores <= 0.5).sum()
tn = (clean_scores <= 0.5).sum(); fp = (clean_scores > 0.5).sum()
print(f"  Clean  -> Detected as adversarial: {fp}/{len(clean_scores)} (FPR: {fp/len(clean_scores)*100:.1f}%)")
print(f"  PGD    -> Detected as adversarial: {tp}/{len(adv_scores)} (TPR: {tp/len(adv_scores)*100:.1f}%)")

# ── Accuracy vs epsilon ──
print("\n[7/7] Accuracy vs epsilon...")
epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]
small_batch = test_images[:15]
small_labels = test_labels[:15]
print(f"  {'Epsilon':<10} {'FGSM':>8} {'PGD':>8}")
for eps in epsilons:
    if eps == 0:
        acc = robust_accuracy(models["base"], small_batch, small_labels)
        print(f"  {eps:<10.3f} {acc:>7.1f}% {acc:>7.1f}%")
    else:
        af, _, _ = fgsm_attack(models["base"], small_batch, small_labels, eps, device)
        ap, _, _ = pgd_attack(models["base"], small_batch, small_labels, eps, device=device)
        fa = robust_accuracy(models["base"], af, small_labels)
        pa = robust_accuracy(models["base"], ap, small_labels)
        print(f"  {eps:<10.3f} {fa:>7.1f}% {pa:>7.1f}%")

elapsed = time.time() - T_TOTAL
print(f"\n{'='*50}")
print(f"ALL TESTS DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*50}")
