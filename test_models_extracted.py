import os, sys, json, time
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.dirname(os.path.abspath('__file__'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.target_model import MNISTNet, DetectorNet
from attacks.fgsm import fgsm_attack, fgsm_attack_single
from attacks.pgd import pgd_attack, pgd_attack_single
from attacks.genetic_attack import genetic_attack
from attacks.differential_evolution_attack import de_attack
from defenses.input_transformation import apply_input_transforms
from defenses.detection_network import detect_and_predict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── Load all saved models ──
def load_model(cls, path, device):
    m = cls().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m

models = {
    'base':        load_model(MNISTNet, 'saved_models/base_model.pth', DEVICE),
    'adv_trained': load_model(MNISTNet, 'saved_models/adv_trained_model.pth', DEVICE),
    'distilled':   load_model(MNISTNet, 'saved_models/distilled_model.pth', DEVICE),
    'detector':    load_model(DetectorNet, 'saved_models/detector_model.pth', DEVICE),
}
print('✅ All 4 models loaded')

# ── Load test dataset ──
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                              transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
print(f'✅ Test set: {len(test_dataset)} images')

def compute_accuracy(model, loader, device=DEVICE):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return 100.0 * correct / total

print('Model                  Clean Accuracy')
print('-' * 42)
acc_results = {}
for name in ['base', 'adv_trained', 'distilled']:
    acc = compute_accuracy(models[name], test_loader)
    acc_results[name] = acc
    print(f'{name:<22} {acc:.2f}%')

# Bar chart
fig, ax = plt.subplots(figsize=(7, 3.5))
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(acc_results.keys(), acc_results.values(), color=colors, width=0.5)
ax.set_ylim(90, 100); ax.set_ylabel('Accuracy (%)')
ax.set_title('Clean Test Accuracy', fontweight='bold')
for b, v in zip(bars, acc_results.values()):
    ax.text(b.get_x()+b.get_width()/2, v+0.1, f'{v:.2f}%', ha='center', fontweight='bold')
ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()

def confusion_matrix(model, loader, num_classes=10, device=DEVICE):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            for t, p in zip(lbls, preds):
                cm[t, p] += 1
    return cm.numpy()

cm = confusion_matrix(models['base'], test_loader)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Base Model — Confusion Matrix', fontweight='bold')
for i in range(10):
    for j in range(10):
        c = 'white' if cm[i,j] > cm.max()/2 else 'black'
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', color=c, fontsize=8)
plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()

per_class = cm.diagonal() / cm.sum(axis=1) * 100
print('Per-class accuracy:')
for d in range(10):
    print(f'  Digit {d}: {per_class[d]:.1f}%')

EPSILON = 0.3
SAMPLE_IDX = 0  # Change this to test different images

image, label = test_dataset[SAMPLE_IDX]
print(f'Sample index: {SAMPLE_IDX}, True label: {label}')

def plot_attack_result(res, title):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    # Original
    axes[0].imshow(res['original'].squeeze(), cmap='gray')
    axes[0].set_title(f"Original\nPred: {res['orig_pred']}")
    # Adversarial
    axes[1].imshow(res['adversarial'].squeeze(), cmap='gray')
    color = 'red' if res['success'] else 'green'
    axes[1].set_title(f"Adversarial\nPred: {res['adv_pred']}", color=color)
    # Perturbation (amplified)
    pert = res['perturbation'].squeeze().numpy()
    axes[2].imshow(pert, cmap='RdBu_r', vmin=-EPSILON, vmax=EPSILON)
    axes[2].set_title(f'Perturbation\nL∞={res["l_inf"]:.3f}')
    # Confidence bars
    x = range(10)
    axes[3].bar(x, res['orig_probs'], alpha=0.5, label='Original', color='#3498db')
    axes[3].bar(x, res['adv_probs'], alpha=0.5, label='Adversarial', color='#e74c3c')
    axes[3].set_title('Confidence'); axes[3].legend(fontsize=8)
    axes[3].set_xticks(x)
    for a in axes[:3]: a.axis('off')
    plt.tight_layout(); plt.show()
    status = '✅ SUCCESS' if res['success'] else '❌ FAILED'
    print(f"  {status} | L∞={res['l_inf']:.4f} | L2={res['l2']:.4f}")

# ── FGSM Attack ──
res_fgsm = fgsm_attack_single(models['base'], image, label, EPSILON, DEVICE)
plot_attack_result(res_fgsm, 'FGSM Attack (single-step gradient)')

# ── PGD Attack ──
res_pgd = pgd_attack_single(models['base'], image, label, EPSILON, steps=40, device=DEVICE)
plot_attack_result(res_pgd, 'PGD Attack (40-step iterative gradient)')

# ── Genetic Algorithm Attack ──
res_ga = genetic_attack(models['base'], image.to(DEVICE), label,
                        EPSILON, pop_size=30, generations=50, device=DEVICE)
plot_attack_result(res_ga, f'Genetic Algorithm Attack (queries={res_ga["queries"]})')

# ── Differential Evolution Attack ──
res_de = de_attack(models['base'], image.to(DEVICE), label,
                   EPSILON, maxiter=50, device=DEVICE)
plot_attack_result(res_de, f'Differential Evolution Attack (queries={res_de["queries"]})')

# Grab a small test batch
test_images, test_labels = next(iter(test_loader))
test_images = test_images[:50].to(DEVICE)
test_labels = test_labels[:50].to(DEVICE)
print(f'Testing on {test_images.size(0)} images')

def robust_accuracy(model, adv_images, labels, device=DEVICE):
    model.eval()
    with torch.no_grad():
        preds = model(adv_images.to(device)).argmax(1)
    return (preds == labels.to(device)).float().mean().item() * 100

# Generate adversarial examples with FGSM & PGD on base model
adv_fgsm, _, _ = fgsm_attack(models['base'], test_images, test_labels, EPSILON, DEVICE)
adv_pgd, _, _ = pgd_attack(models['base'], test_images, test_labels, EPSILON, device=DEVICE)

print(f"\n{'Attack':<10} {'Defense':<22} {'Robust Acc':>10}")
print('-' * 45)

defense_results = {}
for atk_name, adv_imgs in [('FGSM', adv_fgsm), ('PGD', adv_pgd)]:
    defense_results[atk_name] = {}
    for def_name, def_model in [('No Defense', models['base']),
                                 ('Adv. Training', models['adv_trained']),
                                 ('Distillation', models['distilled'])]:
        acc = robust_accuracy(def_model, adv_imgs, test_labels)
        defense_results[atk_name][def_name] = acc
        print(f'{atk_name:<10} {def_name:<22} {acc:>9.1f}%')
    
    # Input transformation defense
    with torch.no_grad():
        transformed = apply_input_transforms(adv_imgs)
        acc = robust_accuracy(models['base'], transformed, test_labels)
    defense_results[atk_name]['Input Transform'] = acc
    print(f'{atk_name:<10} {"Input Transform":<22} {acc:>9.1f}%')
    
    # Detection defense
    preds, detected, _ = detect_and_predict(models['base'], models['detector'],
                                            adv_imgs, DEVICE)
    det_rate = detected.float().mean().item() * 100
    defense_results[atk_name]['Detection'] = det_rate
    print(f'{atk_name:<10} {"Detection (det rate)":<22} {det_rate:>9.1f}%')
    print()

# Test detector on clean vs adversarial inputs
models['base'].eval(); models['detector'].eval()

with torch.no_grad():
    clean_feats = models['base'].get_features(test_images)
    clean_det = F.softmax(models['detector'](clean_feats), dim=1)[:, 1].cpu().numpy()
    
    adv_feats = models['base'].get_features(adv_pgd)
    adv_det = F.softmax(models['detector'](adv_feats), dim=1)[:, 1].cpu().numpy()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(clean_det, bins=30, alpha=0.6, label='Clean', color='#2ecc71', edgecolor='white')
ax.hist(adv_det, bins=30, alpha=0.6, label='Adversarial (PGD)', color='#e74c3c', edgecolor='white')
ax.axvline(0.5, color='black', linestyle='--', label='Threshold')
ax.set_xlabel('P(adversarial)'); ax.set_ylabel('Count')
ax.set_title('Detection Network — Score Distribution', fontweight='bold')
ax.legend(); plt.tight_layout(); plt.show()

tp = (adv_det > 0.5).sum(); fn = (adv_det <= 0.5).sum()
tn = (clean_det <= 0.5).sum(); fp = (clean_det > 0.5).sum()
print(f'Clean  → Detected as adversarial: {fp}/{len(clean_det)} (FPR: {fp/len(clean_det)*100:.1f}%)')
print(f'PGD    → Detected as adversarial: {tp}/{len(adv_det)} (TPR: {tp/len(adv_det)*100:.1f}%)')

epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
small_batch = test_images[:30]
small_labels = test_labels[:30]

fgsm_accs = []; pgd_accs = []

for eps in epsilons:
    if eps == 0:
        acc = robust_accuracy(models['base'], small_batch, small_labels)
        fgsm_accs.append(acc); pgd_accs.append(acc)
    else:
        adv_f, _, _ = fgsm_attack(models['base'], small_batch, small_labels, eps, DEVICE)
        fgsm_accs.append(robust_accuracy(models['base'], adv_f, small_labels))
        adv_p, _, _ = pgd_attack(models['base'], small_batch, small_labels, eps, device=DEVICE)
        pgd_accs.append(robust_accuracy(models['base'], adv_p, small_labels))
    print(f'ε={eps:.2f}: FGSM={fgsm_accs[-1]:.1f}%, PGD={pgd_accs[-1]:.1f}%')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epsilons, fgsm_accs, 'o-', label='FGSM', color='#e74c3c', linewidth=2)
ax.plot(epsilons, pgd_accs, 's-', label='PGD', color='#9b59b6', linewidth=2)
ax.set_xlabel('Epsilon (ε)'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Base Model Accuracy vs. Perturbation Budget', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-5, 105)
plt.tight_layout(); plt.show()

results_path = 'results/evaluation_results.json'
if not os.path.exists(results_path):
    print('⚠️  No evaluation results found — run train_models.ipynb Step 7 first.')
else:
    with open(results_path) as f:
        results = json.load(f)
    
    atk_names = results['attack_names']
    def_names = results['defense_names']
    matrix = results['results']
    
    atk_keys = list(atk_names.keys())
    def_keys = list(def_names.keys())
    
    # Build ASR matrix
    asr = np.zeros((len(atk_keys), len(def_keys)))
    for i, ak in enumerate(atk_keys):
        for j, dk in enumerate(def_keys):
            asr[i,j] = matrix.get(ak, {}).get(dk, {}).get('attack_success_rate', 0)
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(asr, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(def_keys)))
    ax.set_xticklabels([def_names[k] for k in def_keys], rotation=25, ha='right')
    ax.set_yticks(range(len(atk_keys)))
    ax.set_yticklabels([atk_names[k] for k in atk_keys])
    for i in range(len(atk_keys)):
        for j in range(len(def_keys)):
            c = 'white' if asr[i,j] > 60 else 'black'
            ax.text(j, i, f'{asr[i,j]:.1f}%', ha='center', va='center',
                    fontweight='bold', color=c, fontsize=10)
    ax.set_title('Attack Success Rate (%) — Lower = Better Defense',
                 fontsize=13, fontweight='bold', pad=12)
    plt.colorbar(im, ax=ax, label='ASR (%)')
    plt.tight_layout(); plt.show()
    
    # Print table
    print(f"\n{'Attack':<28}", end='')
    for dk in def_keys: print(f"{def_names[dk]:>16}", end='')
    print('\n' + '-'*108)
    for ak in atk_keys:
        print(f"{atk_names[ak]:<28}", end='')
        for dk in def_keys:
            v = matrix.get(ak,{}).get(dk,{}).get('attack_success_rate', 0)
            print(f"{v:>15.1f}%", end='')
        print()

# Pick a sample and attack it
idx = 3
img, lbl = test_dataset[idx]
img_dev = img.unsqueeze(0).to(DEVICE)
lbl_dev = torch.tensor([lbl], device=DEVICE)

adv, pert, _ = pgd_attack(models['base'], img_dev, lbl_dev, 0.3, device=DEVICE)

fig, axes = plt.subplots(1, 5, figsize=(16, 3))
fig.suptitle(f'PGD Attack on digit {lbl} — Defense Comparison', fontweight='bold', fontsize=13)

panels = [
    ('Original', img.squeeze(), models['base'], img_dev),
    ('No Defense', adv.squeeze(0).cpu().squeeze(), models['base'], adv),
    ('Adv Training', adv.squeeze(0).cpu().squeeze(), models['adv_trained'], adv),
    ('Distillation', adv.squeeze(0).cpu().squeeze(), models['distilled'], adv),
]

for i, (title, show_img, model, inp) in enumerate(panels):
    axes[i].imshow(show_img.detach().numpy(), cmap='gray')
    with torch.no_grad():
        pred = model(inp.to(DEVICE)).argmax(1).item()
    color = 'green' if pred == lbl else 'red'
    axes[i].set_title(f'{title}\nPred: {pred}', color=color, fontweight='bold')
    axes[i].axis('off')

# Input transform defense
with torch.no_grad():
    transformed = apply_input_transforms(adv)
    pred_t = models['base'](transformed).argmax(1).item()
axes[4].imshow(transformed.squeeze().cpu().numpy(), cmap='gray')
color = 'green' if pred_t == lbl else 'red'
axes[4].set_title(f'Input Transform\nPred: {pred_t}', color=color, fontweight='bold')
axes[4].axis('off')

plt.tight_layout(); plt.show()

batch_imgs = test_images[:100]
batch_lbls = test_labels[:100]
adv_batch, _, _ = pgd_attack(models['base'], batch_imgs, batch_lbls, 0.3, device=DEVICE)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, [('Base', models['base']),
                                     ('Adv Trained', models['adv_trained']),
                                     ('Distilled', models['distilled'])]):
    model.eval()
    with torch.no_grad():
        clean_conf = F.softmax(model(batch_imgs), dim=1).max(1)[0].cpu().numpy()
        adv_conf = F.softmax(model(adv_batch), dim=1).max(1)[0].cpu().numpy()
    ax.hist(clean_conf, bins=25, alpha=0.6, label='Clean', color='#2ecc71', edgecolor='white')
    ax.hist(adv_conf, bins=25, alpha=0.6, label='PGD Adv', color='#e74c3c', edgecolor='white')
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_xlabel('Max Confidence'); ax.legend(fontsize=9)

fig.suptitle('Confidence Distribution: Clean vs Adversarial', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

print('✅ All tests complete!')
print()
print('Saved models verified:')
for f in ['base_model.pth', 'adv_trained_model.pth', 'distilled_model.pth', 'detector_model.pth']:
    path = f'saved_models/{f}'
    sz = os.path.getsize(path)/1024 if os.path.exists(path) else 0
    print(f'  ✓ {path:<42} ({sz:.0f} KB)')
print()
print('To launch the web interface:')
print('  python app.py')
print('  Open: http://localhost:5000')