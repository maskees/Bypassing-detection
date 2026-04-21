# Bypassing Detection: Adversarial Attacks & Defenses on Road Sign Classifiers

A comprehensive study evaluating the robustness of a deep learning road sign classifier against adversarial attacks. We implement **4 attack methods** and **6 defense strategies**, resulting in 24 unique attack-defense combinations — all visualized through an interactive web dashboard.

---

## Overview

Deep neural networks achieve near-perfect accuracy on road sign classification, yet remain vulnerable to adversarial attacks — imperceptible pixel perturbations that cause confident misclassification. This project quantifies that vulnerability and evaluates countermeasures.

### Key Results (ε = 0.3)
| Attack | No Defense | Best Defense | Best Defense Name |
|--------|-----------|-------------|-------------------|
| FGSM | 100% ASR | 0% ASR | Detection Network |
| PGD | 100% ASR | 0.8% ASR | Defensive Distillation |
| Genetic Algorithm | 87.8% ASR | 59.2% ASR | Adversarial Training |
| Differential Evolution | 97.9% ASR | 71.4% ASR | Adversarial Training |

> **Finding:** No single defense provides universal robustness. Gradient-based defenses fail against evolutionary attacks.

---

## Architecture

```
Image [0,1] → NormalizedModel → ResNet-34 → 4-class prediction
                                    ↓
                              512-dim features → DetectorNet → clean/adversarial

Image [0,1] → DenoisingAutoencoder → Reconstructed [0,1] → NormalizedModel → ResNet-34
```

- **Backbone:** ResNet-34 (ImageNet pretrained, fine-tuned on 4 road sign classes)
- **Dataset:** GTSRB — 43 classes mapped to 4 (crosswalk, speedlimit, stop, trafficlight), balanced at 420 images/class
- **Training:** Altair HPC with NVIDIA H100 GPU

---

## Attacks

| Attack | Type | Gradients? | Speed | Description |
|--------|------|-----------|-------|-------------|
| **FGSM** | ML | Yes | ~0.02s | Single-step gradient sign perturbation |
| **PGD** | ML | Yes | ~0.15s | Iterative 40-step gradient attack with random start |
| **Genetic Algorithm** | EC | No | ~1.3s | Population-based evolutionary attack (selection, crossover, mutation) |
| **Differential Evolution** | EC | No | ~4.5s | Scipy-based continuous optimizer, strongest attack |

## Defenses

| Defense | Type | How it Works |
|---------|------|-------------|
| **No Defense** | Baseline | Standard ResNet-34, no hardening |
| **Adversarial Training** | Proactive | Fine-tune on PGD-attacked images (50/50 clean/adv mix) |
| **Defensive Distillation** | Proactive | Teacher-student training with soft labels (T=20) |
| **Input Transformation** | Preprocessing | Gaussian + median filtering to destroy adversarial noise |
| **Detection Network** | Reactive | Binary classifier on 512-dim features (clean vs adversarial) |
| **Autoencoder Denoise** | Generative | U-Net denoising autoencoder projects adversarial inputs back onto the clean-image manifold |

---

## Project Structure

```
Bypassing-Detection/
├── app.py                          # Flask web dashboard
├── road_sign_data.py               # Dataset loading & preprocessing
├── train_models.ipynb              # Training notebook (all stages)
├── setup_gtsrb.ipynb               # GTSRB download & class mapping
├── requirements.txt                # Python dependencies
│
├── models/
│   ├── road_sign_classifier.py     # RoadSignClassifier + NormalizedModel
│   ├── road_sign_model.py          # RoadSignResNet (with bbox head)
│   ├── target_model.py             # DetectorNet (adversarial detector)
│   └── denoising_autoencoder.py    # U-Net denoising autoencoder
│
├── attacks/
│   ├── fgsm.py                     # FGSM attack
│   ├── pgd.py                      # PGD attack
│   ├── genetic_attack.py           # Genetic Algorithm attack
│   └── differential_evolution_attack.py  # DE attack
│
├── defenses/
│   ├── adversarial_training.py     # Adversarial training defense
│   ├── defensive_distillation.py   # Distillation defense
│   ├── input_transformation.py     # Input transform defense
│   ├── detection_network.py        # Detection network defense
│   └── autoencoder_defense.py      # Autoencoder denoising defense
│
├── evaluation/
│   └── evaluator.py                # Full evaluation pipeline (4×6 matrix)
│
├── templates/                      # HTML templates for dashboard
├── static/                         # CSS/JS for dashboard
├── saved_models/                   # Model checkpoints (.pth)
├── results/                        # Evaluation results (.json)
└── data/GTSRB_mapped/              # Balanced dataset (train/test)
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download & Prepare Dataset
Run `setup_gtsrb.ipynb` to download the GTSRB dataset and map 43 classes to 4 target classes.

### Train Models
Run `train_models.ipynb` which executes 4 training stages:
1. **Base classifier** — ResNet-34 fine-tuning (10 epochs)
2. **Adversarial training** — PGD-based adversarial fine-tuning (5 epochs)
3. **Defensive distillation** — Teacher-student with T=20 (5 epochs)
4. **Detection network** — Binary adversarial detector (5 epochs)

### Train Autoencoder Defense
```bash
python train_autoencoder.py
```
Trains a U-Net denoising autoencoder on clean images with injected adversarial-like noise (15 epochs, ~minutes on GPU). Saves to `saved_models/road_sign_crop_autoencoder.pth`.

### Run Evaluation
The final cell in `train_models.ipynb` runs the full 4×6 evaluation matrix and saves results to `results/evaluation_results.json`.

### Launch Dashboard
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

---

## Web Dashboard

### Pages

- **Dashboard** — ASR heatmap, overview metrics
- **Attack Lab** — Interactive attack testing: select an image, pick an attack, adjust epsilon, see results in real-time with per-defense confidence breakdown
- **Comparison** — Full evaluation results: heatmap, bar charts, radar chart, accuracy-vs-epsilon curves, detailed metrics table

### Attack Lab Features
- Real-time adversarial image generation
- Side-by-side original vs adversarial vs reconstructed comparison
- Perturbation visualization (magnified 5×)
- Autoencoder reconstruction preview (denoised output)
- Confidence bars for base model predictions
- Defense effectiveness cards with per-defense confidence bars
- Three-state defense badges: ✓ Blocked (green), ⚠ Weak Block (yellow), ✗ Bypassed (red)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch, torchvision, CUDA |
| Model | ResNet-34 (ImageNet pretrained) |
| Web Framework | Flask + HTML/CSS/JS |
| Compute | Altair HPC, NVIDIA H100 80GB |
| Dataset | GTSRB (43 → 4 classes, balanced) |
| Evolutionary Attacks | scipy, numpy |

---

## Key Findings

1. **Gradient defenses fail against evolutionary attacks** — Adversarial training and distillation are effective against FGSM/PGD but nearly useless against GA/DE
2. **Distillation is best against PGD (0.8% ASR)** but worst against DE (97.9% ASR)
3. **Detection Network perfectly blocks FGSM (0% ASR)** but misses evolutionary attacks (74-85% ASR)
4. **No single defense is universal** — layered/ensemble defenses are needed
5. **Evolutionary attacks are strongest but slowest** — DE achieves highest ASR across all defenses at ~4.5s/image
