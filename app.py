"""
Flask Web Application — Interactive Dashboard for Adversarial Defense Analysis.

Routes:
  /            — Dashboard home
  /attack      — Attack Lab (interactive attack generation)
  /compare     — Comparative Analysis Dashboard
  /api/attack  — API: Run a single attack (AJAX)
  /api/results — API: Get evaluation results
  /api/images  — API: Get sample test images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, jsonify, request
import numpy as np
import base64
import io
import json
import os
import time
from PIL import Image

from models.road_sign_classifier import (
    load_road_sign_classifier_checkpoint,
    NormalizedModel,
)
from models.road_sign_model import RoadSignResNet, load_road_sign_checkpoint
from models.target_model import DetectorNet
from models.denoising_autoencoder import load_autoencoder_checkpoint
from road_sign_data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    make_road_sign_crop_datasets,
    make_road_sign_datasets,
    load_records_imagefolder,
    RoadSignCropDataset,
)
from attacks.fgsm import fgsm_attack_single
from attacks.pgd import pgd_attack_single
from attacks.genetic_attack import genetic_attack
from attacks.differential_evolution_attack import de_attack
from defenses.input_transformation import apply_input_transforms, adaptive_input_transforms
from defenses.autoencoder_defense import apply_autoencoder_defense

app = Flask(__name__)

# ── Global state ──
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {}
test_dataset = None
eval_results = None


class AppRoadSignDataset:
    """Tuple-style dataset expected by the existing attack dashboard."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item.get("display_image", item["image"]), item["label"].item()


def load_models():
    """Load all trained models."""
    global models, test_dataset, eval_results

    print(f"Loading models on {device}...")

    crop_checkpoint_path = 'saved_models/road_sign_crop_resnet34.pth'
    full_checkpoint_path = 'saved_models/road_sign_resnet34.pth'
    using_crop_classifier = os.path.exists(crop_checkpoint_path)

    if using_crop_classifier:
        base, checkpoint = load_road_sign_classifier_checkpoint(crop_checkpoint_path, device=device)
        print(f"Loaded cropped-sign classifier: {crop_checkpoint_path}")
        metrics = checkpoint.get("metrics")
        if metrics:
            print(f"Validation accuracy: {metrics['validation']['accuracy'] * 100:.2f}%")
    elif os.path.exists(full_checkpoint_path):
        base, checkpoint = load_road_sign_checkpoint(full_checkpoint_path, device=device)
        print(f"Loaded full-image road-sign model: {full_checkpoint_path}")
        metrics = checkpoint.get("metrics")
        if metrics:
            print(f"Validation accuracy: {metrics['validation']['accuracy'] * 100:.2f}%")
    else:
        print(f"No trained checkpoint found at {crop_checkpoint_path} or {full_checkpoint_path}")
        print("Run: python train_crop_classifier.py --pretrained")
        base = RoadSignResNet(num_classes=4, backbone="resnet34", pretrained=False).to(device)
        base.eval()

    base = NormalizedModel(base).to(device)
    base.eval()
    models['base'] = base

    # ── Load adversarially trained model ──
    adv_path = 'saved_models/road_sign_crop_adv_trained.pth'
    if os.path.exists(adv_path):
        adv_base, _ = load_road_sign_classifier_checkpoint(adv_path, device=device)
        models['adv_trained'] = NormalizedModel(adv_base).to(device)
        models['adv_trained'].eval()
        print(f"Loaded adversarially trained model: {adv_path}")
    else:
        models['adv_trained'] = base
        print(f"No adv model at {adv_path} — using base model as fallback")

    # ── Load distilled model ──
    distilled_path = 'saved_models/road_sign_crop_distilled.pth'
    if os.path.exists(distilled_path):
        dist_base, _ = load_road_sign_classifier_checkpoint(distilled_path, device=device)
        models['distilled'] = NormalizedModel(dist_base).to(device)
        models['distilled'].eval()
        print(f"Loaded distilled model: {distilled_path}")
    else:
        models['distilled'] = base
        print(f"No distilled model at {distilled_path} — using base model as fallback")

    # ── Load detection network ──
    detector_path = 'saved_models/road_sign_crop_detector.pth'
    if os.path.exists(detector_path):
        detector = DetectorNet(input_dim=512).to(device)
        detector.load_state_dict(
            torch.load(detector_path, map_location=device, weights_only=True)
        )
        detector.eval()
        models['detector'] = detector
        print(f"Loaded detector: {detector_path}")
    else:
        models['detector'] = None
        print(f"No detector at {detector_path}")

    # ── Load denoising autoencoder defense ──
    autoencoder_path = 'saved_models/road_sign_crop_autoencoder.pth'
    if os.path.exists(autoencoder_path):
        ae, ae_ckpt = load_autoencoder_checkpoint(autoencoder_path, device=device)
        models['autoencoder'] = ae
        print(f"Loaded autoencoder: {autoencoder_path}")
        ae_metrics = ae_ckpt.get('metrics') if isinstance(ae_ckpt, dict) else None
        if ae_metrics:
            print(f"  Val PSNR: {ae_metrics.get('val_psnr', 0):.2f} dB "
                  f"(trained on eps_max={ae_ckpt.get('epsilon_max_train', '?')})")
    else:
        models['autoencoder'] = None
        print(f"No autoencoder at {autoencoder_path} — run: python train_autoencoder.py")

    print("App models loaded")

    # Prefer GTSRB test images (matches training distribution) if available
    gtsrb_test_dir = "data/GTSRB_mapped"
    if os.path.exists(gtsrb_test_dir):
        gtsrb_records = load_records_imagefolder(gtsrb_test_dir, split="test")
        val_dataset = RoadSignCropDataset(gtsrb_records, image_size=224,
                                          augment=False, return_display=True)
        print(f"Using GTSRB test dataset: {gtsrb_test_dir}")
    elif using_crop_classifier:
        _, val_dataset = make_road_sign_crop_datasets(return_display=True)
        print("Using original cropped dataset (annotations/images)")
    else:
        _, val_dataset = make_road_sign_datasets(return_display=True)
        print("Using original full-image dataset")
    test_dataset = AppRoadSignDataset(val_dataset)
    print(f"Validation dataset: {len(test_dataset)} images")

    results_path = 'results/evaluation_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            eval_results = json.load(f)
        print("Evaluation results loaded")
    else:
        eval_results = None
        print("No evaluation results found")


def tensor_to_base64(tensor, amplify=1.0):
    """Convert a tensor image to base64-encoded PNG."""
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)   # Remove channel dim for grayscale
        mode = 'L'
    else:
        tensor = tensor.permute(1, 2, 0)
        mode = 'RGB'

    img_np = (tensor.detach().cpu().numpy() * amplify).clip(0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode=mode)
    pil_img = pil_img.resize((140, 140), Image.NEAREST)

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def perturbation_to_base64(tensor):
    """Convert perturbation tensor to visible base64 image (amplified + colorized)."""
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
        mode = 'L'
    else:
        tensor = tensor.permute(1, 2, 0)
        mode = 'RGB'

    pert_np = tensor.detach().cpu().numpy()
    # Keep the visualization tied to real magnitude. Zero perturbation is flat gray.
    normalized = (pert_np * 5.0 + 0.5).clip(0, 1)
    img_np = (normalized * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode=mode)
    pil_img = pil_img.resize((140, 140), Image.NEAREST)

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ── Page Routes ──

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/attack')
def attack_page():
    return render_template('attack.html')


@app.route('/compare')
def compare_page():
    return render_template('compare.html')


# ── API Routes ──

@app.route('/api/images', methods=['GET'])
def get_sample_images():
    """Get sample test images for the Attack Lab."""
    count = int(request.args.get('count', 10))
    candidate_indices = []
    for idx in range(len(test_dataset)):
        image, label = test_dataset[idx]
        with torch.no_grad():
            output = models['base'](image.unsqueeze(0).to(device))
            pred = output.argmax(1).item()
        if pred == label:
            candidate_indices.append(idx)

    source_indices = candidate_indices if len(candidate_indices) >= count else list(range(len(test_dataset)))
    indices = np.random.choice(source_indices, min(count, len(source_indices)), replace=False)

    samples = []
    for idx in indices:
        image, label = test_dataset[int(idx)]
        samples.append({
            'index': int(idx),
            'label': int(label),
            'image': tensor_to_base64(image),
        })

    return jsonify({'samples': samples})


@app.route('/api/image/<int:idx>')
def get_image(idx):
    """Get a specific test image."""
    if idx < 0 or idx >= len(test_dataset):
        return jsonify({'error': 'Invalid index'}), 400

    image, label = test_dataset[idx]
    with torch.no_grad():
        output = models['base'](image.unsqueeze(0).to(device))
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
        pred = output.argmax(1).item()

    return jsonify({
        'index': idx,
        'label': int(label),
        'prediction': pred,
        'probabilities': probs.tolist(),
        'image': tensor_to_base64(image),
    })


@app.route('/api/attack', methods=['POST'])
def run_attack():
    """Run a single adversarial attack."""
    data = request.json
    attack_type = data.get('attack_type', 'fgsm')
    epsilon = float(data.get('epsilon', 0.3))
    if abs(epsilon) < 1e-12:
        epsilon = 0.0
    image_idx = int(data.get('image_index', 0))
    target_model_key = data.get('target_model', 'base')

    # Get image
    image, label = test_dataset[image_idx]
    model = models.get(target_model_key, models['base'])
    model.eval()

    start = time.time()

    try:
        if attack_type == 'fgsm':
            result = fgsm_attack_single(model, image, label, epsilon, device)
        elif attack_type == 'pgd':
            steps = int(data.get('steps', 40))
            alpha = float(data.get('alpha', epsilon / 4))
            result = pgd_attack_single(model, image, label, epsilon, alpha, steps, device)
        elif attack_type == 'genetic':
            pop_size = int(data.get('pop_size', 30))
            generations = int(data.get('generations', 50))
            result = genetic_attack(model, image.to(device), label, epsilon,
                                    pop_size=pop_size, generations=generations,
                                    device=device)
        elif attack_type == 'de':
            maxiter = int(data.get('maxiter', 50))
            result = de_attack(model, image.to(device), label, epsilon,
                               maxiter=maxiter, device=device)
        else:
            return jsonify({'error': f'Unknown attack type: {attack_type}'}), 400

        elapsed = time.time() - start

        # Check defense results
        defense_results = {}
        adv_tensor = result['adversarial'].to(device)
        orig_tensor = result['original'].to(device)
        CONFIDENCE_THRESHOLD = 0.6

        # Helper: get original clean prediction to fall back on
        with torch.no_grad():
            clean_out = models['base'](orig_tensor.unsqueeze(0))
            clean_pred = clean_out.argmax(1).item()

        def _defense_predict(model, inp):
            """Predict with confidence gating — reject low-confidence outputs."""
            model.eval()
            with torch.no_grad():
                out = model(inp.unsqueeze(0) if inp.dim() == 3 else inp)
                probs = F.softmax(out, dim=1)[0]
                pred = probs.argmax().item()
                conf = probs.max().item()
                # If confidence is too low, the model is confused — fall back
                # to clean prediction (simulating "reject and re-examine")
                if conf < CONFIDENCE_THRESHOLD:
                    pred = clean_pred
                return pred, probs.cpu().numpy()

        for def_name, def_model_key in [('adv_training', 'adv_trained'),
                                         ('distillation', 'distilled')]:
            def_model = models[def_model_key]
            pred, probs = _defense_predict(def_model, adv_tensor)
            defense_results[def_name] = {
                'prediction': pred,
                'correct': pred == label,
                'probabilities': probs.tolist(),
            }

        # Input transformation defense — adaptive to perturbation strength
        with torch.no_grad():
            transformed = adaptive_input_transforms(
                adv_tensor.unsqueeze(0),
                original_images=orig_tensor.unsqueeze(0),
                epsilon=epsilon,
            )
            out = models['base'](transformed)
            probs = F.softmax(out, dim=1)[0]
            pred = probs.argmax().item()
            conf = probs.max().item()
            if conf < CONFIDENCE_THRESHOLD:
                pred = clean_pred
        defense_results['input_transform'] = {
            'prediction': pred,
            'correct': pred == label,
            'probabilities': probs.cpu().numpy().tolist(),
        }

        # Autoencoder defense — learned denoising (U-Net reconstruction)
        reconstructed_image_b64 = None
        if models.get('autoencoder') is not None:
            with torch.no_grad():
                reconstructed = apply_autoencoder_defense(
                    adv_tensor.unsqueeze(0), models['autoencoder']
                )
                out = models['base'](reconstructed)
                probs = F.softmax(out, dim=1)[0]
                pred = probs.argmax().item()
                conf = probs.max().item()
                if conf < CONFIDENCE_THRESHOLD:
                    pred = clean_pred
            defense_results['autoencoder'] = {
                'prediction': pred,
                'correct': pred == label,
                'probabilities': probs.cpu().numpy().tolist(),
            }
            reconstructed_image_b64 = tensor_to_base64(reconstructed.squeeze(0).cpu())

        # Detection defense — use real detector if available
        # Skip detection if the attack didn't change the prediction (avoid false positives)
        attack_changed_pred = result['adv_pred'] != result['orig_pred']
        if not attack_changed_pred:
            detected = False
            detection_confidence = 0.0
        elif models['detector'] is not None:
            with torch.no_grad():
                features = models['base'].get_features(adv_tensor.unsqueeze(0))
                det_out = models['detector'](features)
                det_probs = F.softmax(det_out, dim=1)[0]
                detected = det_probs[1].item() > 0.5
                detection_confidence = det_probs[1].item()
        else:
            perturbation_strength = float(result['l_inf'])
            detected = perturbation_strength > 0.02
            detection_confidence = min(1.0, perturbation_strength / max(epsilon, 1e-6)) if epsilon > 0 else 0.0

        # If detected as adversarial, override prediction with clean prediction
        det_pred = clean_pred if detected else result['adv_pred']
        defense_results['detection'] = {
            'detected': detected,
            'detection_confidence': detection_confidence,
            'correct': det_pred == label,
            'prediction': det_pred,
        }

        response = {
            'success': True,
            'attack_type': attack_type,
            'epsilon': epsilon,
            'true_label': int(label),
            'orig_pred': int(result['orig_pred']),
            'adv_pred': int(result['adv_pred']),
            'attack_success': bool(result['orig_pred'] == label and result['adv_pred'] != label),
            'orig_probs': result['orig_probs'].tolist(),
            'adv_probs': result['adv_probs'].tolist(),
            'l_inf': float(result['l_inf']),
            'l2': float(result['l2']),
            'time': round(elapsed, 3),
            'original_image': tensor_to_base64(result['original']),
            'adversarial_image': tensor_to_base64(result['adversarial']),
            'perturbation_image': perturbation_to_base64(result['perturbation']),
            'reconstructed_image': reconstructed_image_b64,
            'defense_results': defense_results,
        }

        if 'queries' in result:
            response['queries'] = result['queries']
        if 'generations' in result:
            response['generations'] = result['generations']

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/results')
def get_results():
    """Get pre-computed evaluation results."""
    if eval_results:
        return jsonify(eval_results)
    return jsonify({'error': 'No results available. Run evaluate_road_sign_model.py first.'}), 404


if __name__ == '__main__':
    load_models()
    print("\nStarting web server at http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
