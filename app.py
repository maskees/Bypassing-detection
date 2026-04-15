"""
Flask Web Application — Interactive Dashboard for Adversarial Defense Analysis.
Adapted for Indian Traffic Sign dataset (58 classes, RGB 32×32).

Routes:
  /            — Dashboard home
  /attack      — Attack Lab (interactive attack generation)
  /compare     — Comparative Analysis Dashboard
  /api/attack  — API: Run a single attack (AJAX)
  /api/results — API: Get evaluation results
  /api/images  — API: Get sample test images
"""

import torch
import torch.nn.functional as F
from flask import Flask, render_template, jsonify, request
import numpy as np
import base64
import io
import json
import os
import time
from PIL import Image

from models.target_model import TrafficNet, DetectorNet
from models.data_utils import TrafficTestDataset, TEST_TRANSFORM, load_label_names
from attacks.fgsm import fgsm_attack_single
from attacks.pgd import pgd_attack_single
from attacks.genetic_attack import genetic_attack
from attacks.differential_evolution_attack import de_attack
from defenses.input_transformation import apply_input_transforms

app = Flask(__name__)

# ── Global state ──
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {}
test_dataset = None
eval_results = None
label_names = {}


def load_models():
    """Load all trained models."""
    global models, test_dataset, eval_results, label_names

    print(f"Loading models on {device}...")

    # Load label names
    try:
        label_names = load_label_names('data/labels.csv')
        print(f"[OK] Loaded {len(label_names)} class names")
    except Exception as e:
        print(f"[WARN] Could not load label names: {e}")
        label_names = {i: f"Class {i}" for i in range(58)}

    # Base model
    base = TrafficNet().to(device)
    base.load_state_dict(torch.load('saved_models/base_model.pth', map_location=device, weights_only=True))
    base.eval()
    models['base'] = base

    # Adversarially trained model
    adv = TrafficNet().to(device)
    adv.load_state_dict(torch.load('saved_models/adv_trained_model.pth', map_location=device, weights_only=True))
    adv.eval()
    models['adv_trained'] = adv

    # Distilled model
    distilled = TrafficNet().to(device)
    distilled.load_state_dict(torch.load('saved_models/distilled_model.pth', map_location=device, weights_only=True))
    distilled.eval()
    models['distilled'] = distilled

    # Detector
    detector = DetectorNet().to(device)
    detector.load_state_dict(torch.load('saved_models/detector_model.pth', map_location=device, weights_only=True))
    detector.eval()
    models['detector'] = detector

    print("[OK] All models loaded")

    # Load test dataset
    test_dataset = TrafficTestDataset(
        root_dir='data/traffic_Data/TEST',
        transform=TEST_TRANSFORM,
    )
    print(f"[OK] Test dataset: {len(test_dataset)} images")

    # Load evaluation results
    results_path = 'results/evaluation_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            eval_results = json.load(f)
        print("[OK] Evaluation results loaded")
    else:
        eval_results = None
        print("[WARN] No evaluation results found — run train_all.py first")


def tensor_to_base64(tensor, amplify=1.0):
    """Convert a tensor image to base64-encoded PNG. Handles RGB (3ch)."""
    img = tensor.detach().cpu()

    if img.dim() == 4:
        img = img.squeeze(0)

    # img is (C, H, W) — C=3 for RGB or C=1 for grayscale
    if img.shape[0] == 3:
        # RGB
        img_np = (img.numpy() * amplify).clip(0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC
        pil_img = Image.fromarray(img_np, mode='RGB')
    else:
        # Grayscale
        img_np = (img.squeeze(0).numpy() * amplify).clip(0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='L')

    pil_img = pil_img.resize((140, 140), Image.NEAREST)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def perturbation_to_base64(tensor):
    """Convert perturbation tensor to visible base64 image (amplified)."""
    img = tensor.detach().cpu()
    if img.dim() == 4:
        img = img.squeeze(0)

    # Average across channels for display
    if img.shape[0] == 3:
        pert_np = img.mean(dim=0).numpy()
    else:
        pert_np = img.squeeze(0).numpy()

    abs_max = max(abs(pert_np.min()), abs(pert_np.max()), 1e-8)
    normalized = (pert_np / abs_max + 1) / 2  # Map [-1,1] to [0,1]
    img_np = (normalized * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode='L')
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

@app.route('/api/labels')
def get_labels():
    """Get class label names."""
    return jsonify(label_names)


@app.route('/api/images', methods=['GET'])
def get_sample_images():
    """Get sample test images for the Attack Lab."""
    count = int(request.args.get('count', 10))
    indices = np.random.choice(len(test_dataset), count, replace=False)

    samples = []
    for idx in indices:
        image, label = test_dataset[int(idx)]
        name = label_names.get(label, f"Class {label}")
        samples.append({
            'index': int(idx),
            'label': int(label),
            'name': name,
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

    # Get top-5 predictions
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = [{'class': int(i), 'name': label_names.get(int(i), f"Class {i}"),
             'prob': float(probs[i])} for i in top5_idx]

    return jsonify({
        'index': idx,
        'label': int(label),
        'label_name': label_names.get(int(label), f"Class {label}"),
        'prediction': pred,
        'pred_name': label_names.get(pred, f"Class {pred}"),
        'top5': top5,
        'image': tensor_to_base64(image),
    })


@app.route('/api/attack', methods=['POST'])
def run_attack():
    """Run a single adversarial attack."""
    data = request.json
    attack_type = data.get('attack_type', 'fgsm')
    epsilon = float(data.get('epsilon', 0.03))
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

        for def_name, def_model_key in [('adv_training', 'adv_trained'),
                                         ('distillation', 'distilled')]:
            def_model = models[def_model_key]
            def_model.eval()
            with torch.no_grad():
                out = def_model(adv_tensor.unsqueeze(0))
                pred = out.argmax(1).item()
                probs = F.softmax(out, dim=1)[0].cpu().numpy()
            top5_idx = np.argsort(probs)[-5:][::-1]
            defense_results[def_name] = {
                'prediction': pred,
                'pred_name': label_names.get(pred, f"Class {pred}"),
                'correct': pred == label,
                'top5': [{'class': int(i), 'name': label_names.get(int(i), ''),
                          'prob': float(probs[i])} for i in top5_idx],
            }

        # Input transformation defense
        with torch.no_grad():
            transformed = apply_input_transforms(adv_tensor.unsqueeze(0))
            out = models['base'](transformed)
            pred = out.argmax(1).item()
            probs = F.softmax(out, dim=1)[0].cpu().numpy()
        top5_idx = np.argsort(probs)[-5:][::-1]
        defense_results['input_transform'] = {
            'prediction': pred,
            'pred_name': label_names.get(pred, f"Class {pred}"),
            'correct': pred == label,
            'top5': [{'class': int(i), 'name': label_names.get(int(i), ''),
                      'prob': float(probs[i])} for i in top5_idx],
        }

        # Detection defense
        with torch.no_grad():
            features = models['base'].get_features(adv_tensor.unsqueeze(0))
            det_out = models['detector'](features)
            det_probs = F.softmax(det_out, dim=1)[0]
            detected = det_probs[1].item() > 0.5
        defense_results['detection'] = {
            'detected': detected,
            'detection_confidence': det_probs[1].item(),
            'correct': detected,
        }

        # Get top-5 for orig and adv
        orig_probs = result['orig_probs']
        adv_probs = result['adv_probs']
        orig_top5 = [{'class': int(i), 'name': label_names.get(int(i), ''),
                       'prob': float(orig_probs[i])}
                      for i in np.argsort(orig_probs)[-5:][::-1]]
        adv_top5 = [{'class': int(i), 'name': label_names.get(int(i), ''),
                      'prob': float(adv_probs[i])}
                     for i in np.argsort(adv_probs)[-5:][::-1]]

        response = {
            'success': True,
            'attack_type': attack_type,
            'epsilon': epsilon,
            'true_label': int(label),
            'true_label_name': label_names.get(int(label), f"Class {label}"),
            'orig_pred': int(result['orig_pred']),
            'orig_pred_name': label_names.get(int(result['orig_pred']), ''),
            'adv_pred': int(result['adv_pred']),
            'adv_pred_name': label_names.get(int(result['adv_pred']), ''),
            'attack_success': bool(result['success']),
            'orig_top5': orig_top5,
            'adv_top5': adv_top5,
            'l_inf': float(result['l_inf']),
            'l2': float(result['l2']),
            'time': round(elapsed, 3),
            'original_image': tensor_to_base64(result['original']),
            'adversarial_image': tensor_to_base64(result['adversarial']),
            'perturbation_image': perturbation_to_base64(result['perturbation']),
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
    return jsonify({'error': 'No results available. Run train_all.py first.'}), 404


if __name__ == '__main__':
    load_models()
    print("\n🚀 Starting web server at http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
