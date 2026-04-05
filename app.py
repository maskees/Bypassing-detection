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
import torch.nn.functional as F
from torchvision import datasets, transforms
from flask import Flask, render_template, jsonify, request
import numpy as np
import base64
import io
import json
import os
import time
from PIL import Image

from models.target_model import MNISTNet, DetectorNet
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


def load_models():
    """Load all trained models."""
    global models, test_dataset, eval_results

    print(f"Loading models on {device}...")

    # Base model
    base = MNISTNet().to(device)
    base.load_state_dict(torch.load('saved_models/base_model.pth', map_location=device, weights_only=True))
    base.eval()
    models['base'] = base

    # Adversarially trained model
    adv = MNISTNet().to(device)
    adv.load_state_dict(torch.load('saved_models/adv_trained_model.pth', map_location=device, weights_only=True))
    adv.eval()
    models['adv_trained'] = adv

    # Distilled model
    distilled = MNISTNet().to(device)
    distilled.load_state_dict(torch.load('saved_models/distilled_model.pth', map_location=device, weights_only=True))
    distilled.eval()
    models['distilled'] = distilled

    # Detector
    detector = DetectorNet().to(device)
    detector.load_state_dict(torch.load('saved_models/detector_model.pth', map_location=device, weights_only=True))
    detector.eval()
    models['detector'] = detector

    print("✓ All models loaded")

    # Load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)
    print(f"✓ Test dataset: {len(test_dataset)} images")

    # Load evaluation results
    results_path = 'results/evaluation_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            eval_results = json.load(f)
        print("✓ Evaluation results loaded")
    else:
        eval_results = None
        print("⚠ No evaluation results found — run train_all.py first")


def tensor_to_base64(tensor, amplify=1.0):
    """Convert a tensor image to base64-encoded PNG."""
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)   # Remove channel dim for grayscale

    img_np = (tensor.detach().cpu().numpy() * amplify).clip(0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode='L')
    pil_img = pil_img.resize((140, 140), Image.NEAREST)

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def perturbation_to_base64(tensor):
    """Convert perturbation tensor to visible base64 image (amplified + colorized)."""
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    pert_np = tensor.detach().cpu().numpy()
    # Normalize to [0, 1] for visualization
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

@app.route('/api/images', methods=['GET'])
def get_sample_images():
    """Get sample test images for the Attack Lab."""
    count = int(request.args.get('count', 10))
    indices = np.random.choice(len(test_dataset), count, replace=False)

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
            defense_results[def_name] = {
                'prediction': pred,
                'correct': pred == label,
                'probabilities': probs.tolist(),
            }

        # Input transformation defense
        with torch.no_grad():
            transformed = apply_input_transforms(adv_tensor.unsqueeze(0))
            out = models['base'](transformed)
            pred = out.argmax(1).item()
            probs = F.softmax(out, dim=1)[0].cpu().numpy()
        defense_results['input_transform'] = {
            'prediction': pred,
            'correct': pred == label,
            'probabilities': probs.tolist(),
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
            'correct': detected,  # If detected, adversarial was caught
        }

        response = {
            'success': True,
            'attack_type': attack_type,
            'epsilon': epsilon,
            'true_label': int(label),
            'orig_pred': int(result['orig_pred']),
            'adv_pred': int(result['adv_pred']),
            'attack_success': bool(result['success']),
            'orig_probs': result['orig_probs'].tolist(),
            'adv_probs': result['adv_probs'].tolist(),
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
