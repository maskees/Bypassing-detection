"""
Detection Network Defense — Reactive defense.

Trains a binary classifier on intermediate features extracted from the
target model to distinguish clean inputs from adversarial examples.

If the detector flags an input as adversarial, it is rejected (not classified).
This prevents adversarial examples from reaching the main classifier.

Architecture:
  Target Model → get_features() → DetectorNet → {clean, adversarial}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def generate_detection_dataset(target_model, data_loader, epsilon=0.3, device='cuda'):
    """
    Generate a dataset of clean and adversarial feature pairs for detector training.

    Uses both FGSM and PGD to create diverse adversarial examples.
    """
    target_model.eval()
    features_list = []
    labels_list = []  # 0 = clean, 1 = adversarial

    print("  Generating detection dataset...")

    for batch_idx, (images, true_labels) in enumerate(data_loader):
        images, true_labels = images.to(device), true_labels.to(device)

        with torch.no_grad():
            clean_features = target_model.get_features(images)
            features_list.append(clean_features.cpu())
            labels_list.append(torch.zeros(images.size(0), dtype=torch.long))

        # FGSM adversarial examples
        adv_fgsm, _, _ = fgsm_attack(target_model, images, true_labels, epsilon, device)
        with torch.no_grad():
            adv_features_fgsm = target_model.get_features(adv_fgsm)
            features_list.append(adv_features_fgsm.cpu())
            labels_list.append(torch.ones(images.size(0), dtype=torch.long))

        # PGD adversarial examples (every other batch to save time)
        if batch_idx % 2 == 0:
            adv_pgd, _, _ = pgd_attack(target_model, images, true_labels,
                                       epsilon, steps=10, device=device)
            with torch.no_grad():
                adv_features_pgd = target_model.get_features(adv_pgd)
                features_list.append(adv_features_pgd.cpu())
                labels_list.append(torch.ones(images.size(0), dtype=torch.long))

        if batch_idx >= 30:  # Limit dataset size for speed
            break

    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    print(f"  Detection dataset: {all_features.shape[0]} samples "
          f"({(all_labels == 0).sum()} clean, {(all_labels == 1).sum()} adversarial)")

    return all_features, all_labels


def train_detector(target_model, detector_model, train_loader,
                   epsilon=0.3, epochs=10, lr=0.001, device='cuda'):
    """
    Train the detection network.

    Args:
        target_model: Pre-trained target classifier (fixed)
        detector_model: DetectorNet to train
        train_loader: Training data loader
        epsilon: Epsilon for generating adversarial training data
        epochs: Training epochs
        lr: Learning rate
        device: Computation device

    Returns:
        Trained detector model
    """
    target_model.eval()
    detector_model = detector_model.to(device)

    print("\n[Defense] Training Detection Network")

    # Generate detection dataset
    features, labels = generate_detection_dataset(
        target_model, train_loader, epsilon, device
    )

    # Create data loader for detection training
    dataset = torch.utils.data.TensorDataset(features, labels)
    det_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    optimizer = optim.Adam(detector_model.parameters(), lr=lr)

    for epoch in range(epochs):
        detector_model.train()
        total_loss = 0
        correct = 0
        total = 0

        for feat_batch, label_batch in det_loader:
            feat_batch = feat_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            outputs = detector_model(feat_batch)
            loss = F.cross_entropy(outputs, label_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label_batch.size(0)
            correct += predicted.eq(label_batch).sum().item()

        acc = 100. * correct / total
        avg_loss = total_loss / len(det_loader)
        print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, "
              f"Detection Acc: {acc:.2f}%")

    return detector_model


def detect_and_predict(target_model, detector, images, device='cuda',
                       detection_threshold=0.5):
    """
    Use detection network to filter adversarial inputs, then classify.

    Args:
        target_model: Main classifier
        detector: Trained detection network
        images: Input images
        device: Computation device
        detection_threshold: Probability threshold for flagging as adversarial

    Returns:
        predictions: Predicted labels (-1 for rejected/detected adversarial)
        is_adversarial: Boolean mask of detected adversarial inputs
        detection_probs: Probability of being adversarial for each input
    """
    target_model.eval()
    detector.eval()
    images = images.to(device)

    with torch.no_grad():
        # Extract features and detect
        features = target_model.get_features(images)
        det_output = detector(features)
        det_probs = F.softmax(det_output, dim=1)
        is_adversarial = det_probs[:, 1] > detection_threshold

        # Classify with target model
        class_output = target_model(images)
        predictions = class_output.argmax(dim=1)

        # Reject detected adversarial inputs (set prediction to -1)
        predictions[is_adversarial] = -1

    return predictions, is_adversarial, det_probs[:, 1]
