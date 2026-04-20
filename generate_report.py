"""
Comprehensive Project Report — Bypassing Detection
Generates a professional PDF covering the entire workflow.
"""
import json
from fpdf import FPDF

# ── Load evaluation results ──
with open("results/evaluation_results.json") as f:
    data = json.load(f)


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Bypassing Detection - Project Report", align="L")
            self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(30, 60, 100)
            self.line(10, 14, 200, 14)
            self.ln(4)

    def footer(self):
        pass

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(15, 43, 79)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(31, 135, 224)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 80, self.get_y())
        self.set_line_width(0.2)
        self.ln(6)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(31, 135, 224)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def bullet(self, text, indent=15):
        x = self.get_x()
        self.set_font("Helvetica", "", 11)
        self.set_text_color(40, 40, 40)
        self.set_x(x + indent)
        self.cell(5, 6, "-", new_x="END")
        self.multi_cell(0, 6, f"  {text}")
        self.ln(1)

    def code_block(self, code, title=None):
        if title:
            self.set_font("Helvetica", "BI", 10)
            self.set_text_color(80, 80, 80)
            self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)
        self.set_fill_color(240, 243, 247)
        self.set_font("Courier", "", 9)
        self.set_text_color(30, 30, 30)
        x0, y0 = self.get_x(), self.get_y()
        # Draw background
        lines = code.strip().split("\n")
        block_h = len(lines) * 5 + 6
        if self.get_y() + block_h > 270:
            self.add_page()
            y0 = self.get_y()
        self.rect(10, y0, 190, block_h, style="F")
        self.set_xy(13, y0 + 3)
        for line in lines:
            self.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
            self.set_x(13)
        self.ln(4)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(15, 43, 79)
        self.set_text_color(255, 255, 255)
        for i, col in enumerate(cols):
            self.cell(widths[i], 8, col, border=1, fill=True, align="C")
        self.ln()

    def table_row(self, cols, widths, even=False):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        if even:
            self.set_fill_color(244, 247, 250)
        else:
            self.set_fill_color(255, 255, 255)
        for i, col in enumerate(cols):
            self.cell(widths[i], 7, str(col), border=1, fill=True, align="C")
        self.ln()


pdf = Report()
pdf.set_auto_page_break(auto=True, margin=20)

# ═══════════════════════════════════════════════════════
# TITLE PAGE
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.ln(50)
pdf.set_font("Helvetica", "B", 36)
pdf.set_text_color(15, 43, 79)
pdf.cell(0, 15, "BYPASSING DETECTION", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_draw_color(31, 135, 224)
pdf.set_line_width(1.2)
pdf.line(50, pdf.get_y(), 160, pdf.get_y())
pdf.set_line_width(0.2)
pdf.ln(8)
pdf.set_font("Helvetica", "", 18)
pdf.set_text_color(90, 106, 123)
pdf.cell(0, 10, "Adversarial Attacks & Defenses", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 10, "on Road Sign Classifiers", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(15)
pdf.set_font("Helvetica", "I", 14)
pdf.set_text_color(31, 135, 224)
pdf.cell(0, 8, "4 Attack Methods  |  5 Defense Strategies  |  ResNet-34  |  GTSRB Dataset", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(25)
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(90, 106, 123)
pdf.cell(0, 8, "Comprehensive Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "PyTorch  |  Flask  |  Altair HPC (H100)", align="C", new_x="LMARGIN", new_y="NEXT")

# ═══════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("Table of Contents")
pdf.ln(5)
toc = [
    ("1.", "Introduction & Motivation"),
    ("2.", "Dataset - GTSRB"),
    ("3.", "Model Architecture - Why ResNet-34"),
    ("4.", "Attack Methods"),
    ("  4.1", "FGSM (Fast Gradient Sign Method)"),
    ("  4.2", "PGD (Projected Gradient Descent)"),
    ("  4.3", "Genetic Algorithm"),
    ("  4.4", "Differential Evolution"),
    ("5.", "Defense Strategies"),
    ("  5.1", "Adversarial Training"),
    ("  5.2", "Defensive Distillation"),
    ("  5.3", "Input Transformation"),
    ("  5.4", "Detection Network"),
    ("6.", "Training Pipeline"),
    ("7.", "Evaluation Methodology"),
    ("8.", "Results & Analysis"),
    ("9.", "Web Dashboard"),
    ("10.", "Key Findings"),
    ("11.", "Conclusion & Future Work"),
    ("A.", "Appendix - Project Structure"),
]
for num, title in toc:
    pdf.set_font("Helvetica", "B" if not num.startswith(" ") else "", 12)
    pdf.set_text_color(15, 43, 79)
    indent = 5 if num.startswith(" ") else 0
    pdf.set_x(15 + indent)
    pdf.cell(15, 8, num)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")

# ═══════════════════════════════════════════════════════
# 1. INTRODUCTION
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("1. Introduction & Motivation")

pdf.body_text(
    "Deep neural networks, particularly Convolutional Neural Networks (CNNs), have become the "
    "backbone of modern autonomous driving perception systems. These models achieve remarkable "
    "accuracy in recognizing road signs, traffic lights, and other critical objects. However, "
    "they are vulnerable to adversarial attacks - carefully crafted, often imperceptible "
    "perturbations to input images that cause the model to misclassify with high confidence."
)

pdf.body_text(
    "This project, 'Bypassing Detection,' investigates the robustness of a road sign classifier "
    "against four adversarial attack methods and evaluates five defense strategies. The goal is "
    "to understand which attacks are most effective, which defenses hold up, and whether any "
    "single defense can provide universal protection."
)

pdf.section_title("Problem Statement")
pdf.body_text(
    "An adversarial attacker can modify a stop sign image by changing pixel values by as little "
    "as 2-3% such that a CNN classifier reads it as a speed limit sign. This is invisible to "
    "the human eye but can have catastrophic consequences in autonomous driving. We quantify "
    "this vulnerability and evaluate countermeasures."
)

pdf.section_title("Research Questions")
pdf.bullet("How effective are gradient-based vs. evolutionary attacks at fooling a road sign classifier?")
pdf.bullet("Which defense strategies provide the best robustness against each attack type?")
pdf.bullet("Is there a single defense that works universally against all attacks?")
pdf.bullet("What is the trade-off between clean accuracy and robust accuracy for each defense?")

pdf.section_title("Scope")
pdf.body_text(
    "We implement 4 attacks (2 gradient-based ML, 2 evolutionary computation) and 5 defenses "
    "(including a no-defense baseline), resulting in 20 unique attack-defense combinations. "
    "All experiments use the GTSRB dataset mapped to 4 road sign classes, trained on an "
    "NVIDIA H100 GPU via Altair HPC."
)

# ═══════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("2. Dataset - GTSRB")

pdf.section_title("About GTSRB")
pdf.body_text(
    "The German Traffic Sign Recognition Benchmark (GTSRB) is a large-scale, multi-class "
    "dataset of traffic sign images captured in real-world conditions. It contains over 50,000 "
    "images across 43 classes, with varying lighting, weather, occlusion, and resolution. "
    "It is the standard benchmark for traffic sign classification research."
)

pdf.section_title("Class Mapping: 43 to 4 Classes")
pdf.body_text(
    "We mapped the original 43 GTSRB classes to 4 target classes relevant to our road sign "
    "adversarial attack study:"
)

widths = [40, 80, 40]
pdf.table_header(["Target Class", "GTSRB Source Classes", "# Source"], widths)
pdf.table_row(["crosswalk", "Pedestrians, Children crossing, ...", "3"], widths, True)
pdf.table_row(["speedlimit", "Speed limit 20-120 km/h", "12"], widths)
pdf.table_row(["stop", "Stop sign", "1"], widths, True)
pdf.table_row(["trafficlight", "Traffic signals, General caution", "3"], widths)
pdf.ln(4)

pdf.section_title("Balancing Strategy")
pdf.body_text(
    "The raw GTSRB data is highly imbalanced (e.g., speedlimit has 8,610 training images vs. "
    "trafficlight with only 420). To prevent the model from biasing toward majority classes, "
    "we balanced all classes to the minimum count:"
)

widths = [50, 40, 40]
pdf.table_header(["Class", "Train", "Test"], widths)
pdf.table_row(["crosswalk", "420", "180"], widths, True)
pdf.table_row(["speedlimit", "420", "180"], widths)
pdf.table_row(["stop", "420", "180"], widths, True)
pdf.table_row(["trafficlight", "420", "180"], widths)
pdf.table_row(["TOTAL", "1,680", "720"], widths, True)
pdf.ln(4)

pdf.body_text(
    "Images were resized to 224x224 pixels and organized into an ImageFolder structure: "
    "data/GTSRB_mapped/{train,test}/{class_name}/*.png"
)

pdf.section_title("Data Augmentation")
pdf.body_text("During training, the following augmentations were applied:")
pdf.bullet("Random horizontal flip")
pdf.bullet("Random rotation (+/- 15 degrees)")
pdf.bullet("Color jitter (brightness, contrast, saturation)")
pdf.bullet("Normalization to ImageNet mean/std via NormalizedModel wrapper")

# ═══════════════════════════════════════════════════════
# 3. MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("3. Model Architecture")

pdf.section_title("Why ResNet-34?")
pdf.body_text(
    "We chose ResNet-34 as our backbone for several reasons:"
)
pdf.bullet("Proven performance: ResNet architectures dominate image classification benchmarks.")
pdf.bullet("Transfer learning: ImageNet-pretrained weights provide strong feature initialization, "
           "critical for our small balanced dataset (1,680 training images).")
pdf.bullet("Feature dimensionality: ResNet-34 produces 512-dimensional feature vectors, "
           "which our detection network uses for adversarial example detection.")
pdf.bullet("Gradient accessibility: ResNet's differentiable architecture enables gradient-based "
           "attacks (FGSM, PGD) and adversarial training.")
pdf.bullet("Efficiency: 34 layers provide a good accuracy-speed trade-off compared to ResNet-50/101.")

pdf.section_title("RoadSignClassifier")
pdf.body_text(
    "Our main classifier wraps a pretrained ResNet-34, replacing the final fully-connected "
    "layer with a 4-class output head. It supports feature extraction for the detection network."
)
pdf.code_block(
    "class RoadSignClassifier(nn.Module):\n"
    "    def __init__(self, num_classes=4, backbone='resnet34', pretrained=False):\n"
    "        super().__init__()\n"
    "        model = models.resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)\n"
    "        self._feature_dim = model.fc.in_features  # 512\n"
    "        model.fc = nn.Linear(self._feature_dim, num_classes)\n"
    "        self.model = model\n"
    "\n"
    "    def get_features(self, x):   # Used by DetectorNet\n"
    "        # Returns 512-dim vector before final FC layer\n"
    "        ...\n"
    "\n"
    "    def forward(self, x):\n"
    "        return self.model(x)     # Returns 4-class logits",
    "RoadSignClassifier (road_sign_classifier.py)"
)

pdf.section_title("NormalizedModel Wrapper")
pdf.body_text(
    "Adversarial attacks operate on raw pixel values in [0, 1]. However, the ResNet backbone "
    "expects ImageNet-normalized inputs. The NormalizedModel wrapper solves this by internally "
    "normalizing inputs before passing them to the backbone. This ensures attacks perturb "
    "the actual pixel space, not the normalized space."
)
pdf.code_block(
    "class NormalizedModel(nn.Module):\n"
    "    def __init__(self, model, mean=IMAGENET_MEAN, std=IMAGENET_STD):\n"
    "        super().__init__()\n"
    "        self.model = model\n"
    "        self.register_buffer('mean', torch.tensor(mean).view(1,3,1,1))\n"
    "        self.register_buffer('std',  torch.tensor(std).view(1,3,1,1))\n"
    "\n"
    "    def forward(self, x):           # x in [0,1]\n"
    "        return self.model((x - self.mean) / self.std)",
    "NormalizedModel (road_sign_classifier.py)"
)

pdf.section_title("DetectorNet")
pdf.body_text(
    "A binary classifier that detects adversarial examples by analyzing the 512-dimensional "
    "feature vectors extracted from the ResNet backbone. It classifies inputs as either 'clean' "
    "or 'adversarial'."
)
pdf.code_block(
    "class DetectorNet(nn.Module):\n"
    "    def __init__(self, input_dim=512):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(input_dim, 256)\n"
    "        self.bn1 = nn.BatchNorm1d(256)\n"
    "        self.fc2 = nn.Linear(256, 64)\n"
    "        self.bn2 = nn.BatchNorm1d(64)\n"
    "        self.fc3 = nn.Linear(64, 2)       # clean vs adversarial\n"
    "        self.dropout = nn.Dropout(0.5)",
    "DetectorNet (target_model.py)"
)

# ═══════════════════════════════════════════════════════
# 4. ATTACK METHODS
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("4. Attack Methods")

pdf.body_text(
    "We implement four adversarial attack methods: two gradient-based (ML) and two evolutionary "
    "(EC). All attacks operate in the L-infinity threat model, bounding the maximum per-pixel "
    "change to epsilon."
)

# 4.1 FGSM
pdf.section_title("4.1 FGSM (Fast Gradient Sign Method)")
pdf.body_text(
    "FGSM is a single-step, gradient-based attack proposed by Goodfellow et al. (2015). "
    "It computes the gradient of the loss function with respect to the input image and "
    "perturbs each pixel in the direction of the gradient sign."
)
pdf.body_text("Formula:  x_adv = x + epsilon * sign(grad_x L(theta, x, y))")
pdf.body_text(
    "Properties: Fast (single forward + backward pass), but relatively weak. At higher epsilon "
    "values, we use a targeted variant that aims for a random wrong class to produce diverse "
    "misclassifications rather than collapsing all outputs to one class."
)
pdf.code_block(
    "def fgsm_attack(model, images, labels, epsilon, device='cuda'):\n"
    "    images.requires_grad_(True)\n"
    "    outputs = model(images)\n"
    "    loss = F.cross_entropy(outputs, labels)\n"
    "    loss.backward()\n"
    "    grad_sign = images.grad.data.sign()\n"
    "    adv_images = torch.clamp(images + epsilon * grad_sign, 0.0, 1.0)\n"
    "    return adv_images",
    "FGSM core logic (attacks/fgsm.py)"
)

# 4.2 PGD
pdf.section_title("4.2 PGD (Projected Gradient Descent)")
pdf.body_text(
    "PGD (Madry et al., 2018) is an iterative version of FGSM. It starts from a random "
    "point within the epsilon-ball and applies multiple small gradient steps, projecting "
    "back onto the constraint set after each step. It is considered the strongest first-order "
    "adversary."
)
pdf.body_text(
    "Algorithm:\n"
    "  1. x' = x + uniform(-eps, eps)        [random start]\n"
    "  2. For each of 40 steps:\n"
    "     x' = x' + alpha * sign(grad L)     [gradient step]\n"
    "     x' = clip(x', x-eps, x+eps)        [project to eps-ball]\n"
    "     x' = clip(x', 0, 1)                [valid pixel range]"
)
pdf.body_text("Default: 40 steps, alpha = epsilon/4. Stronger than FGSM but 40x slower.")

# 4.3 Genetic Algorithm
pdf.add_page()
pdf.section_title("4.3 Genetic Algorithm (EC)")
pdf.body_text(
    "The Genetic Algorithm (GA) is a black-box evolutionary attack that does NOT use gradients. "
    "It evolves a population of perturbations using selection, crossover, and mutation to find "
    "adversarial examples. This makes it effective against gradient-masking defenses."
)
pdf.body_text(
    "Algorithm:\n"
    "  1. Initialize population of random perturbations within eps-ball\n"
    "  2. Evaluate fitness: how much the perturbation reduces correct-class confidence\n"
    "  3. Select top-performing individuals (tournament selection)\n"
    "  4. Crossover: blend pairs of perturbations\n"
    "  5. Mutation: randomly modify pixels\n"
    "  6. Repeat for N generations"
)
pdf.body_text("Default: population=30, generations=50. Slower than gradient attacks but bypasses gradient defenses.")

# 4.4 DE
pdf.section_title("4.4 Differential Evolution (EC)")
pdf.body_text(
    "Differential Evolution (DE) is another black-box evolutionary attack. Unlike GA, DE uses "
    "difference vectors between population members to create mutations, providing better "
    "exploration of the perturbation space. It is our strongest attack overall."
)
pdf.body_text(
    "Algorithm:\n"
    "  1. Initialize population within eps-ball\n"
    "  2. For each individual, create a mutant: v = a + F*(b - c)\n"
    "  3. Crossover: blend mutant with current individual\n"
    "  4. Selection: keep whichever is fitter\n"
    "  5. Repeat for N iterations"
)
pdf.body_text("Default: max_iter=50. Slowest attack (4.5s/image) but highest attack success rate.")

# Comparison table
pdf.section_title("Attack Comparison")
widths = [35, 30, 30, 30, 35]
pdf.table_header(["Attack", "Type", "Gradients?", "Speed", "Strength"], widths)
pdf.table_row(["FGSM", "ML", "Yes", "0.02s", "Medium"], widths, True)
pdf.table_row(["PGD", "ML", "Yes", "0.15s", "Strong"], widths)
pdf.table_row(["Genetic", "EC", "No", "1.3s", "Strong"], widths, True)
pdf.table_row(["Diff. Evo.", "EC", "No", "4.5s", "Strongest"], widths)

# ═══════════════════════════════════════════════════════
# 5. DEFENSE STRATEGIES
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("5. Defense Strategies")

pdf.body_text(
    "We implement five defense configurations spanning three categories: proactive "
    "(adversarial training, distillation), preprocessing (input transformation), and "
    "reactive (detection network), plus a no-defense baseline."
)

# 5.0 No Defense
pdf.section_title("5.0 No Defense (Baseline)")
pdf.body_text(
    "The standard ResNet-34 classifier with no adversarial hardening. Serves as the baseline "
    "to measure how much each defense improves robustness. Clean accuracy: 99%."
)

# 5.1 Adversarial Training
pdf.section_title("5.1 Adversarial Training")
pdf.body_text(
    "The model is fine-tuned on a mixture of clean and PGD-adversarial examples. During each "
    "training batch, adversarial counterparts are generated on-the-fly and the model learns "
    "to classify both correctly. This is a proactive defense that directly hardens the model's "
    "decision boundaries."
)
pdf.body_text(
    "Hyperparameters: 5 epochs, PGD with eps=8/255, 7 steps, alpha=2/255. "
    "50% clean + 50% adversarial training mix."
)
pdf.body_text("Result: Clean accuracy 98.2%. Strong against FGSM (9.4% ASR) and PGD (20.5% ASR), "
              "but vulnerable to evolutionary attacks.")

# 5.2 Defensive Distillation
pdf.section_title("5.2 Defensive Distillation")
pdf.body_text(
    "A two-stage defense: first train a 'teacher' model, then train a 'student' model on the "
    "teacher's soft probability outputs (with temperature T=20). The softened labels smooth "
    "the model's output distribution, making gradients less useful for attackers."
)
pdf.body_text(
    "Hyperparameters: T=20, alpha=0.7 (soft label weight), 5 epochs student training."
)
pdf.body_text("Result: Best defense against PGD (0.8% ASR), but almost completely bypassed by DE (97.9% ASR).")

# 5.3 Input Transformation
pdf.section_title("5.3 Input Transformation")
pdf.body_text(
    "A preprocessing defense that applies Gaussian smoothing and median filtering to input "
    "images before classification. The idea is that these filters destroy the high-frequency "
    "adversarial perturbations while preserving the image content. No model retraining needed."
)
pdf.body_text("Result: Moderate effectiveness. 33.3% ASR vs FGSM, but 81.5% ASR vs DE.")

# 5.4 Detection Network
pdf.section_title("5.4 Detection Network")
pdf.body_text(
    "A reactive defense: a separate binary classifier (DetectorNet) is trained to distinguish "
    "clean images from adversarial ones. It operates on the 512-dimensional feature vectors "
    "from the ResNet backbone. If an input is flagged as adversarial, the system rejects it "
    "instead of classifying."
)
pdf.body_text(
    "Training: DetectorNet is trained on feature vectors from clean images (label=0) and "
    "FGSM+PGD adversarial images (label=1). 5 epochs, 50% dropout."
)
pdf.body_text("Result: Perfect against FGSM (0% ASR), good against PGD (8%), but weak against "
              "evolutionary attacks (74-85% ASR) since it was trained on gradient-attack features.")

# ═══════════════════════════════════════════════════════
# 6. TRAINING PIPELINE
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("6. Training Pipeline")

pdf.body_text("All training was performed on Altair HPC with an NVIDIA H100 GPU (80GB).")

pdf.section_title("Stage 1: Base Classifier")
pdf.bullet("Architecture: ResNet-34 pretrained on ImageNet, wrapped in NormalizedModel")
pdf.bullet("Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)")
pdf.bullet("Scheduler: Cosine annealing LR")
pdf.bullet("Loss: Weighted cross-entropy (to handle any residual imbalance)")
pdf.bullet("Epochs: 10")
pdf.bullet("Result: 100% validation accuracy (on balanced GTSRB test set)")

pdf.section_title("Stage 2: Adversarial Training")
pdf.bullet("Start from Stage 1 checkpoint")
pdf.bullet("Generate PGD adversarial examples on-the-fly (eps=8/255, 7 steps)")
pdf.bullet("Train on 50% clean + 50% adversarial mix")
pdf.bullet("Optimizer: SGD (lr=1e-3, momentum=0.9)")
pdf.bullet("Epochs: 5")

pdf.section_title("Stage 3: Defensive Distillation")
pdf.bullet("Teacher: Stage 1 base model")
pdf.bullet("Student: fresh ResNet-34 trained on teacher's soft outputs")
pdf.bullet("Temperature: T=20, alpha=0.7")
pdf.bullet("Optimizer: AdamW (lr=5e-4)")
pdf.bullet("Epochs: 5")

pdf.section_title("Stage 4: Detection Network")
pdf.bullet("Generate adversarial features using FGSM and PGD")
pdf.bullet("Train DetectorNet (512 -> 256 -> 64 -> 2) on clean vs adversarial features")
pdf.bullet("Dropout: 0.5, BatchNorm after each FC layer")
pdf.bullet("Epochs: 5")

pdf.section_title("Stage 5: Full Evaluation")
pdf.bullet("Run all 4 attacks x 5 defenses = 20 combinations")
pdf.bullet("50 samples for gradient attacks, 20 for evolutionary (slower)")
pdf.bullet("Metrics: clean accuracy, robust accuracy, ASR, L-inf, L2, attack time")

pdf.code_block(
    "Pipeline: Image [0,1] -> NormalizedModel -> ResNet-34 -> Softmax -> 4 classes\n"
    "\n"
    "Checkpoint files saved:\n"
    "  road_sign_crop_resnet34.pth      (base classifier)\n"
    "  road_sign_crop_adv_trained.pth   (adversarial training)\n"
    "  road_sign_crop_distilled.pth     (defensive distillation)\n"
    "  road_sign_crop_detector.pth      (detection network)",
    "Model pipeline and checkpoints"
)

# ═══════════════════════════════════════════════════════
# 7. EVALUATION METHODOLOGY
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("7. Evaluation Methodology")

pdf.section_title("Metrics")
pdf.bullet("Clean Accuracy: Model accuracy on unperturbed test images (baseline performance).")
pdf.bullet("Robust Accuracy: Model accuracy on adversarial images (how much survives the attack).")
pdf.bullet("Attack Success Rate (ASR): Percentage of correctly-classified images that the attack "
           "successfully fools. ASR = (1 - robust_acc/clean_acc) * 100. Lower is better for defenders.")
pdf.bullet("Average L-inf: Maximum per-pixel perturbation magnitude (bounded by epsilon).")
pdf.bullet("Average L2: Total perturbation magnitude (Euclidean norm).")
pdf.bullet("Average Attack Time: Wall-clock time per adversarial example.")

pdf.section_title("Evaluation Protocol")
pdf.body_text(
    "For each of the 20 attack-defense combinations:\n"
    "1. Load the defense-specific model checkpoint\n"
    "2. Select N correctly-classified test images (N=50 for ML attacks, N=20 for EC attacks)\n"
    "3. Run the attack with epsilon=0.3 (L-inf bound)\n"
    "4. Record metrics: was the attack successful? What was the perturbation norm? Time?\n"
    "5. Aggregate results across all samples"
)

pdf.body_text(
    "Epsilon = 0.3 is a relatively strong perturbation bound (30% of pixel range). "
    "This stress-tests defenses and reveals their breaking points."
)

# ═══════════════════════════════════════════════════════
# 8. RESULTS
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("8. Results & Analysis")

pdf.section_title("Attack Success Rate (ASR) Matrix")
pdf.body_text("Lower ASR = better defense. Values show % of images successfully fooled.")
pdf.ln(2)

attacks_list = ["fgsm", "pgd", "genetic", "de"]
defenses_list = ["none", "adv_training", "input_transform", "detection", "distillation"]
attack_labels = ["FGSM", "PGD", "Genetic", "Diff. Evo."]
defense_labels = ["No Def.", "Adv Train", "Input Tr.", "Detection", "Distill."]

widths = [28, 28, 28, 28, 32, 32]
pdf.table_header(["Attack"] + defense_labels, widths)
for i, akey in enumerate(attacks_list):
    row = [attack_labels[i]]
    for dkey in defenses_list:
        asr = data["results"][akey][dkey].get("attack_success_rate", 0)
        row.append(f"{asr:.1f}%")
    pdf.table_row(row, widths, i % 2 == 0)

pdf.ln(4)
pdf.section_title("Robust Accuracy Matrix")
pdf.body_text("Higher robust accuracy = better defense. Values show % of images still correctly classified after attack.")
pdf.ln(2)

pdf.table_header(["Attack"] + defense_labels, widths)
for i, akey in enumerate(attacks_list):
    row = [attack_labels[i]]
    for dkey in defenses_list:
        ra = data["results"][akey][dkey].get("robust_accuracy", 0)
        row.append(f"{ra:.1f}%")
    pdf.table_row(row, widths, i % 2 == 0)

pdf.ln(4)
pdf.section_title("Clean Accuracy per Defense")
widths2 = [50, 40]
pdf.table_header(["Defense", "Clean Acc."], widths2)
for i, dkey in enumerate(defenses_list):
    ca = data["results"]["fgsm"][dkey].get("clean_accuracy", 0)
    dname = ["No Defense", "Adv. Training", "Input Transform", "Detection", "Distillation"][i]
    pdf.table_row([dname, f"{ca:.1f}%"], widths2, i % 2 == 0)

pdf.ln(4)
pdf.section_title("Attack Speed Comparison")
widths3 = [35, 30, 30, 35]
pdf.table_header(["Attack", "Avg Time", "Avg L-inf", "Avg L2"], widths3)
for i, akey in enumerate(attacks_list):
    r = data["results"][akey]["none"]
    pdf.table_row([
        attack_labels[i],
        f"{r['avg_time']:.2f}s",
        f"{r['avg_l_inf']:.3f}",
        f"{r['avg_l2']:.1f}"
    ], widths3, i % 2 == 0)

# ═══════════════════════════════════════════════════════
# 9. WEB DASHBOARD
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("9. Web Dashboard")

pdf.body_text(
    "We built an interactive web dashboard using Flask + HTML/CSS/JS that allows users to "
    "explore adversarial attacks and defenses in real-time."
)

pdf.section_title("Features")
pdf.bullet("Image Gallery: Browse and select correctly-classified road sign images from the test set")
pdf.bullet("Attack Configuration: Choose attack method (FGSM/PGD/GA/DE) and adjust epsilon with a slider")
pdf.bullet("Real-time Attack: Run the selected attack and see the adversarial image generated live")
pdf.bullet("Perturbation Visualization: See the difference between original and adversarial (magnified 5x)")
pdf.bullet("Confidence Bars: Compare model confidence distributions before and after attack")
pdf.bullet("Defense Comparison: All 5 defenses evaluated simultaneously on the adversarial example")

pdf.section_title("Architecture")
pdf.code_block(
    "Frontend:  HTML/CSS/JS with modern dark theme\n"
    "Backend:   Flask (Python) serving REST API\n"
    "Models:    Loaded once at startup, kept in GPU memory\n"
    "API:       /api/attack (POST) - runs attack and all defenses\n"
    "           /api/sample_images (GET) - returns random test images\n"
    "           /api/models_status (GET) - reports loaded models",
    "Web dashboard architecture"
)

pdf.section_title("Three Output Images Explained")
pdf.bullet("Original: The clean, unmodified input image with its true label.")
pdf.bullet("Adversarial: The attacked image - looks identical to humans but may fool the model. "
           "Pixel values differ by at most epsilon from the original.")
pdf.bullet("Perturbation (x5): The difference between original and adversarial, magnified 5 times. "
           "Gray = no change, colored = modified pixels. Shows where the attack focused.")

# ═══════════════════════════════════════════════════════
# 10. KEY FINDINGS
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("10. Key Findings")

pdf.section_title("Finding 1: Gradient defenses fail against evolutionary attacks")
pdf.body_text(
    "Adversarial training and distillation were designed to counter gradient-based attacks. "
    "They excel against FGSM and PGD (ASR < 21%) but are nearly useless against GA and DE "
    "(ASR > 59%). This is because evolutionary attacks do not use gradients at all - they "
    "treat the model as a black box."
)

pdf.section_title("Finding 2: Distillation is best against PGD, worst against DE")
pdf.body_text(
    "Defensive distillation achieves the best single result: 0.8% ASR against PGD. "
    "The soft labels at T=20 effectively mask gradient information. However, DE bypasses "
    "this completely (97.9% ASR) because it never queries gradients."
)

pdf.section_title("Finding 3: Detection Network is perfect against FGSM")
pdf.body_text(
    "The DetectorNet achieves 0% ASR against FGSM (catches every adversarial example) and "
    "8% against PGD. However, since it was trained on FGSM/PGD features, it doesn't "
    "generalize to evolutionary attack patterns (74-85% ASR)."
)

pdf.section_title("Finding 4: No single defense is universal")
pdf.body_text(
    "Every defense has blind spots. The best defense varies by attack type:\n"
    "  - vs FGSM:   Detection Network (0% ASR)\n"
    "  - vs PGD:    Distillation (0.8% ASR)\n"
    "  - vs GA:     Adversarial Training (59.2% ASR) - still weak\n"
    "  - vs DE:     Adversarial Training (71.4% ASR) - still weak\n"
    "This strongly suggests that layered/ensemble defenses are needed."
)

pdf.section_title("Finding 5: Evolutionary attacks are the strongest")
pdf.body_text(
    "Differential Evolution achieves the highest ASR across all defenses (avg 86.7%). "
    "Even the best defense (adversarial training) only reduces DE's ASR to 71.4%. "
    "GA is similarly strong (avg 75.3%). The trade-off is speed: DE takes 4.5s per image "
    "vs 0.02s for FGSM."
)

# ═══════════════════════════════════════════════════════
# 11. CONCLUSION
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("11. Conclusion & Future Work")

pdf.section_title("Conclusion")
pdf.bullet("Successfully implemented and evaluated 20 attack-defense combinations on a road sign classifier.")
pdf.bullet("Achieved 100% clean accuracy with balanced GTSRB dataset (4 classes, 420 images/class).")
pdf.bullet("Demonstrated that gradient-based defenses (adv training, distillation) are effective against "
           "gradient attacks but fail against evolutionary methods.")
pdf.bullet("Showed that black-box evolutionary attacks (GA, DE) can bypass most defenses because they "
           "don't rely on model internals.")
pdf.bullet("Built an interactive web dashboard for real-time attack visualization and defense comparison.")
pdf.bullet("No single defense provides universal robustness - this is a fundamental challenge in adversarial ML.")

pdf.section_title("Future Work")
pdf.bullet("Ensemble Defenses: Combine multiple defenses (e.g., adversarial training + detection + "
           "input transformation) for layered protection.")
pdf.bullet("Certified Robustness: Use randomized smoothing to provide provable robustness guarantees "
           "within a certified radius.")
pdf.bullet("Adaptive Attacks: Implement attacks that are specifically designed to defeat each defense, "
           "providing a more rigorous evaluation.")
pdf.bullet("Larger Dataset: Train on the full GTSRB (43 classes) or combine with other traffic sign "
           "datasets for better generalization.")
pdf.bullet("Real-World Testing: Evaluate attacks on physically printed signs photographed in varying conditions.")

# ═══════════════════════════════════════════════════════
# APPENDIX
# ═══════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("A. Appendix - Project Structure")

pdf.code_block(
    "Bypassing-Detection/\n"
    "|\n"
    "|-- app.py                     Flask web dashboard\n"
    "|-- road_sign_data.py          Dataset loading & preprocessing\n"
    "|-- train_models.ipynb         Training notebook (all 5 stages)\n"
    "|-- setup_gtsrb.ipynb          GTSRB download & class mapping\n"
    "|\n"
    "|-- models/\n"
    "|   |-- road_sign_classifier.py   RoadSignClassifier + NormalizedModel\n"
    "|   |-- road_sign_model.py        RoadSignResNet (with bbox head)\n"
    "|   |-- target_model.py           DetectorNet (adversarial detector)\n"
    "|\n"
    "|-- attacks/\n"
    "|   |-- fgsm.py                   FGSM attack\n"
    "|   |-- pgd.py                    PGD attack\n"
    "|   |-- genetic_attack.py         Genetic Algorithm attack\n"
    "|   |-- differential_evolution_attack.py   DE attack\n"
    "|\n"
    "|-- defenses/\n"
    "|   |-- adversarial_training.py   Adv training defense\n"
    "|   |-- defensive_distillation.py Distillation defense\n"
    "|   |-- input_transformation.py   Input transform defense\n"
    "|   |-- detection_network.py      Detection network defense\n"
    "|\n"
    "|-- evaluation/\n"
    "|   |-- evaluator.py              Full evaluation pipeline\n"
    "|\n"
    "|-- templates/                  HTML templates for dashboard\n"
    "|-- static/                    CSS/JS for dashboard\n"
    "|-- saved_models/              Model checkpoints (.pth)\n"
    "|-- results/                   Evaluation results (.json)\n"
    "|-- data/GTSRB_mapped/         Balanced dataset (train/test)",
    "Project directory structure"
)

pdf.section_title("Tech Stack")
widths4 = [50, 100]
pdf.table_header(["Component", "Technology"], widths4)
pdf.table_row(["Deep Learning", "PyTorch 2.6, torchvision, CUDA 12.1"], widths4, True)
pdf.table_row(["Model", "ResNet-34 (ImageNet pretrained)"], widths4)
pdf.table_row(["Web Framework", "Flask + HTML/CSS/JS"], widths4, True)
pdf.table_row(["Compute", "Altair HPC, NVIDIA H100 80GB"], widths4)
pdf.table_row(["Dataset", "GTSRB (43 -> 4 classes, balanced)"], widths4, True)
pdf.table_row(["Visualization", "matplotlib, PIL, custom JS"], widths4)

pdf.section_title("Dependencies")
pdf.code_block(
    "torch>=2.0         torchvision>=0.15    flask>=3.0\n"
    "numpy>=1.24        Pillow>=10.0         matplotlib>=3.7\n"
    "fpdf2>=2.7         python-pptx>=0.6     scipy>=1.10",
    "Key Python packages"
)

# ── Save ──
pdf.output("Bypassing_Detection_Report.pdf")
print("Saved: Bypassing_Detection_Report.pdf")
