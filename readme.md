# PDD Traffic Sign Classifier

> A CNN-powered classifier that identifies 43 road sign types from camera
> images in real time — a core computer vision component for autonomous
> driving assistance and road safety monitoring systems.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120-teal)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)]()
[![Accuracy](https://img.shields.io/badge/Val%20Accuracy-99.95%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Business Problem

Autonomous vehicles and driver assistance systems must recognize road signs
reliably under varying lighting, weather, and viewing angles — a failure
rate in safety-critical sign detection (stop signs, speed limits, yield)
can directly cause accidents and regulatory non-compliance. A production-grade
sign classifier that handles 43 sign categories at 99.95% validation accuracy
provides the foundation for ADAS pipelines, dashcam analytics platforms, and
smart infrastructure monitoring systems — reducing dependence on manual road
audit teams.

---

## Demo

**REST API:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@stop_sign.jpg"
```

**Response:**
```json
{
  "class_id": 14,
  "sign": "Stop"
}
```

**Or use the Streamlit UI** for manual image upload and instant visual
classification — see [How to Run](#how-to-run).

**Supported categories:** 43 German road sign types — speed limits,
priority, prohibitory, mandatory, and warning signs (full GTSRB set).

---

## Results

| Metric              | Score  |
|----------------------|--------|
| Train Accuracy       | 99.72% |
| Validation Accuracy  | 99.95% |

Model: Custom 3-block CNN (Conv-Conv-Pool ×3, BatchNorm2d, Dropout2d,
Dropout), trained from scratch, no transfer learning.

Baseline (random classifier, 43 classes): Accuracy = 2.3%
↑ +97.65 pp improvement vs baseline

---

## Dataset

- **Source:** GTSRB — German Traffic Sign Recognition Benchmark (Kaggle:
  `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`)
- **Size:** full `Train/` directory loaded via `ImageFolder`, 43 sign classes
- **Split:** 80% train / 20% validation via `random_split` (seed=42),
  applied consistently across separate train- and val-transform datasets
- **Features:** real-world RGB photos of road signs at varying distances,
  angles, and lighting conditions; resized to 32×32 for both training
  and inference
- **Class balance:** imbalanced — speed limit signs are overrepresented
  vs rare regulatory signs; not explicitly rebalanced, handled implicitly
  via `shuffle=True` and per-class CrossEntropyLoss averaging

---

## Approach

1. **Data Loading** — downloaded via `kagglehub`, loaded twice with
   `torchvision.datasets.ImageFolder` (once with train-transform, once
   with test-transform) so the same `random_split` indices apply cleanly
   to both augmented and clean versions
2. **Augmentation (train only)** — `RandomRotation(15)`,
   `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)`,
   `RandomAffine(translate=(0.1, 0.1))` to simulate varied camera angles,
   lighting, and sign positioning in real driving conditions
3. **Normalization** — `Normalize([0.5]*3, [0.5]*3)`, identical for
   train and validation
4. **Model Architecture** — deeper 3-block CNN with double convolutions
   per block: `[Conv2d→BN→ReLU]×2 → MaxPool2d → Dropout2d(0.25)` for
   32→64→128 channels, followed by `Flatten → Linear(2048→512) → ReLU
   → Dropout(0.5) → Linear(512→43)`
5. **Training** — 30 epochs, Adam (lr=0.001), CrossEntropyLoss,
   `StepLR` scheduler (step_size=10, gamma=0.5) halving the learning
   rate every 10 epochs; loss reduced from ~1235 to ~34 over training
6. **Deployment** — two interfaces sharing the same model class and
   transform pipeline: a FastAPI `/predict` endpoint for programmatic
   integration, and a Streamlit UI for manual image upload and review

---

## Key Challenges & Solutions

**Applying train-time augmentation without leaking it into the validation set**
`ImageFolder` binds a single transform to the whole dataset, but
`random_split` operates on indices — naively splitting one transformed
dataset would either augment the validation images or leave training
images unaugmented → loaded the same directory twice into two separate
`ImageFolder` instances (train-transform and test-transform), then applied
`random_split` with an identical `torch.Generator().manual_seed(42)` to
both, keeping only the train half from one and the val half from the other
→ guarantees validation images are never touched by `RandomRotation`,
`ColorJitter`, or `RandomAffine`, while training images get full augmentation.

**Overfitting on real-world photos with high visual variance**
Real-world sign photos vary significantly in brightness, blur, angle, and
occlusion — a plain CNN risks overfitting to clean training samples →
combined `RandomRotation(15)`, `ColorJitter`, and `RandomAffine` augmentation
with `Dropout2d(0.25)` after every pooling block and `Dropout(0.5)` in the
classifier head → train/val accuracy gap stayed negative (99.72% train vs
99.95% val), indicating augmentation-driven regularization rather than
memorization.

**Learning rate decay over 30-epoch runs**
A fixed learning rate of 0.001 risks oscillation or overshoot as loss
approaches convergence in later epochs → added `StepLR(step_size=10,
gamma=0.5)`, halving the learning rate every 10 epochs → loss dropped
from ~120 (epoch 10) to ~34 (epoch 30) as the step-downs took effect,
supporting stable late-stage convergence.

---

## Tech Stack

| Category       | Tools                                   |
|-----------------|------------------------------------------|
| Language        | Python 3.11                              |
| ML              | PyTorch, torchvision                     |
| API             | FastAPI, Uvicorn                         |
| App             | Streamlit                                |
| Data            | KaggleHub, Pillow, Matplotlib            |
| Regularization  | BatchNorm2d, Dropout2d, Dropout, StepLR  |

---

## Project Structure
```
ml_PDD_TrafficSignClassifier/
├── .gitignore
├── readme.md
├── requirements.txt
└── PDD_TrafficSignClassifier/
    ├── PDD_TrafficSignClassifier.ipynb
    ├── documents/
    │   └── ПДД.docx
    ├── labels_PDD_TrafficSignClassifier.pth
    ├── main.py
    ├── model_PDD_TrafficSignClassifier.pth
    ├── sample_signs.png
    └── tests/
```

---

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/your-username/pdd-traffic-sign-classifier
cd pdd-traffic-sign-classifier/PDD_TrafficSignClassifier
pip install torch torchvision fastapi uvicorn pillow streamlit
```

```bash
# 2. Train the model (optional — pretrained weights included)
# open and run PDD_TrafficSignClassifier.ipynb (Colab-ready, GPU recommended)
```

```bash
# 3a. Launch the REST API
uvicorn main:app --host 127.0.0.1 --port 8000
```

```bash
# 3b. Or launch the Streamlit UI
streamlit run main.py
# Opens at http://localhost:8501
```

---

## Business Impact

- ↓ reduction in manual road sign audit costs for infrastructure
  monitoring teams vs field inspection workflows (estimated)
- ↑ 99.95% automated sign recognition accuracy across 43 categories,
  covering the full GTSRB regulatory sign set
- ↑ dual-interface design (REST API + Streamlit) supports both automated
  conveyor/dashcam pipelines and manual review workflows without
  maintaining two separate model implementations
- ↑ model weights portable to edge deployment on dashcam hardware
  (Raspberry Pi / Jetson Nano) for real-time ADAS integration
- ↑ retrainable on country-specific sign sets (US, JP, CN) with dataset
  swap only — no architecture changes required

---
