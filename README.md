# Deep Learning-Based Deepfake Detection Under Unseen Manipulation Techniques

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning framework designed to detect facial manipulation and evaluate model generalization against **unseen manipulation techniques**. Built with **TensorFlow/Keras**, **MobileNetV2** transfer learning, **OpenCV** facial detection, and **Streamlit**.

---

## 🌟 Key Features

1. **Transfer Learning with MobileNetV2**:
   - Pretrained on ImageNet with frozen feature extraction.
   - GlobalAveragePooling2D, Batch Normalization, Dropout regularization, and Sigmoid classification head.
   - Two-phase training: Head training followed by optional fine-tuning of upper convolutional layers.
2. **Leakage-Free Splitting by Video Source**:
   - Strictly partitions data by source video ID and subject identity, preventing frame-level leakage.
3. **Dual Evaluation Framework**:
   - **Standard Test Evaluation**: Evaluates generalization to unseen video subjects using *known* manipulation techniques.
   - **Unseen Manipulation Evaluation**: Evaluates robustness against a *held-out manipulation family* (e.g. NeuralTextures or FaceSwap).
4. **Automated Facial Preprocessing**:
   - OpenCV face detection with margin padding and fallback full-frame resizing.
   - Exact mathematical normalization `(x / 127.5) - 1.0` consistent across training and inference.
5. **Interactive Streamlit Web Application**:
   - Real-time image upload, face cropping visualization, raw sigmoid score, and confidence percentage.
   - Interactive model evaluation dashboards with confusion matrices and ROC curves.
6. **Laptop & CPU Friendly**:
   - Configurable batch size (16) and frame sampling intervals requiring no GPU.
7. **Storage-Aware Dataset Downloader**:
   - Integrates KaggleHub with automated disk space detection (redirects cache to high-capacity drives like `D:\` if `C:\` has low free space).

---

## 🏗️ Project Architecture

```
Deep_Learning_project/
├── app.py                      # Interactive Streamlit Web Application
├── train.py                    # Two-Phase Model Training Script
├── evaluate.py                 # Standard & Unseen Test Evaluation Script
├── predict.py                  # CLI Single-Image Prediction Tool
├── config.py                   # Centralized Configuration & Hyperparameters
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Overview & Quickstart
├── DOCUMENTATION.md            # Comprehensive Theory, Viva Q&A, and Reference Guide
├── PROJECT_REPORT.md           # Academic Project Report
├── DATASET_GUIDE.md            # Dataset Acquisition & Kaggle/FaceForensics Guide
│
├── src/                        # Core Python Modules
│   ├── data_loader.py          # tf.data Pipeline with Training Augmentation
│   ├── preprocessing.py        # Face Detection, Bounding Box Padding & Normalization
│   ├── model.py                # MobileNetV2 Architecture & Fine-Tuning Setup
│   ├── metrics.py              # Precision, Recall, Specificity, F1, ROC-AUC & Confusion Matrix
│   ├── visualization.py        # Training Curves, Heatmaps & ROC Plots
│   └── utils.py                # Logging, Seed Setting, JSON Serialization
│
├── scripts/                    # Automation Scripts
│   ├── download_dataset.py     # Programmatic Download via KaggleHub
│   ├── download_instructions.py# Manual Dataset Instructions
│   ├── extract_frames.py       # Configurable Video Frame & Facial Cropper
│   ├── prepare_dataset.py      # Source-Level Split & Manifest Generator
│   ├── validate_dataset.py     # Data Leakage & Integrity Auditor
│   └── create_demo_data.py     # Synthetic Placeholder Generator for Pipeline Testing
│
├── tests/                      # Automated Pytest Suite
│   ├── test_preprocessing.py   # Image Validation, Corrupt Files & Normalization
│   ├── test_model.py           # Architecture Shapes, Weights & Range
│   ├── test_dataset_validation.py # Source Leakage & Metric Computations
│   └── test_prediction.py      # Inference & Error Handling
│
├── data/                       # Dataset Storage
│   ├── raw/                    # Raw Downloaded Archives / Videos
│   ├── processed/              # Cropped Facial Frames
│   ├── manifests/              # Manifest CSV & Dataset Inspection JSON
│   ├── train/ (real/fake)      # Training Set (70%)
│   ├── val/ (real/fake)        # Validation Set (15%)
│   ├── test/ (real/fake)       # Standard Test Set (15%)
│   └── unseen_test/ (real/fake)# Held-Out Manipulation Test Set
│
├── models/                     # Saved Trained Models (.keras)
├── results/                    # Metric Summaries & Plot Artifacts
└── logs/                       # Execution Logs
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Dataset Acquisition

#### Option A: KaggleHub Programmatic Download
```bash
python scripts/download_dataset.py
```
*Note: The script automatically detects if C: has <25GB free and redirects download cache to D: drive.*

#### Option B: Immediate Pipeline & UI Testing with Synthetic Demo Data
To test the pipeline immediately without waiting for a large download:
```bash
python scripts/create_demo_data.py
python scripts/validate_dataset.py
```

---

### 3. Model Training

Train MobileNetV2 (Phase 1 head training + Phase 2 fine-tuning):

```bash
# Standard training
python train.py --epochs 5 --fine-tune-epochs 2

# Lightweight / Fast Run on Laptop
python train.py --epochs 2 --fine-tune-epochs 1 --batch-size 16
```

Generated artifacts:
- Best model: `models/mobilenetv2_deepfake_detector.keras`
- Training curves: `results/plots/training_curves.png`
- History JSON: `results/training_history.json`
- Summary JSON: `results/training_summary.json`

---

### 4. Model Evaluation

Evaluate the saved model on the test splits:

```bash
python evaluate.py
```

Outputs:
- Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC.
- Confusion matrix heatmap: `results/plots/confusion_matrix_standard_test.png`.
- ROC curve plot: `results/plots/roc_curve_standard_test.png`.
- Detailed report: `results/evaluation_report.json`.

---

### 5. Single-Image Prediction (CLI)

Classify an individual image file:

```bash
python predict.py --image path/to/sample_face.jpg
```

Options:
- `--no-crop`: Disables OpenCV face detection and uses the full image.
- `--threshold 0.50`: Adjusts the classification decision boundary.

---

### 6. Launch Streamlit Web Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to interact with:
- **Image Prediction**: Drag-and-drop face images with real-time classification badges and probability bars.
- **Model Evaluation**: Interactive charts, confusion matrices, and ROC curves.
- **About & Research**: Academic explanations of deepfakes, MobileNetV2 architecture, and viva prep.

---

### 7. Run Automated Tests

Execute the complete pytest test suite:

```bash
python -m pytest tests/ -v
```

---

## 📊 Classification Threshold & Score Interpretation

The model output $p \in [0, 1]$ represents the raw sigmoid activation:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

- If $p \ge 0.50 \implies \textbf{FAKE}$ (Manipulated)
- If $p < 0.50 \implies \textbf{REAL}$ (Authentic)

> [!WARNING]
> The displayed score is a neural network activation indicating similarity to training manipulation patterns. It is **not** a calibrated statistical probability and should not be treated as absolute legal proof.

---

## 📜 Academic References

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381).
2. Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images*. ICCV. [arXiv:1901.08971](https://arxiv.org/abs/1901.08971).
3. Dolhansky, B., et al. (2020). *The Deepfake Detection Challenge (DFDC) Dataset*. [arXiv:2006.07397](https://arxiv.org/abs/2006.07397).
