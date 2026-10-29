"""
Configuration Module for Deepfake Detection System.

Centralized settings for dataset paths, preprocessing parameters,
MobileNetV2 model architecture, training hyperparameters, evaluation metrics,
and runtime execution options.
"""

import os
from pathlib import Path

# Thread and memory stability for OpenBLAS on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONUTF8", "1")

# ==============================================================================
# 1. Base Paths & Directory Structure
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
UNSEEN_TEST_DIR = DATA_DIR / "unseen_test"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"

# Artifact file paths
BEST_MODEL_PATH = MODELS_DIR / "mobilenetv2_deepfake_detector.keras"
BACKUP_MODEL_PATH = MODELS_DIR / "mobilenetv2_deepfake_detector.h5"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
TRAINING_SUMMARY_PATH = RESULTS_DIR / "training_summary.json"
EVALUATION_REPORT_PATH = RESULTS_DIR / "evaluation_report.json"
UNSEEN_REPORT_PATH = RESULTS_DIR / "unseen_evaluation_report.json"
DATASET_MANIFEST_PATH = MANIFESTS_DIR / "dataset_manifest.csv"
DATASET_STATS_PATH = MANIFESTS_DIR / "dataset_stats.json"

# ==============================================================================
# 2. Dataset & KaggleHub Settings
# ==============================================================================
KAGGLE_DATASET_ID = "sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset"

# Video to frame extraction parameters
FRAMES_PER_VIDEO = 10           # Configurable cap per video for laptop friendliness
FRAME_INTERVAL = 15             # Extract 1 frame every N frames to avoid duplicate expressions
FACE_DETECTION_CONFIDENCE = 0.5 # Confidence threshold for OpenCV Haar/DNN face detection
FACE_PADDING_RATIO = 0.15       # Margin to expand bounding box around face crops

# Supported image formats
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Train / Val / Test split ratios (based on source videos/identities)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==============================================================================
# 3. Image Preprocessing & Model Input Specifications
# ==============================================================================
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# MobileNetV2 uses [-1, 1] normalization: (x / 127.5) - 1.0
PREPROCESSING_MODE = "tf"  # matches tf.keras.applications.mobilenet_v2.preprocess_input

# Class label mapping: Binary classification (0: REAL, 1: FAKE)
CLASS_NAMES = ["REAL", "FAKE"]
LABEL_MAP = {"REAL": 0, "FAKE": 1}
INV_LABEL_MAP = {0: "REAL", 1: "FAKE"}

# Decision threshold for binary classification
CLASSIFICATION_THRESHOLD = 0.50

# ==============================================================================
# 4. Model Architecture & Transfer Learning
# ==============================================================================
BACKBONE_NAME = "MobileNetV2"
PRETRAINED_WEIGHTS = "imagenet"
DENSE_UNITS = 128
DROPOUT_RATE = 0.35
L2_REGULARIZATION = 1e-4

# Fine-tuning: number of top layers of MobileNetV2 to unfreeze
FINE_TUNE_LAYERS = 30

# ==============================================================================
# 5. Training Hyperparameters
# ==============================================================================
RANDOM_SEED = 42

BATCH_SIZE = 16                 # Laptop friendly batch size
INITIAL_EPOCHS = 10             # Phase 1: Train classification head with frozen backbone
FINE_TUNE_EPOCHS = 5            # Phase 2: Fine-tune top backbone layers
INITIAL_LR = 1e-4               # Adam learning rate for Phase 1
FINE_TUNE_LR = 1e-5             # Lower learning rate for Phase 2 fine-tuning

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Data augmentation flags (applied ONLY to training set)
AUGMENTATION_CONFIG = {
    "rotation_range": 0.1,       # +/- 10%
    "zoom_range": 0.1,           # +/- 10%
    "horizontal_flip": True,
    "contrast_factor": 0.1,
}

# ==============================================================================
# 6. Helper to Ensure All Directories Exist
# ==============================================================================
def init_directories():
    """Ensure all required project directories exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MANIFESTS_DIR,
        TRAIN_DIR / "real",
        TRAIN_DIR / "fake",
        VAL_DIR / "real",
        VAL_DIR / "fake",
        TEST_DIR / "real",
        TEST_DIR / "fake",
        UNSEEN_TEST_DIR / "real",
        UNSEEN_TEST_DIR / "fake",
        MODELS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
