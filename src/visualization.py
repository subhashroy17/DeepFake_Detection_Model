"""
Visualization Module for Training Curves, Confusion Matrices, and ROC Curves.

Generates presentation-ready Matplotlib plots saved as PNG artifacts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
# Use non-interactive backend for headless execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.utils import get_logger

logger = get_logger("visualization")


def plot_training_history(
    history_dict: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    fine_tune_epoch: Optional[int] = None,
) -> Path:
    """
    Plot training & validation accuracy and loss over epochs.
    Optionally marks the transition where fine-tuning started.
    """
    target_path = save_path or (config.PLOTS_DIR / "training_curves.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history_dict.get("accuracy", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MobileNetV2 Deepfake Detector Training History", fontsize=14, fontweight="bold")

    # Accuracy subplot
    if "accuracy" in history_dict:
        ax1.plot(epochs, history_dict["accuracy"], label="Training Accuracy", color="#1f77b4", linewidth=2)
    if "val_accuracy" in history_dict:
        ax1.plot(epochs, history_dict["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2)

    ax1.set_title("Model Accuracy over Epochs", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    if fine_tune_epoch and fine_tune_epoch < len(epochs):
        ax1.axvline(x=fine_tune_epoch, color="red", linestyle=":", label="Fine-Tuning Start")

    # Loss subplot
    if "loss" in history_dict:
        ax2.plot(epochs, history_dict["loss"], label="Training Loss", color="#2ca02c", linewidth=2)
    if "val_loss" in history_dict:
        ax2.plot(epochs, history_dict["val_loss"], label="Validation Loss", color="#d62728", linewidth=2)

    ax2.set_title("Model Loss (Binary Cross-Entropy)", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    if fine_tune_epoch and fine_tune_epoch < len(epochs):
        ax2.axvline(x=fine_tune_epoch, color="red", linestyle=":", label="Fine-Tuning Start")

    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()

    logger.info(f"Saved training curves to {target_path}")
    return target_path


def plot_confusion_matrix(
    cm_matrix: Union[List[List[int]], np.ndarray],
    save_path: Optional[Path] = None,
    title: str = "Test Set Confusion Matrix",
) -> Path:
    """
    Generate and save a 2x2 confusion matrix heatmap with counts and percentages.
    """
    target_path = save_path or (config.PLOTS_DIR / "confusion_matrix.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array(cm_matrix)
    total = np.sum(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = config.CLASS_NAMES
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / total * 100.0) if total > 0 else 0.0
            ax.text(
                j,
                i,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if count > thresh else "black",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()

    logger.info(f"Saved confusion matrix plot to {target_path}")
    return target_path


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: Optional[Path] = None,
    title: str = "Receiver Operating Characteristic (ROC)",
) -> Path:
    """
    Plot and save ROC curve with AUC score annotation and random baseline.
    """
    target_path = save_path or (config.PLOTS_DIR / "roc_curve.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"MobileNetV2 (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()

    logger.info(f"Saved ROC curve plot to {target_path}")
    return target_path
