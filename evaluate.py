"""
Evaluation Script for Deepfake Detection System.

Performs rigorous offline evaluation on:
1. Standard Test Set (disjoint source videos from known manipulation techniques)
2. Unseen Manipulation Test Set (disjoint source videos from held-out manipulation technique)

Calculates Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC,
Confusion Matrix, generates visualization artifacts, and persists JSON reports.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from src.metrics import compute_binary_metrics, compute_roc_curve_data
from src.model import load_trained_model
from src.preprocessing import preprocess_for_inference
from src.utils import get_logger, save_json
from src.visualization import plot_confusion_matrix, plot_roc_curve

logger = get_logger("evaluate")


def collect_split_images(split_dir: Path) -> Tuple[List[Path], List[int]]:
    """Scan a split directory and return all image paths with corresponding ground truth labels."""
    real_dir = split_dir / "real"
    fake_dir = split_dir / "fake"

    real_files = [
        p for p in real_dir.glob("*") if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS and p.is_file()
    ] if real_dir.exists() else []

    fake_files = [
        p for p in fake_dir.glob("*") if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS and p.is_file()
    ] if fake_dir.exists() else []

    all_paths = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    return all_paths, all_labels


def evaluate_split(
    model,
    split_dir: Path,
    split_name: str = "Standard Test",
    threshold: float = config.CLASSIFICATION_THRESHOLD,
    save_report_path: Optional[Path] = None,
    plot_prefix: str = "test",
) -> Optional[Dict[str, Any]]:
    """
    Run full model inference over a split directory, compute metrics, and save visual plots.
    """
    logger.info(f"Evaluating model on {split_name} split at: {split_dir} ...")

    image_paths, y_true = collect_split_images(split_dir)
    total_images = len(image_paths)

    if total_images == 0:
        logger.warning(f"No valid images found in {split_dir}. Evaluation skipped for {split_name}.")
        return None

    y_scores = []
    valid_y_true = []
    failed_images = 0

    for idx, img_path in enumerate(image_paths):
        try:
            # Consistent MobileNetV2 preprocessing
            tensor, _, _ = preprocess_for_inference(img_path, crop_face=True)
            prob = float(model.predict(tensor, verbose=0)[0][0])
            y_scores.append(prob)
            valid_y_true.append(y_true[idx])
        except Exception as e:
            logger.warning(f"Failed to evaluate image {img_path}: {e}")
            failed_images += 1

    if not valid_y_true:
        logger.error(f"All images failed to process in {split_dir}.")
        return None

    # Compute comprehensive metrics
    metrics_summary = compute_binary_metrics(
        y_true=valid_y_true,
        y_scores=y_scores,
        threshold=threshold,
    )

    metrics_summary["split_name"] = split_name
    metrics_summary["split_path"] = str(split_dir)
    metrics_summary["failed_images"] = failed_images

    # Generate Confusion Matrix plot
    cm_plot_path = config.PLOTS_DIR / f"confusion_matrix_{plot_prefix}.png"
    plot_confusion_matrix(
        cm_matrix=metrics_summary["confusion_matrix"]["matrix"],
        save_path=cm_plot_path,
        title=f"{split_name} Confusion Matrix",
    )
    metrics_summary["confusion_matrix_plot"] = str(cm_plot_path)

    # Generate ROC Curve plot if multiple classes exist
    roc_data = compute_roc_curve_data(valid_y_true, y_scores)
    if roc_data is not None and isinstance(metrics_summary["roc_auc"], (float, int)):
        fpr, tpr, _ = roc_data
        roc_plot_path = config.PLOTS_DIR / f"roc_curve_{plot_prefix}.png"
        plot_roc_curve(
            fpr=fpr,
            tpr=tpr,
            roc_auc=metrics_summary["roc_auc"],
            save_path=roc_plot_path,
            title=f"{split_name} ROC Curve",
        )
        metrics_summary["roc_curve_plot"] = str(roc_plot_path)
    else:
        metrics_summary["roc_curve_plot"] = "N/A (single-class split)"

    # Persist report JSON
    if save_report_path:
        save_json(metrics_summary, save_report_path)
        logger.info(f"Saved {split_name} evaluation report to {save_report_path}")

    # Print Formatted Report to stdout
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {split_name.upper()}")
    print("=" * 70)
    print(f"Total Evaluated   : {metrics_summary['total_samples']:,} images (Failed: {failed_images})")
    print(f"Class Breakdown   : REAL={metrics_summary['class_distribution']['REAL']}, FAKE={metrics_summary['class_distribution']['FAKE']}")
    print(f"Decision Threshold: {threshold:.2f}")
    print("-" * 70)
    print(f"Accuracy          : {metrics_summary['accuracy'] * 100:.2f}%")
    print(f"Precision         : {metrics_summary['precision'] * 100:.2f}%")
    print(f"Recall (Sens.)    : {metrics_summary['recall'] * 100:.2f}%")
    print(f"Specificity       : {metrics_summary['specificity'] * 100:.2f}%")
    print(f"F1-Score          : {metrics_summary['f1_score']:.4f}")
    print(f"ROC-AUC           : {metrics_summary['roc_auc']}")
    print("-" * 70)
    cm = metrics_summary["confusion_matrix"]
    print("Confusion Matrix:")
    print(f"  True Negatives  (REAL correctly classified) : {cm['true_negatives_real']:,}")
    print(f"  False Positives (REAL misclassified as FAKE): {cm['false_positives_fake']:,}")
    print(f"  False Negatives (FAKE misclassified as REAL): {cm['false_negatives_real']:,}")
    print(f"  True Positives  (FAKE correctly classified) : {cm['true_positives_fake']:,}")
    print("=" * 70 + "\n")

    return metrics_summary


def run_full_evaluation(
    model_path: Optional[Path] = None,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run evaluation on both standard test and unseen test splits.
    """
    config.init_directories()

    try:
        model = load_trained_model(model_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        print("\n" + "!" * 70)
        print("MODEL NOT FOUND")
        print("!" * 70)
        print("Please train a model first before evaluating:")
        print("  python train.py")
        print("!" * 70 + "\n")
        return {"status": "error", "message": "Model not found"}

    # 1. Standard Test Split Evaluation
    standard_results = evaluate_split(
        model=model,
        split_dir=config.TEST_DIR,
        split_name="Standard Test Set",
        threshold=threshold,
        save_report_path=config.EVALUATION_REPORT_PATH,
        plot_prefix="standard_test",
    )

    # 2. Unseen Test Split Evaluation
    unseen_results = evaluate_split(
        model=model,
        split_dir=config.UNSEEN_TEST_DIR,
        split_name="Unseen Manipulation Test Set",
        threshold=threshold,
        save_report_path=config.UNSEEN_REPORT_PATH,
        plot_prefix="unseen_test",
    )

    if unseen_results is None:
        logger.info(
            "Unseen manipulation test split is currently empty. "
            "Marked unseen evaluation as PENDING. (Documented in results/unseen_evaluation_report.json)"
        )
        save_json(
            {
                "status": "pending",
                "message": (
                    "Unseen manipulation evaluation is pending. The current dataset does not include "
                    "per-manipulation technique labels (e.g. FaceForensics++ Deepfakes/Face2Face/FaceSwap/NeuralTextures). "
                    "To enable, follow instructions in DATASET_GUIDE.md."
                ),
            },
            config.UNSEEN_REPORT_PATH,
        )

    return {
        "status": "success",
        "standard_test": standard_results,
        "unseen_test": unseen_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 Deepfake Detector.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained .keras model.")
    parser.add_argument("--threshold", type=float, default=config.CLASSIFICATION_THRESHOLD, help="Classification threshold.")
    args = parser.parse_args()

    run_full_evaluation(
        model_path=Path(args.model_path) if args.model_path else None,
        threshold=args.threshold,
    )
