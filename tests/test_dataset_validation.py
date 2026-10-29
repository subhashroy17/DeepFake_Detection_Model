"""
Unit Tests for Dataset Splitting & Data Leakage Prevention.
"""

import pytest

from scripts.prepare_dataset import split_by_source_id
from src.metrics import compute_binary_metrics


def test_split_by_source_id_prevents_leakage():
    """Verify that multiple frames from the same source ID are never split across train/test."""
    # Simulate 10 videos (5 real, 5 fake), each with 4 frames
    mock_records = []
    for vid_idx in range(5):
        sid = f"real_vid_{vid_idx}"
        for f in range(4):
            mock_records.append(
                {
                    "filename": f"{sid}_f{f}.jpg",
                    "label": 0,
                    "class_name": "REAL",
                    "source_id": sid,
                    "manipulation_category": "authentic",
                }
            )

    for vid_idx in range(5):
        sid = f"fake_vid_{vid_idx}"
        for f in range(4):
            mock_records.append(
                {
                    "filename": f"{sid}_f{f}.jpg",
                    "label": 1,
                    "class_name": "FAKE",
                    "source_id": sid,
                    "manipulation_category": "deepfakes",
                }
            )

    allocated, stats = split_by_source_id(mock_records, seed=42)

    # Collect source IDs in each split
    train_sources = {r["source_id"] for r in allocated if r["split"] == "train"}
    val_sources = {r["source_id"] for r in allocated if r["split"] == "val"}
    test_sources = {r["source_id"] for r in allocated if r["split"] == "test"}

    # Assert sets are strictly disjoint
    assert len(train_sources.intersection(val_sources)) == 0
    assert len(train_sources.intersection(test_sources)) == 0
    assert len(val_sources.intersection(test_sources)) == 0


def test_metrics_computation_accuracy():
    """Verify compute_binary_metrics calculates exact accuracy, precision, recall, and F1."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.8, 0.9]  # Perfect predictions at threshold 0.5

    results = compute_binary_metrics(y_true, y_scores, threshold=0.5)

    assert results["accuracy"] == 1.0
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1_score"] == 1.0
    assert results["confusion_matrix"]["true_positives_fake"] == 2
    assert results["confusion_matrix"]["true_negatives_real"] == 2
    assert results["confusion_matrix"]["false_positives_fake"] == 0
    assert results["confusion_matrix"]["false_negatives_real"] == 0


def test_metrics_handles_empty_dataset():
    """Verify compute_binary_metrics handles empty inputs gracefully without crashing."""
    results = compute_binary_metrics([], [])
    assert "error" in results
    assert results["total_samples"] == 0
