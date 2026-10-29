"""
Unit Tests for Prediction Script and Error Handling.
"""

from pathlib import Path

import pytest
from PIL import Image

from predict import predict_single_image
from src.model import load_trained_model


def test_missing_model_raises_clean_error():
    """Verify that calling load_trained_model when no model exists raises FileNotFoundError."""
    non_existent = Path("models/definitely_not_here_12345.keras")
    with pytest.raises(FileNotFoundError):
        load_trained_model(non_existent)


def test_predict_missing_image(tmp_path: Path):
    """Verify predict_single_image raises FileNotFoundError on missing image path."""
    with pytest.raises(FileNotFoundError):
        predict_single_image(tmp_path / "does_not_exist.jpg")


def test_predict_with_saved_model(tmp_path: Path):
    """Verify inference pipeline executes end-to-end with a dummy trained model."""
    from src.model import build_model

    # Build and save a temporary dummy model
    dummy_model_path = tmp_path / "test_model.keras"
    model = build_model(weights=None)
    model.save(dummy_model_path)

    # Create a test image
    test_img = Image.new("RGB", (250, 250), color=(180, 160, 140))
    test_img_path = tmp_path / "test_face.jpg"
    test_img.save(test_img_path)

    # Run predict
    result = predict_single_image(
        image_path=test_img_path,
        model_path=dummy_model_path,
        crop_face=False,
    )

    assert "predicted_label" in result
    assert result["predicted_label"] in ["REAL", "FAKE"]
    assert 0.0 <= result["raw_score"] <= 1.0
    assert 0.0 <= result["confidence_percentage"] <= 100.0
    assert "face_detected" in result
