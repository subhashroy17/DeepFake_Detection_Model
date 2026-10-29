"""
Unit Tests for Preprocessing and Face Detection Module.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import config
from src.preprocessing import (
    FaceDetector,
    get_face_detector,
    load_and_validate_image,
    normalize_mobilenet_v2,
    preprocess_for_inference,
)


@pytest.fixture
def sample_rgb_image(tmp_path: Path) -> Path:
    """Create a temporary valid RGB test image."""
    img = Image.new("RGB", (300, 300), color=(120, 150, 180))
    img_path = tmp_path / "sample_valid.jpg"
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """Create a 0-byte corrupted image file."""
    empty_path = tmp_path / "corrupt_empty.jpg"
    empty_path.write_bytes(b"")
    return empty_path


def test_load_and_validate_valid_image(sample_rgb_image: Path):
    """Verify loading of a valid image returns RGB PIL Image."""
    img = load_and_validate_image(sample_rgb_image)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (300, 300)


def test_load_and_validate_corrupt_image(empty_file: Path):
    """Verify loading an empty or corrupt file raises ValueError."""
    with pytest.raises(ValueError):
        load_and_validate_image(empty_file)


def test_load_and_validate_missing_file():
    """Verify loading a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_and_validate_image(Path("non_existent_image_12345.jpg"))


def test_normalization_range():
    """Verify MobileNetV2 normalization maps [0, 255] exactly to [-1.0, 1.0]."""
    test_arr = np.array([0, 127.5, 255], dtype=np.float32)
    norm = normalize_mobilenet_v2(test_arr)

    assert np.isclose(norm[0], -1.0)
    assert np.isclose(norm[1], 0.0)
    assert np.isclose(norm[2], 1.0)


def test_preprocess_for_inference_dimensions(sample_rgb_image: Path):
    """Verify output batch tensor has shape (1, 224, 224, 3) and valid range."""
    batch_tensor, display_crop, face_found = preprocess_for_inference(
        sample_rgb_image, crop_face=False
    )

    assert batch_tensor.shape == (1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    assert batch_tensor.min() >= -1.0
    assert batch_tensor.max() <= 1.0
    assert display_crop.shape == (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    assert isinstance(face_found, bool)


def test_face_detector_initialization():
    """Verify FaceDetector loads without throwing uncaught exceptions."""
    detector = get_face_detector()
    assert isinstance(detector, FaceDetector)
    # Test on uniform image (no face should be found)
    blank_arr = np.zeros((200, 200, 3), dtype=np.uint8)
    crop, found, bbox = detector.detect_and_crop(blank_arr)
    assert found is False
    assert bbox is None
    assert crop.shape == (200, 200, 3)
