"""
Unit Tests for MobileNetV2 Architecture & Compilation.
"""

import numpy as np
import pytest
import tensorflow as tf

import config
from src.model import build_model, unfreeze_for_fine_tuning


@pytest.fixture(scope="module")
def initialized_model():
    """Build a MobileNetV2 model once for test suite (weights=None for fast unit testing)."""
    # Use weights=None so test does not depend on internet downloading ImageNet weights
    model = build_model(weights=None)
    return model


def test_model_output_shape(initialized_model):
    """Verify model output shape is (None, 1)."""
    assert initialized_model.output_shape == (None, 1)


def test_model_output_range(initialized_model):
    """Verify random input returns output strictly in [0.0, 1.0] due to Sigmoid activation."""
    dummy_input = np.random.uniform(-1.0, 1.0, size=(2, 224, 224, 3)).astype(np.float32)
    predictions = initialized_model.predict(dummy_input, verbose=0)

    assert predictions.shape == (2, 1)
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)


def test_frozen_backbone(initialized_model):
    """Verify backbone layers are initially frozen."""
    base_model = None
    for layer in initialized_model.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    assert base_model is not None
    assert base_model.trainable is False


def test_fine_tuning_unfreezes_layers(initialized_model):
    """Verify unfreeze_for_fine_tuning sets backbone to trainable and updates weights."""
    model = unfreeze_for_fine_tuning(initialized_model, fine_tune_layers=10)
    base_model = [l for l in model.layers if "mobilenetv2" in l.name.lower()][0]
    assert base_model.trainable is True

    # At least some layers in base_model must now be trainable
    trainable_sublayers = [l for l in base_model.layers if l.trainable]
    assert len(trainable_sublayers) > 0
