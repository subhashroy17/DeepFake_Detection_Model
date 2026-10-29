"""
MobileNetV2 Model Architecture for Deepfake Detection.

Implements transfer learning with a frozen MobileNetV2 backbone,
GlobalAveragePooling2D, Dropout, Dense classification head with Sigmoid activation,
and optional fine-tuning for upper convolutional layers.
"""

from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2

import config
from src.utils import get_logger

logger = get_logger("model")


def build_model(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    dropout_rate: float = config.DROPOUT_RATE,
    dense_units: int = config.DENSE_UNITS,
    learning_rate: float = config.INITIAL_LR,
    weights: Optional[str] = config.PRETRAINED_WEIGHTS,
) -> tf.keras.Model:
    """
    Construct the binary classification model using MobileNetV2 as a feature extractor.

    Architecture:
        1. Input Layer: (224, 224, 3)
        2. Pretrained MobileNetV2 backbone (ImageNet weights, include_top=False, frozen)
        3. GlobalAveragePooling2D
        4. Optional BatchNormalization & Dense intermediate projection
        5. Dropout layer for regularization
        6. Dense(1, activation='sigmoid') for binary prediction (0: REAL, 1: FAKE)
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Load MobileNetV2 backbone
    logger.info(f"Loading MobileNetV2 backbone with weights={weights}...")
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False  # Freeze pretrained weights initially

    # Feature extraction
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Optional intermediate projection with dropout
    if dense_units and dense_units > 0:
        x = layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(config.L2_REGULARIZATION),
            name="dense_projection",
        )(x)
        x = layers.BatchNormalization(name="batch_norm")(x)

    x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # Output layer: single unit with sigmoid activation
    # Output represents P(y = 1 | x), where 1 = FAKE, 0 = REAL
    outputs = layers.Dense(1, activation="sigmoid", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Deepfake_Detector")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=config.CLASSIFICATION_THRESHOLD),
            tf.keras.metrics.Precision(name="precision", thresholds=config.CLASSIFICATION_THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=config.CLASSIFICATION_THRESHOLD),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    logger.info("Successfully constructed and compiled initial MobileNetV2 model.")
    return model


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    fine_tune_layers: int = config.FINE_TUNE_LAYERS,
    fine_tune_lr: float = config.FINE_TUNE_LR,
) -> tf.keras.Model:
    """
    Unfreeze the top N layers of the MobileNetV2 backbone and recompile with lower learning rate.
    """
    logger.info(f"Preparing model for fine-tuning. Unfreezing top {fine_tune_layers} layers...")

    # Locate the MobileNetV2 layer inside the model
    base_model = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("MobileNetV2 backbone layer not found in the model.")

    base_model.trainable = True

    # Freeze all layers before the last `fine_tune_layers`
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - fine_tune_layers)

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    logger.info(
        f"Fine-tuning enabled: {total_layers - freeze_until} backbone layers unfrozen. "
        f"Total trainable params: {trainable_count:,}"
    )

    # Recompile with smaller learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=config.CLASSIFICATION_THRESHOLD),
            tf.keras.metrics.Precision(name="precision", thresholds=config.CLASSIFICATION_THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=config.CLASSIFICATION_THRESHOLD),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def load_trained_model(model_path: Optional[Path] = None) -> tf.keras.Model:
    """
    Safely load a trained .keras or .h5 model.
    """
    target_path = model_path or config.BEST_MODEL_PATH
    if not target_path.exists():
        if config.BACKUP_MODEL_PATH.exists():
            target_path = config.BACKUP_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"Trained model not found at '{target_path}'. Please train a model first using 'python train.py'."
            )

    logger.info(f"Loading trained model from {target_path}...")
    try:
        model = tf.keras.models.load_model(target_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {target_path}: {e}")
        raise
