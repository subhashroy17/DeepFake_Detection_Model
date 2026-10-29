"""
Data Pipeline and tf.data Loader for Deepfake Detection.

Constructs efficient tf.data.Dataset pipelines with augmentation (training only),
MobileNetV2 normalization, shuffling, batching, and prefetching.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import config
from src.utils import get_logger

logger = get_logger("data_loader")


def get_augmentation_layer() -> tf.keras.Sequential:
    """
    Construct data augmentation sequential model.
    Applied ONLY to the training set to prevent synthetic bias during evaluation.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(config.AUGMENTATION_CONFIG.get("rotation_range", 0.1)),
            layers.RandomZoom(config.AUGMENTATION_CONFIG.get("zoom_range", 0.1)),
            layers.RandomContrast(config.AUGMENTATION_CONFIG.get("contrast_factor", 0.1)),
        ],
        name="data_augmentation",
    )


def parse_and_preprocess_image(file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load raw image file, decode as JPEG/PNG, resize, and normalize to [-1, 1].
    """
    img_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_image(img_bytes, channels=config.IMG_CHANNELS, expand_animations=False)
    img = tf.image.resize(img, (config.IMG_HEIGHT, config.IMG_WIDTH), method="bicubic")
    # MobileNetV2 normalization: scale [0, 255] to [-1.0, 1.0]
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    img.set_shape(config.INPUT_SHAPE)
    return img, label


def create_dataset_from_paths(
    file_paths: List[str],
    labels: List[int],
    batch_size: int = config.BATCH_SIZE,
    is_training: bool = False,
    shuffle_buffer: int = 1000,
) -> tf.data.Dataset:
    """
    Create an optimized tf.data.Dataset from lists of image file paths and labels.
    """
    if len(file_paths) == 0:
        raise ValueError("Cannot create tf.data.Dataset from empty file list.")

    paths_tensor = tf.constant(file_paths, dtype=tf.string)
    labels_tensor = tf.constant(labels, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), shuffle_buffer), seed=config.RANDOM_SEED)

    # Parallel parsing
    dataset = dataset.map(parse_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Augmentation strictly for training
    if is_training:
        aug_layer = get_augmentation_layer()
        dataset = dataset.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_dataset_from_directory(
    directory: Union[str, Path],
    batch_size: int = config.BATCH_SIZE,
    is_training: bool = False,
) -> Tuple[tf.data.Dataset, int, Dict[str, int]]:
    """
    Scan a split directory structured as directory/real and directory/fake,
    validate images, and build a tf.data.Dataset.

    Returns:
        Tuple of (dataset, total_samples, class_counts)
    """
    target_dir = Path(directory)
    if not target_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {target_dir}")

    real_dir = target_dir / "real"
    fake_dir = target_dir / "fake"

    real_files = [
        str(p) for p in real_dir.glob("*") if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS and p.is_file()
    ]
    fake_files = [
        str(p) for p in fake_dir.glob("*") if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS and p.is_file()
    ]

    class_counts = {"REAL": len(real_files), "FAKE": len(fake_files)}
    total_samples = len(real_files) + len(fake_files)

    if total_samples == 0:
        raise ValueError(f"No valid images found in directory: {target_dir}")

    all_paths = real_files + fake_files
    all_labels = [config.LABEL_MAP["REAL"]] * len(real_files) + [config.LABEL_MAP["FAKE"]] * len(fake_files)

    dataset = create_dataset_from_paths(
        file_paths=all_paths,
        labels=all_labels,
        batch_size=batch_size,
        is_training=is_training,
    )

    logger.info(
        f"Created dataset from {target_dir.name}: Total={total_samples} "
        f"(REAL={class_counts['REAL']}, FAKE={class_counts['FAKE']})"
    )
    return dataset, total_samples, class_counts


def create_dataset_from_dataframe(
    df: pd.DataFrame,
    batch_size: int = config.BATCH_SIZE,
    is_training: bool = False,
) -> Tuple[tf.data.Dataset, int, Dict[str, int]]:
    """
    Build a tf.data.Dataset from a manifest DataFrame containing 'image_path' and 'label' columns.
    """
    if "image_path" not in df.columns or "label" not in df.columns:
        raise KeyError("DataFrame must contain 'image_path' and 'label' columns.")

    # Filter only existing files
    valid_mask = df["image_path"].apply(lambda p: Path(p).exists())
    valid_df = df[valid_mask].copy()

    total_samples = len(valid_df)
    if total_samples == 0:
        raise ValueError("No valid image files exist among the DataFrame entries.")

    real_count = int((valid_df["label"] == 0).sum())
    fake_count = int((valid_df["label"] == 1).sum())
    class_counts = {"REAL": real_count, "FAKE": fake_count}

    dataset = create_dataset_from_paths(
        file_paths=valid_df["image_path"].tolist(),
        labels=valid_df["label"].astype(int).tolist(),
        batch_size=batch_size,
        is_training=is_training,
    )
    return dataset, total_samples, class_counts
