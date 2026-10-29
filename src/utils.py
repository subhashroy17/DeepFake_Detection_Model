"""
Utility Functions for Deepfake Detection System.

Provides logging setup, seed initialization for reproducibility,
JSON serialization, and safe file system helpers.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import config


def get_logger(name: str = "deepfake_detector", log_file: Optional[Path] = None) -> logging.Logger:
    """Configure and return a standardized logger with console and file handlers."""
    config.init_directories()
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        target_log_file = log_file or (config.LOGS_DIR / "system.log")
        file_handler = logging.FileHandler(target_log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set random seed across python, numpy, and tensorflow for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Safely serialize a dictionary to a JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def load_json(file_path: Path) -> Dict[str, Any]:
    """Safely deserialize a JSON file into a dictionary."""
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_bytes(size_bytes: int) -> str:
    """Convert raw byte count into human-readable string (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
