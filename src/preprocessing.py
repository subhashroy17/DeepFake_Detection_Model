"""
Image Preprocessing and Face Detection Module.

Handles robust image loading, validation, face detection & cropping via OpenCV,
resizing, and consistent MobileNetV2 normalization [-1.0, 1.0].
Supports both legacy OpenCV Haar cascades and modern OpenCV 5+ environments.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

import config
from src.utils import get_logger

logger = get_logger("preprocessing")


class FaceDetector:
    """OpenCV Face Detector supporting Haar cascades and graceful full-frame fallback."""

    def __init__(self, padding_ratio: float = config.FACE_PADDING_RATIO):
        self.padding_ratio = padding_ratio
        self.cascade = None
        self.detector_type = "fallback"

        # Check for legacy OpenCV CascadeClassifier
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            try:
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self.cascade = cascade
                    self.detector_type = "cascade"
                    logger.info("Loaded OpenCV Haar Cascade face detector.")
            except Exception as e:
                logger.warning(f"Failed to initialize CascadeClassifier: {e}")

        if self.cascade is None:
            logger.info("FaceDetector initialized with robust direct crop & resize fallback.")

    def detect_and_crop(
        self, image_rgb: np.ndarray
    ) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detect the most prominent face in an RGB image and crop with padding.

        Args:
            image_rgb: RGB image as a uint8 NumPy array of shape (H, W, 3).

        Returns:
            Tuple of:
                - cropped_image: RGB uint8 array (either cropped face or original image)
                - face_found: bool indicating whether a face was detected
                - bbox: (x, y, w, h) of detection or None
        """
        if self.cascade is None or self.cascade.empty():
            # Graceful fallback: return original image for direct resizing
            return image_rgb, False, None

        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        try:
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            if len(faces) == 0:
                return image_rgb, False, None

            # Pick largest detected face by area
            best_face = max(faces, key=lambda rect: rect[2] * rect[3])
            fx, fy, fw, fh = best_face

            # Add padding
            pad_x = int(fw * self.padding_ratio)
            pad_y = int(fh * self.padding_ratio)

            x1 = max(0, fx - pad_x)
            y1 = max(0, fy - pad_y)
            x2 = min(w, fx + fw + pad_x)
            y2 = min(h, fy + fh + pad_y)

            cropped = image_rgb[y1:y2, x1:x2]
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                return image_rgb, False, None

            return cropped, True, (x1, y1, x2 - x1, y2 - y1)

        except Exception as e:
            logger.warning(f"Detection error: {e}. Falling back to full image.")
            return image_rgb, False, None


# Singleton detector instance
_detector = None


def get_face_detector() -> FaceDetector:
    """Retrieve or initialize the global FaceDetector."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def load_and_validate_image(
    image_input: Union[str, Path, bytes, Image.Image, np.ndarray]
) -> Image.Image:
    """
    Safely load an image and convert it to standard RGB PIL Image.

    Handles file paths, raw bytes, corrupted images, and different color modes.
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Image file is empty (0 bytes): {path}")
        try:
            img = Image.open(path)
            img.verify()
            img = Image.open(path)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid image file: {path}. Details: {e}") from e
    elif isinstance(image_input, bytes):
        import io
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def preprocess_for_inference(
    image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
    crop_face: bool = True,
    target_size: Tuple[int, int] = (config.IMG_WIDTH, config.IMG_HEIGHT),
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Complete inference preprocessing pipeline.

    Args:
        image_input: Path, bytes, PIL Image, or array.
        crop_face: Whether to attempt face detection & cropping.
        target_size: (width, height) tuple for MobileNetV2 input.

    Returns:
        Tuple of:
            - processed_batch: np.ndarray of shape (1, H, W, 3) normalized to [-1, 1]
            - display_crop: uint8 RGB np.ndarray of the cropped/resized face for UI display
            - face_detected: bool flag
    """
    pil_img = load_and_validate_image(image_input)
    img_array = np.array(pil_img, dtype=np.uint8)

    face_detected = False
    if crop_face:
        detector = get_face_detector()
        cropped, face_detected, _ = detector.detect_and_crop(img_array)
    else:
        cropped = img_array

    resized_pil = Image.fromarray(cropped).resize(target_size, Image.Resampling.BICUBIC)
    resized_np = np.array(resized_pil, dtype=np.uint8)

    normalized = (resized_np.astype(np.float32) / 127.5) - 1.0
    batch_tensor = np.expand_dims(normalized, axis=0)

    return batch_tensor, resized_np, face_detected


def normalize_mobilenet_v2(image_tensor: np.ndarray) -> np.ndarray:
    """
    Mathematical normalization identical to tf.keras.applications.mobilenet_v2.preprocess_input.
    Maps uint8 [0, 255] or float [0.0, 255.0] to float32 [-1.0, 1.0].
    """
    return (image_tensor.astype(np.float32) / 127.5) - 1.0
