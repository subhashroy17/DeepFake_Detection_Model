"""
Inference & Image Prediction Script for Deepfake Detection.

Predicts whether a single facial image is REAL or FAKE using the trained
MobileNetV2 model. Performs OpenCV face detection and cropping, applies exact
MobileNetV2 normalization [-1, 1], and reports prediction details.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.model import load_trained_model
from src.preprocessing import preprocess_for_inference
from src.utils import get_logger

logger = get_logger("predict")


def predict_single_image(
    image_path: Path,
    model_path: Optional[Path] = None,
    crop_face: bool = True,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run prediction on a single image file.

    Returns:
        Dictionary with classification result, raw score, face detection status,
        and advisory explanation.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load model
    model = load_trained_model(model_path)

    # Preprocess image
    tensor, display_crop, face_detected = preprocess_for_inference(
        image_path, crop_face=crop_face
    )

    # Model inference
    raw_score = float(model.predict(tensor, verbose=0)[0][0])

    # 0 = REAL, 1 = FAKE
    is_fake = raw_score >= threshold
    predicted_label = "FAKE" if is_fake else "REAL"
    confidence = raw_score if is_fake else (1.0 - raw_score)

    explanation = (
        "Face successfully detected and cropped before classification."
        if face_detected
        else "No distinct face detected; the entire image was resized and evaluated."
    )

    disclaimer = (
        "NOTE: The displayed score is the raw neural network sigmoid activation. "
        "It is not a calibrated Bayesian probability that the image is authentic or manipulated."
    )

    result = {
        "image_path": str(image_path),
        "predicted_label": predicted_label,
        "is_fake": is_fake,
        "raw_score": round(raw_score, 4),
        "confidence_percentage": round(confidence * 100, 2),
        "threshold": threshold,
        "face_detected": face_detected,
        "preprocessing_note": explanation,
        "disclaimer": disclaimer,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict whether an image is REAL or FAKE.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file (JPG/PNG).")
    parser.add_argument("--model-path", type=str, default=None, help="Path to custom model file (.keras).")
    parser.add_argument("--no-crop", action="store_true", help="Disable automatic face detection and cropping.")
    parser.add_argument("--threshold", type=float, default=config.CLASSIFICATION_THRESHOLD, help="Decision threshold.")
    args = parser.parse_args()

    image_path = Path(args.image)

    try:
        res = predict_single_image(
            image_path=image_path,
            model_path=Path(args.model_path) if args.model_path else None,
            crop_face=not args.no_crop,
            threshold=args.threshold,
        )

        print("\n" + "=" * 65)
        print("DEEPFAKE DETECTION PREDICTION RESULT")
        print("=" * 65)
        print(f"Target Image       : {res['image_path']}")
        print(f"Prediction         : {res['predicted_label']}")
        print(f"Confidence         : {res['confidence_percentage']}%")
        print(f"Raw Sigmoid Score  : {res['raw_score']:.4f} (Threshold: {res['threshold']:.2f})")
        print(f"Face Detection     : {'Detected & Cropped' if res['face_detected'] else 'Full Image Fallback'}")
        print(f"Preprocessing Note : {res['preprocessing_note']}")
        print("-" * 65)
        print(res["disclaimer"])
        print("=" * 65 + "\n")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
