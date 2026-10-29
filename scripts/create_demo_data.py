"""
Synthetic Placeholder Data Generator for Pipeline & Integration Testing.

Generates clearly marked synthetic placeholder images so developers and students
can test data pipelines, unit tests, model training, and the Streamlit interface
immediately without waiting for multi-gigabyte dataset downloads.

NOTE: This synthetic data is purely for functional verification and MUST NOT be
used to report scientific performance or genuine deepfake detection accuracy.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_logger

logger = get_logger("create_demo_data")


def draw_synthetic_face(
    is_fake: bool,
    source_id: str,
    frame_idx: int,
    manipulation_type: str = "none",
) -> Image.Image:
    """
    Generate an RGB image simulating a facial crop with distinctive patterns,
    labeled clearly with synthetic watermarks.
    """
    img = Image.new("RGB", (config.IMG_WIDTH, config.IMG_HEIGHT), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)

    # Background gradient or pattern
    rng = random.Random(hash(source_id) + frame_idx)

    # Base skin tone
    skin_color = (
        rng.randint(210, 255),
        rng.randint(180, 220),
        rng.randint(160, 200),
    )

    # Face contour oval
    draw.ellipse([40, 30, 184, 194], fill=skin_color, outline=(150, 120, 100), width=2)

    # Eyes
    eye_color = (40, 30, 20)
    draw.ellipse([70, 75, 95, 95], fill=eye_color)
    draw.ellipse([129, 75, 154, 95], fill=eye_color)

    # Nose
    draw.polygon([(112, 100), (105, 125), (119, 125)], fill=(180, 140, 120))

    # Mouth
    if is_fake:
        # Subtle synthetic artifact or distortion pattern
        draw.arc([75, 140, 149, 165], start=0, end=180, fill=(180, 50, 50), width=3)
        # Synthetic manipulation boundary / noise
        for _ in range(5):
            rx = rng.randint(50, 170)
            ry = rng.randint(60, 170)
            draw.point((rx, ry), fill=(255, 0, 0))
    else:
        draw.arc([75, 140, 149, 165], start=0, end=180, fill=(120, 40, 40), width=2)

    # Prominent Synthetic Banner Watermark
    banner_text = f"DEMO {'FAKE' if is_fake else 'REAL'}"
    draw.rectangle([0, 195, config.IMG_WIDTH, config.IMG_HEIGHT], fill=(30, 30, 30))
    draw.text((10, 200), f"SYNTHETIC: {banner_text}", fill=(255, 255, 255))
    draw.text((10, 210), f"ID:{source_id} F:{frame_idx}", fill=(180, 180, 180))

    return img


def create_demo_dataset(num_videos_per_class: int = 4, frames_per_video: int = 5) -> None:
    """
    Generate synthetic demo data across train, val, test, and unseen_test.
    """
    config.init_directories()

    demo_marker = config.DATA_DIR / "DEMO_DATA_PLACEHOLDER.txt"
    with open(demo_marker, "w", encoding="utf-8") as f:
        f.write(
            "NOTICE: The dataset in this folder consists of SYNTHETIC PLACEHOLDER images.\n"
            "Created for code testing, pipeline validation, and user interface demonstration.\n"
            "DO NOT report metrics computed on this data as genuine deepfake detection results.\n"
        )

    splits = {
        "train": {"real_vids": 6, "fake_vids": 6},
        "val": {"real_vids": 2, "fake_vids": 2},
        "test": {"real_vids": 2, "fake_vids": 2},
        "unseen_test": {"real_vids": 2, "fake_vids": 2},
    }

    manifest_records = []
    total_created = 0

    for split_name, counts in splits.items():
        # REAL videos
        for v_idx in range(counts["real_vids"]):
            source_id = f"demo_real_{split_name}_{v_idx:02d}"
            for f_idx in range(frames_per_video):
                img = draw_synthetic_face(is_fake=False, source_id=source_id, frame_idx=f_idx)
                target_dir = (
                    config.TRAIN_DIR / "real" if split_name == "train"
                    else (config.VAL_DIR / "real" if split_name == "val"
                    else (config.TEST_DIR / "real" if split_name == "test"
                    else config.UNSEEN_TEST_DIR / "real"))
                )
                target_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{source_id}_f{f_idx:03d}.jpg"
                file_path = target_dir / filename
                img.save(file_path, format="JPEG", quality=90)
                total_created += 1

                manifest_records.append(
                    {
                        "image_path": str(file_path),
                        "label": 0,
                        "class_name": "REAL",
                        "source_id": source_id,
                        "split": split_name,
                        "manipulation_category": "authentic_synthetic",
                    }
                )

        # FAKE videos
        for v_idx in range(counts["fake_vids"]):
            manipulation = "synthetic_unseen_diff" if split_name == "unseen_test" else "synthetic_blend"
            source_id = f"demo_fake_{split_name}_{v_idx:02d}"
            for f_idx in range(frames_per_video):
                img = draw_synthetic_face(
                    is_fake=True,
                    source_id=source_id,
                    frame_idx=f_idx,
                    manipulation_type=manipulation,
                )
                target_dir = (
                    config.TRAIN_DIR / "fake" if split_name == "train"
                    else (config.VAL_DIR / "fake" if split_name == "val"
                    else (config.TEST_DIR / "fake" if split_name == "test"
                    else config.UNSEEN_TEST_DIR / "fake"))
                )
                target_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{source_id}_f{f_idx:03d}.jpg"
                file_path = target_dir / filename
                img.save(file_path, format="JPEG", quality=90)
                total_created += 1

                manifest_records.append(
                    {
                        "image_path": str(file_path),
                        "label": 1,
                        "class_name": "FAKE",
                        "source_id": source_id,
                        "split": split_name,
                        "manipulation_category": manipulation,
                    }
                )

    import pandas as pd
    df = pd.DataFrame(manifest_records)
    df.to_csv(config.DATASET_MANIFEST_PATH, index=False)

    logger.info(f"Generated {total_created} synthetic demo images across train, val, test, and unseen_test.")
    print("\n" + "=" * 70)
    print("SYNTHETIC DEMO DATASET GENERATED")
    print("=" * 70)
    print(f"Total Images Created: {total_created}")
    print(f"Manifest Path       : {config.DATASET_MANIFEST_PATH}")
    print("Note: Images are watermarked and marked in manifest as synthetic.")
    print("=" * 70 + "\n")


def clean_demo_data():
    """Remove synthetic demo images from split folders."""
    logger.info("Cleaning up synthetic demo data...")
    for split_dir in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR, config.UNSEEN_TEST_DIR]:
        for sub in ["real", "fake"]:
            d = split_dir / sub
            if d.exists():
                for p in d.glob("demo_*"):
                    try:
                        p.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete {p}: {e}")

    demo_marker = config.DATA_DIR / "DEMO_DATA_PLACEHOLDER.txt"
    if demo_marker.exists():
        demo_marker.unlink()
    logger.info("Synthetic demo data cleaned successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or clean synthetic placeholder demo data.")
    parser.add_argument("--clean", action="store_true", help="Remove synthetic demo data.")
    args = parser.parse_args()

    if args.clean:
        clean_demo_data()
    else:
        create_demo_dataset()
