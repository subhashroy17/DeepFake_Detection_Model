"""
Video Frame Extraction & Facial Crop Pipeline.

Iterates over source videos, samples frames at intervals, extracts prominent
facial regions using OpenCV Haar cascade, resizes to target shape, and writes
organized JPG crops while recording source video IDs to prevent data leakage.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.preprocessing import get_face_detector
from src.utils import get_logger, save_json

logger = get_logger("extract_frames")


def find_metadata_file(search_dir: Path) -> Optional[Path]:
    """Search for metadata.json or similar label files in directory."""
    candidates = list(search_dir.glob("**/metadata.json")) + list(search_dir.glob("**/*labels*.json"))
    if candidates:
        return candidates[0]
    return None


def load_video_metadata(metadata_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load JSON metadata dictionary mapping video filenames to label attributes."""
    if metadata_path is None or not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Loaded metadata for {len(data)} videos from {metadata_path.name}")
            return data
    except Exception as e:
        logger.warning(f"Could not parse metadata at {metadata_path}: {e}")
        return {}


def infer_label_and_meta(
    video_path: Path, metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Infer label (REAL vs FAKE), source video ID, and manipulation technique
    from metadata or directory structure.
    """
    filename = video_path.name
    # 1. Check metadata dictionary
    if filename in metadata:
        entry = metadata[filename]
        raw_label = str(entry.get("label", "")).upper()
        label = "FAKE" if "FAKE" in raw_label else "REAL"
        original = entry.get("original", video_path.stem)
        manipulation = entry.get("manipulation", entry.get("type", "unknown"))
        return {
            "label": label,
            "source_id": original or video_path.stem,
            "manipulation_category": manipulation,
        }

    # 2. Check parent folder names
    parent_names = [p.name.lower() for p in video_path.parents]
    if any("fake" in p or "manipulated" in p or "deepfake" in p for p in parent_names):
        label = "FAKE"
    elif any("real" in p or "original" in p or "authentic" in p for p in parent_names):
        label = "REAL"
    else:
        label = "UNKNOWN"

    # Identify potential manipulation category from folder hierarchy
    manipulation = "unknown"
    for cat in ["deepfakes", "face2face", "faceswap", "neuraltextures"]:
        if any(cat in p for p in parent_names):
            manipulation = cat
            break

    return {
        "label": label,
        "source_id": video_path.stem,
        "manipulation_category": manipulation,
    }


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    max_frames: int = config.FRAMES_PER_VIDEO,
    interval: int = config.FRAME_INTERVAL,
) -> List[Dict[str, Any]]:
    """
    Extract facial crops from a single video file.

    Returns:
        List of dictionaries with crop metadata.
    """
    detector = get_face_detector()
    extracted_records = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video file: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved_count = 0

    video_id = video_path.stem

    while cap.isOpened() and saved_count < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Sample at regular intervals to prevent redundant identical expressions
        if frame_idx % interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            crop_rgb, face_found, bbox = detector.detect_and_crop(frame_rgb)

            # High quality resize
            pil_crop = Image.fromarray(crop_rgb).resize(
                (config.IMG_WIDTH, config.IMG_HEIGHT), Image.Resampling.BICUBIC
            )

            crop_filename = f"{video_id}_f{frame_idx:05d}.jpg"
            crop_path = output_dir / crop_filename
            pil_crop.save(crop_path, format="JPEG", quality=95)

            extracted_records.append(
                {
                    "crop_filename": crop_filename,
                    "crop_path": str(crop_path),
                    "video_id": video_id,
                    "frame_index": frame_idx,
                    "face_detected": face_found,
                    "bbox": bbox,
                }
            )
            saved_count += 1

        frame_idx += 1

    cap.release()
    return extracted_records


def run_extraction(
    input_dir: Path,
    output_dir: Path = config.PROCESSED_DATA_DIR,
    max_frames: int = config.FRAMES_PER_VIDEO,
    interval: int = config.FRAME_INTERVAL,
) -> Dict[str, Any]:
    """
    Process all videos in input_dir and extract facial crops.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = find_metadata_file(input_dir)
    metadata = load_video_metadata(metadata_path)

    # Find all supported video files
    video_files = [
        p for p in input_dir.rglob("*") if p.suffix.lower() in config.VALID_VIDEO_EXTENSIONS and p.is_file()
    ]

    logger.info(f"Found {len(video_files)} video files in {input_dir}")
    if len(video_files) == 0:
        logger.warning(f"No video files found with extensions {config.VALID_VIDEO_EXTENSIONS}")
        return {"total_videos": 0, "extracted_crops": 0, "records": []}

    all_records = []
    failed_videos = []

    for idx, video_path in enumerate(video_files, start=1):
        try:
            info = infer_label_and_meta(video_path, metadata)
            records = extract_frames_from_video(
                video_path=video_path,
                output_dir=output_dir,
                max_frames=max_frames,
                interval=interval,
            )

            for rec in records:
                rec.update(
                    {
                        "source_video_path": str(video_path),
                        "label": info["label"],
                        "source_id": info["source_id"],
                        "manipulation_category": info["manipulation_category"],
                    }
                )
                all_records.append(rec)

            if idx % 10 == 0 or idx == len(video_files):
                logger.info(f"Processed [{idx}/{len(video_files)}] videos. Extracted crops: {len(all_records)}")

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            failed_videos.append({"video": str(video_path), "error": str(e)})

    summary = {
        "total_videos_found": len(video_files),
        "total_crops_extracted": len(all_records),
        "failed_videos": failed_videos,
        "max_frames_per_video": max_frames,
        "frame_interval": interval,
    }

    summary_file = config.MANIFESTS_DIR / "extraction_summary.json"
    save_json(summary, summary_file)
    logger.info(f"Saved extraction summary to {summary_file}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract facial crops from videos.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(config.RAW_DATA_DIR),
        help="Directory containing video files or raw dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.PROCESSED_DATA_DIR),
        help="Directory to save extracted facial images.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=config.FRAMES_PER_VIDEO,
        help="Maximum frames per video.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=config.FRAME_INTERVAL,
        help="Frame interval for sampling.",
    )
    args = parser.parse_args()

    run_extraction(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        max_frames=args.max_frames,
        interval=args.interval,
    )
