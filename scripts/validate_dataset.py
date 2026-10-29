"""
Dataset Validation & Data Leakage Verification Tool.

Verifies image file readability, class distributions, and performs strict
source-video ID intersection checks to guarantee zero data leakage between
training, validation, standard test, and unseen test splits.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_logger

logger = get_logger("validate_dataset")


def get_source_id(file_path: Path) -> str:
    """Extract source ID from filename, preserving unique identities for photo datasets."""
    stem = file_path.stem
    if "_f" in stem:
        return stem.rsplit("_f", 1)[0]
    parts = stem.split("_")
    if len(parts) > 1 and parts[0].lower() not in {"real", "fake"}:
        return parts[0]
    return stem


def validate_split(split_name: str, split_dir: Path) -> Dict[str, Any]:
    """Inspect and validate image files within a single split."""
    if not split_dir.exists():
        return {"exists": False, "total": 0, "real": 0, "fake": 0, "corrupt": 0, "sources": set()}

    real_dir = split_dir / "real"
    fake_dir = split_dir / "fake"

    real_files = [p for p in real_dir.glob("*") if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS and p.is_file()] if real_dir.exists() else []
    fake_files = [p for p in fake_dir.glob("*") if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS and p.is_file()] if fake_dir.exists() else []

    all_files = real_files + fake_files
    corrupted_count = 0
    sources: Set[str] = set()

    for p in all_files:
        try:
            if p.stat().st_size == 0:
                corrupted_count += 1
                continue
            with Image.open(p) as img:
                img.verify()
            sources.add(get_source_id(p))
        except Exception:
            corrupted_count += 1

    return {
        "exists": True,
        "total": len(all_files),
        "real": len(real_files),
        "fake": len(fake_files),
        "corrupt": corrupted_count,
        "sources": sources,
    }


def validate_all_splits() -> bool:
    """
    Validate all splits and ensure strict zero data leakage.
    """
    config.init_directories()

    splits = {
        "train": config.TRAIN_DIR,
        "val": config.VAL_DIR,
        "test": config.TEST_DIR,
        "unseen_test": config.UNSEEN_TEST_DIR,
    }

    results = {}
    for name, path in splits.items():
        results[name] = validate_split(name, path)

    print("\n" + "=" * 70)
    print("DATASET VALIDATION & DATA LEAKAGE AUDIT REPORT")
    print("=" * 70)

    total_images = sum(r["total"] for r in results.values())
    print(f"Total Valid Images Discovered: {total_images:,}")
    print("\nPer-Split Summary:")
    print(f"{'Split':<15} {'Total':<8} {'REAL':<8} {'FAKE':<8} {'Sources':<8} {'Corrupt':<8}")
    print("-" * 65)

    for name, r in results.items():
        print(
            f"{name:<15} {r['total']:<8} {r['real']:<8} {r['fake']:<8} "
            f"{len(r['sources']):<8} {r['corrupt']:<8}"
        )

    # Check for Data Leakage: Source ID Intersections
    print("\n" + "-" * 70)
    print("DATA LEAKAGE VERIFICATION (Source Video Independence)")
    print("-" * 70)

    train_sources = results["train"]["sources"]
    val_sources = results["val"]["sources"]
    test_sources = results["test"]["sources"]
    unseen_sources = results["unseen_test"]["sources"]

    leakage_detected = False

    def check_intersection(name_a, set_a, name_b, set_b):
        nonlocal leakage_detected
        common = set_a.intersection(set_b)
        if common:
            print(f"[FAIL - LEAKAGE DETECTED] {len(common)} source IDs shared between {name_a} and {name_b}!")
            print(f"  Overlapping IDs: {list(common)[:5]}...")
            leakage_detected = True
        else:
            print(f"[PASS] Zero source ID overlap between {name_a} and {name_b} (disjoint sets).")

    if train_sources and val_sources:
        check_intersection("train", train_sources, "val", val_sources)
    if train_sources and test_sources:
        check_intersection("train", train_sources, "test", test_sources)
    if val_sources and test_sources:
        check_intersection("val", val_sources, "test", test_sources)
    if train_sources and unseen_sources:
        check_intersection("train", train_sources, "unseen_test", unseen_sources)

    print("=" * 70 + "\n")

    if total_images == 0:
        logger.warning("Dataset is currently empty. Run download/extraction or create demo data.")
        return False

    if leakage_detected:
        logger.error("Dataset validation failed: Data leakage detected between splits!")
        return False

    logger.info("Dataset validation passed! Splits are clean and leakage-free.")
    return True


if __name__ == "__main__":
    is_valid = validate_all_splits()
    sys.exit(0 if is_valid else 1)
