"""
Dataset Preparation and Leakage-Free Splitting Pipeline.

Groups extracted frames or image directories by source video / identity to
guarantee strict independence between splits (no data leakage). Organizes data
into train, val, standard test, and (when manipulation category labels exist)
unseen-manipulation test sets. Generates the master CSV manifest and JSON stats.
"""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_logger, save_json, set_seed

logger = get_logger("prepare_dataset")


def discover_processed_images(
    source_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Scan source directory for images and deduce video source IDs and labels.
    Handles both flat extracted frames and subfolder-based datasets (e.g., real/fake).
    """
    records = []
    # Search all valid image files
    image_paths = [
        p for p in source_dir.rglob("*") if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS and p.is_file()
    ]

    logger.info(f"Discovered {len(image_paths)} total images in {source_dir}")

    for img_path in image_paths:
        filename = img_path.name
        parent_names = [p.name.lower() for p in img_path.parents]

        # Determine label
        if any("fake" in p or "manipulated" in p or "deepfake" in p for p in parent_names) or "fake" in filename.lower():
            label = 1
            class_name = "FAKE"
        elif any("real" in p or "original" in p or "authentic" in p for p in parent_names) or "real" in filename.lower():
            label = 0
            class_name = "REAL"
        else:
            # Fallback based on filename conventions
            label = 0 if "real" in filename.lower() else 1
            class_name = "REAL" if label == 0 else "FAKE"

        # Determine manipulation technique if tagged
        manipulation = "unknown"
        for tech in ["deepfakes", "face2face", "faceswap", "neuraltextures"]:
            if any(tech in p for p in parent_names) or tech in filename.lower():
                manipulation = tech
                break

        # Extract source ID from filename (e.g. video123_f00015.jpg -> video123)
        stem = img_path.stem
        if "_f" in stem:
            source_id = stem.rsplit("_f", 1)[0]
        else:
            source_id = stem.split("_")[0] if "_" in stem else stem

        records.append(
            {
                "original_path": str(img_path),
                "filename": filename,
                "label": label,
                "class_name": class_name,
                "source_id": source_id,
                "manipulation_category": manipulation,
            }
        )

    return records


def split_by_source_id(
    records: List[Dict[str, Any]],
    unseen_method: Optional[str] = None,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.RANDOM_SEED,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split records by source_id into train, val, test, and unseen_test.
    Guarantees that all frames belonging to a source_id stay in one single split.
    """
    set_seed(seed)
    rng = random.Random(seed)

    # Separate unseen manipulation records if genuinely specified and present
    unseen_records = []
    standard_records = []

    available_manipulations = {r["manipulation_category"] for r in records if r["manipulation_category"] != "unknown"}

    if unseen_method and unseen_method.lower() in {m.lower() for m in available_manipulations}:
        target_unseen = unseen_method.lower()
        logger.info(f"Holding out manipulation category '{target_unseen}' exclusively for UNSEEN evaluation.")
        for r in records:
            if r["manipulation_category"].lower() == target_unseen:
                r["split"] = "unseen_test"
                unseen_records.append(r)
            else:
                standard_records.append(r)
    else:
        if unseen_method:
            logger.warning(
                f"Requested unseen method '{unseen_method}' not found among labeled categories {available_manipulations}. "
                "All samples will be assigned across standard train/val/test splits."
            )
        standard_records = records

    # Group standard records by source_id
    real_sources = defaultdict(list)
    fake_sources = defaultdict(list)

    for r in standard_records:
        if r["label"] == 0:
            real_sources[r["source_id"]].append(r)
        else:
            fake_sources[r["source_id"]].append(r)

    def allocate_sources(source_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        sources = list(source_dict.keys())
        rng.shuffle(sources)

        n = len(sources)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0

        train_s = set(sources[:n_train])
        val_s = set(sources[n_train : n_train + n_val])
        test_s = set(sources[n_train + n_val :])

        if not test_s and n > 1:
            # Ensure at least 1 source in test if multiple exist
            test_s = {sources[-1]}
            if sources[-1] in train_s:
                train_s.remove(sources[-1])
            elif sources[-1] in val_s:
                val_s.remove(sources[-1])

        allocated = []
        for sid, recs in source_dict.items():
            split_name = "train" if sid in train_s else ("val" if sid in val_s else "test")
            for rec in recs:
                rec["split"] = split_name
                allocated.append(rec)
        return allocated

    final_records = allocate_sources(real_sources) + allocate_sources(fake_sources) + unseen_records

    # Summary statistics
    split_counts = defaultdict(lambda: {"REAL": 0, "FAKE": 0, "total": 0})
    split_sources = defaultdict(lambda: defaultdict(set))

    for r in final_records:
        sp = r["split"]
        cname = r["class_name"]
        split_counts[sp][cname] += 1
        split_counts[sp]["total"] += 1
        split_sources[sp][cname].add(r["source_id"])

    stats = {
        "split_counts": dict(split_counts),
        "unique_source_counts": {
            sp: {k: len(v) for k, v in s_dict.items()} for sp, s_dict in split_sources.items()
        },
        "manipulation_categories_found": list(available_manipulations),
        "held_out_unseen_method": unseen_method if unseen_records else None,
    }

    return final_records, stats


def organize_split_directories(
    records: List[Dict[str, Any]],
    copy_files: bool = True,
) -> None:
    """
    Populate data/{train,val,test,unseen_test}/{real,fake}/ with processed images.
    """
    config.init_directories()

    logger.info("Organizing image files into split directories...")
    for rec in records:
        src = Path(rec["original_path"])
        split = rec["split"]
        subfolder = "real" if rec["label"] == 0 else "fake"

        if split == "train":
            dst_dir = config.TRAIN_DIR / subfolder
        elif split == "val":
            dst_dir = config.VAL_DIR / subfolder
        elif split == "test":
            dst_dir = config.TEST_DIR / subfolder
        elif split == "unseen_test":
            dst_dir = config.UNSEEN_TEST_DIR / subfolder
        else:
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / rec["filename"]

        if copy_files and src != dst_path:
            shutil.copy2(src, dst_path)

        rec["target_path"] = str(dst_path)


def prepare_dataset(
    source_dir: Optional[Path] = None,
    unseen_method: Optional[str] = None,
    copy_files: bool = True,
) -> Dict[str, Any]:
    """
    Complete preparation pipeline: discovery, source-level split, file organization,
    and manifest generation.
    """
    config.init_directories()
    target_source = source_dir or config.PROCESSED_DATA_DIR
    if not target_source.exists():
        logger.error(f"Source directory does not exist: {target_source}")
        return {"status": "error", "message": "Source directory missing"}

    records = discover_processed_images(target_source)
    if not records:
        logger.warning(f"No usable images found in {target_source}")
        return {"status": "empty", "total_images": 0}

    split_records, stats = split_by_source_id(records, unseen_method=unseen_method)
    organize_split_directories(split_records, copy_files=copy_files)

    # Save CSV manifest
    df = pd.DataFrame(split_records)
    manifest_csv = config.DATASET_MANIFEST_PATH
    df.to_csv(manifest_csv, index=False)
    logger.info(f"Saved dataset manifest to {manifest_csv}")

    # Save stats JSON
    save_json(stats, config.DATASET_STATS_PATH)
    logger.info(f"Saved dataset stats to {config.DATASET_STATS_PATH}")

    print("\n" + "=" * 70)
    print("DATASET PREPARATION & SPLITTING REPORT")
    print("=" * 70)
    print(f"Total Usable Images : {len(split_records):,}")
    print(f"Manifest CSV Path   : {manifest_csv}")
    print("\nSplit Distributions:")
    for split_name, counts in stats["split_counts"].items():
        print(f"  [{split_name.upper():<12}] Total: {counts['total']:<5} (REAL: {counts['REAL']}, FAKE: {counts['FAKE']})")

    if stats["held_out_unseen_method"]:
        print(f"\nHeld-out Unseen Manipulation: {stats['held_out_unseen_method']}")
    else:
        print("\nHeld-out Unseen Manipulation: None (manipulation families not labeled in dataset)")
    print("=" * 70 + "\n")

    return {"status": "success", "total_images": len(split_records), "stats": stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and split without data leakage.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(config.PROCESSED_DATA_DIR),
        help="Directory with processed facial images.",
    )
    parser.add_argument(
        "--unseen-method",
        type=str,
        default=None,
        help="Optional manipulation technique name to hold out exclusively for unseen evaluation.",
    )
    args = parser.parse_args()

    prepare_dataset(
        source_dir=Path(args.source_dir) if args.source_dir else None,
        unseen_method=args.unseen_method,
    )
