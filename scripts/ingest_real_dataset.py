"""
Ingest and Split Genuine Real Human Face & Deepfake Dataset.

Loads real face photographs and deepfake images from KaggleHub cache on D: drive,
creates a lightweight, balanced subset (500 train, 100 val, 100 test, 100 unseen_test, ~16MB total),
copies them into the project data/ directory, and generates the master CSV manifest.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONUTF8"] = "1"

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_logger, save_json, set_seed

logger = get_logger("ingest_real_dataset")

SOURCE_CACHE_DIR = Path(r"D:\kagglehub_cache\datasets\kshitizbhargava\deepfake-face-images\versions\1\Final Dataset")


def ingest_dataset(
    train_per_class: int = 250,
    val_per_class: int = 50,
    test_per_class: int = 50,
    unseen_per_class: int = 50,
    seed: int = config.RANDOM_SEED,
) -> None:
    """
    Ingest and split real and fake face images into project directories.
    """
    set_seed(seed)
    rng = random.Random(seed)
    config.init_directories()

    real_src_dir = SOURCE_CACHE_DIR / "Real"
    fake_src_dir = SOURCE_CACHE_DIR / "Fake"

    if not real_src_dir.exists() or not fake_src_dir.exists():
        raise FileNotFoundError(
            f"Source dataset not found at {SOURCE_CACHE_DIR}."
        )

    all_real = sorted(list(real_src_dir.glob("*.jpg")) + list(real_src_dir.glob("*.png")))
    all_fake = sorted(list(fake_src_dir.glob("*.jpg")) + list(fake_src_dir.glob("*.png")))

    total_required = train_per_class + val_per_class + test_per_class + unseen_per_class

    logger.info(f"Available in source dataset: Real={len(all_real)}, Fake={len(all_fake)}")
    logger.info(f"Sampling {total_required} images per class (Total: {total_required * 2})")

    rng.shuffle(all_real)
    rng.shuffle(all_fake)

    t_end = train_per_class
    v_end = t_end + val_per_class
    te_end = v_end + test_per_class
    u_end = te_end + unseen_per_class

    splits_map = {
        "train": {
            "real": all_real[:t_end],
            "fake": all_fake[:t_end],
            "dir": config.TRAIN_DIR,
        },
        "val": {
            "real": all_real[t_end:v_end],
            "fake": all_fake[t_end:v_end],
            "dir": config.VAL_DIR,
        },
        "test": {
            "real": all_real[v_end:te_end],
            "fake": all_fake[v_end:te_end],
            "dir": config.TEST_DIR,
        },
        "unseen_test": {
            "real": all_real[te_end:u_end],
            "fake": all_fake[te_end:u_end],
            "dir": config.UNSEEN_TEST_DIR,
        },
    }

    manifest_records = []
    total_copied = 0

    print("\n" + "=" * 70)
    print("INGESTING REAL HUMAN FACE & DEEPFAKE DATASET")
    print("=" * 70)

    for split_name, data in splits_map.items():
        dst_real_dir = data["dir"] / "real"
        dst_fake_dir = data["dir"] / "fake"
        dst_real_dir.mkdir(parents=True, exist_ok=True)
        dst_fake_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing files in target split
        for f in dst_real_dir.glob("*"):
            if f.is_file():
                f.unlink()
        for f in dst_fake_dir.glob("*"):
            if f.is_file():
                f.unlink()

        # Copy REAL images
        for p in data["real"]:
            target_path = dst_real_dir / p.name
            shutil.copy2(p, target_path)
            total_copied += 1
            manifest_records.append(
                {
                    "image_path": str(target_path),
                    "label": 0,
                    "class_name": "REAL",
                    "source_id": p.stem,
                    "split": split_name,
                    "manipulation_category": "authentic_face",
                }
            )

        # Copy FAKE images
        for p in data["fake"]:
            target_path = dst_fake_dir / p.name
            shutil.copy2(p, target_path)
            total_copied += 1
            manifest_records.append(
                {
                    "image_path": str(target_path),
                    "label": 1,
                    "class_name": "FAKE",
                    "source_id": p.stem,
                    "split": split_name,
                    "manipulation_category": "deepfake_face",
                }
            )

        print(
            f"  [{split_name.upper():<12}] REAL: {len(data['real']):<4} | "
            f"FAKE: {len(data['fake']):<4} -> {data['dir'].relative_to(config.PROJECT_ROOT)}"
        )

    # Save manifest CSV
    df = pd.DataFrame(manifest_records)
    manifest_csv = config.DATASET_MANIFEST_PATH
    df.to_csv(manifest_csv, index=False)
    logger.info(f"Saved dataset manifest to {manifest_csv}")

    # Save stats JSON
    stats = {
        "dataset_source": "kshitizbhargava/deepfake-face-images",
        "total_images": total_copied,
        "splits": {
            split: {
                "real": len(data["real"]),
                "fake": len(data["fake"]),
                "total": len(data["real"]) + len(data["fake"]),
            }
            for split, data in splits_map.items()
        },
    }
    save_json(stats, config.DATASET_STATS_PATH)

    print("=" * 70)
    print(f"Total Ingested Real Face Images: {total_copied:,}")
    print(f"Manifest CSV Path              : {manifest_csv}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest genuine real human face deepfake dataset.")
    parser.add_argument("--train", type=int, default=250, help="Images per class in train split.")
    parser.add_argument("--val", type=int, default=50, help="Images per class in val split.")
    parser.add_argument("--test", type=int, default=50, help="Images per class in test split.")
    parser.add_argument("--unseen", type=int, default=50, help="Images per class in unseen_test split.")
    args = parser.parse_args()

    ingest_dataset(
        train_per_class=args.train,
        val_per_class=args.val,
        test_per_class=args.test,
        unseen_per_class=args.unseen,
    )
