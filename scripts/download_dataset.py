"""
Dataset Downloader & Structure Inspector using KaggleHub.

Safely checks available disk space, redirects cache to high-capacity storage
drives (e.g., D: drive when C: drive has low space), downloads the dataset via
KaggleHub, and inspects its internal directory hierarchy, metadata, and file types.
"""

import argparse
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import format_bytes, get_logger, save_json

logger = get_logger("download_dataset")

# Estimated archive + extraction requirement for the full DFD dataset
REQUIRED_FREE_SPACE_BYTES = 25 * 1024 * 1024 * 1024  # 25 GB


def configure_storage_drive(min_free_bytes: int = REQUIRED_FREE_SPACE_BYTES) -> Path:
    """
    Check disk space on default cache drive. If insufficient (<25GB),
    automatically detect and configure an alternate drive with sufficient space (e.g. D: drive).
    """
    default_home = Path.home()
    c_usage = shutil.disk_usage(default_home)

    logger.info(f"System Drive ({default_home.anchor}) Free Space: {format_bytes(c_usage.free)}")

    if c_usage.free >= min_free_bytes:
        cache_dir = default_home / ".cache" / "kagglehub"
        logger.info(f"Using default system cache directory: {cache_dir}")
        return cache_dir

    logger.warning(
        f"Default drive ({default_home.anchor}) has only {format_bytes(c_usage.free)} free space. "
        f"Required: {format_bytes(min_free_bytes)}. Searching alternative drives..."
    )

    # Check Windows alternate drive letters (D:, E:, etc.)
    for drive_letter in ["D", "E", "F"]:
        drive_path = Path(f"{drive_letter}:\\")
        if drive_path.exists():
            try:
                usage = shutil.disk_usage(drive_path)
                logger.info(f"Drive {drive_letter}: Free Space: {format_bytes(usage.free)}")
                if usage.free >= min_free_bytes:
                    alt_cache = drive_path / "kagglehub_cache"
                    alt_cache.mkdir(parents=True, exist_ok=True)
                    os.environ["KAGGLEHUB_CACHE"] = str(alt_cache)
                    logger.info(f"Redirected KAGGLEHUB_CACHE to high-capacity storage: {alt_cache}")
                    return alt_cache
            except Exception as e:
                logger.warning(f"Could not inspect drive {drive_letter}: {e}")

    raise OSError(
        f"Insufficient disk space across all available storage drives! "
        f"The dataset requires at least {format_bytes(min_free_bytes)} free space."
    )


def inspect_directory(root_path: Path) -> dict:
    """
    Recursively catalog file types, subdirectories, total size, and metadata candidates.
    """
    total_size = 0
    extension_counts = Counter()
    dir_structure = defaultdict(list)
    metadata_candidates = []
    all_files = []

    logger.info(f"Recursively inspecting dataset at: {root_path} ...")

    for dirpath, dirnames, filenames in os.walk(root_path):
        rel_dir = os.path.relpath(dirpath, root_path)
        if rel_dir == ".":
            rel_dir = "root"

        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                file_size = file_path.stat().st_size
                total_size += file_size
                ext = file_path.suffix.lower() if file_path.suffix else "no_extension"
                extension_counts[ext] += 1
                all_files.append(str(file_path))

                # Track potential metadata files
                if ext in {".json", ".csv", ".txt", ".tsv", ".yaml", ".xml"}:
                    metadata_candidates.append(
                        {
                            "relative_path": str(Path(rel_dir) / filename),
                            "size_bytes": file_size,
                            "extension": ext,
                        }
                    )
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not stat file: {file_path}: {e}")

        dir_structure[rel_dir] = dirnames[:10]

    summary = {
        "dataset_root": str(root_path),
        "total_files": len(all_files),
        "total_size_bytes": total_size,
        "total_size_formatted": format_bytes(total_size),
        "file_extensions": dict(extension_counts),
        "metadata_files": metadata_candidates,
        "sample_directories": dict(dir_structure),
    }

    return summary


def download_dataset(dataset_id: str = config.KAGGLE_DATASET_ID, check_only: bool = False) -> dict:
    """
    Execute KaggleHub dataset download with disk space protection and structure inspection.
    """
    config.init_directories()

    try:
        cache_dir = configure_storage_drive()
    except OSError as e:
        logger.error(f"Storage check failed: {e}")
        return {"status": "error", "error": str(e)}

    if check_only:
        return {"status": "ready", "cache_dir": str(cache_dir)}

    logger.info(f"Initiating download for Kaggle dataset: '{dataset_id}' ...")

    try:
        import kagglehub

        download_path_str = kagglehub.dataset_download(dataset_id)
        download_path = Path(download_path_str).resolve()
        logger.info(f"Dataset successfully retrieved at: {download_path}")

        inspection = inspect_directory(download_path)

        manifest_info_path = config.MANIFESTS_DIR / "dataset_inspection.json"
        save_json(inspection, manifest_info_path)
        logger.info(f"Saved dataset inspection report to {manifest_info_path}")

        print("\n" + "=" * 70)
        print("DATASET ACQUISITION & INSPECTION REPORT")
        print("=" * 70)
        print(f"Dataset Identifier : {dataset_id}")
        print(f"Local Storage Path : {download_path}")
        print(f"Total Size         : {inspection['total_size_formatted']}")
        print(f"Total Files        : {inspection['total_files']:,}")
        print("File Type Breakdown:")
        for ext, count in sorted(inspection["file_extensions"].items(), key=lambda x: -x[1]):
            print(f"  - {ext}: {count:,} files")

        if inspection["metadata_files"]:
            print("Found Metadata Candidates:")
            for meta in inspection["metadata_files"]:
                print(f"  - {meta['relative_path']} ({format_bytes(meta['size_bytes'])})")
        else:
            print("No metadata (.json/.csv) files detected at root.")
        print("=" * 70 + "\n")

        return {"status": "success", "path": str(download_path), "inspection": inspection}

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to acquire dataset via kagglehub: {error_msg}")

        print("\n" + "!" * 70)
        print("KAGGLEHUB DOWNLOAD ISSUE / AUTHENTICATION")
        print("!" * 70)
        print(f"Error details: {error_msg}")
        print("\nGuidance:")
        print("1. KAGGLE AUTHENTICATION:")
        print("   Visit https://www.kaggle.com/settings -> 'Create New Token'")
        print("   Place 'kaggle.json' in your user directory: ~/.kaggle/kaggle.json")
        print("   (On Windows: C:\\Users\\<Username>\\.kaggle\\kaggle.json)")
        print("2. DATASET IDENTIFIER:")
        print(f"   Verify dataset accessibility at https://www.kaggle.com/datasets/{dataset_id}")
        print("3. MANUAL PREPARATION:")
        print(f"   Place images or videos into {config.RAW_DATA_DIR} and run:")
        print("   python scripts/prepare_dataset.py")
        print("!" * 70 + "\n")

        return {"status": "error", "error": error_msg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and inspect dataset using KaggleHub.")
    parser.add_argument("--check-only", action="store_true", help="Only verify disk space and configuration without downloading.")
    parser.add_argument("--dataset-id", type=str, default=config.KAGGLE_DATASET_ID, help="Kaggle dataset identifier.")
    args = parser.parse_args()

    download_dataset(dataset_id=args.dataset_id, check_only=args.check_only)
