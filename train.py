"""
Model Training Script for MobileNetV2 Deepfake Detector.

Orchestrates two-phase transfer learning:
Phase 1: Train Dense classification head with frozen MobileNetV2 backbone.
Phase 2 (Optional): Fine-tune upper convolutional layers with reduced learning rate.

Saves best model (.keras), JSON history, training summary, and loss/accuracy plots.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf

import config
from src.data_loader import create_dataset_from_directory
from src.model import build_model, unfreeze_for_fine_tuning
from src.utils import get_logger, save_json, set_seed
from src.visualization import plot_training_history

logger = get_logger("train")


def train_model(
    batch_size: int = config.BATCH_SIZE,
    initial_epochs: int = config.INITIAL_EPOCHS,
    fine_tune_epochs: int = config.FINE_TUNE_EPOCHS,
    initial_lr: float = config.INITIAL_LR,
    fine_tune_lr: float = config.FINE_TUNE_LR,
    enable_fine_tuning: bool = True,
) -> Dict[str, Any]:
    """
    Execute full training pipeline.
    """
    set_seed(config.RANDOM_SEED)
    config.init_directories()

    start_time = time.time()
    logger.info("Initializing dataset pipelines for training and validation...")

    try:
        train_ds, train_count, train_dist = create_dataset_from_directory(
            config.TRAIN_DIR, batch_size=batch_size, is_training=True
        )
        val_ds, val_count, val_dist = create_dataset_from_directory(
            config.VAL_DIR, batch_size=batch_size, is_training=False
        )
    except Exception as e:
        logger.error(f"Cannot load datasets from {config.DATA_DIR}: {e}")
        print("\n" + "!" * 70)
        print("DATASET NOT FOUND OR EMPTY FOR TRAINING")
        print("!" * 70)
        print("Please prepare the dataset first using:")
        print("  python scripts/download_dataset.py  (or scripts/download_instructions.py)")
        print("  python scripts/prepare_dataset.py")
        print("To run with synthetic testing data immediately:")
        print("  python scripts/create_demo_data.py")
        print("!" * 70 + "\n")
        raise

    logger.info(
        f"Training set: {train_count} images (REAL={train_dist['REAL']}, FAKE={train_dist['FAKE']})"
    )
    logger.info(
        f"Validation set: {val_count} images (REAL={val_dist['REAL']}, FAKE={val_dist['FAKE']})"
    )

    # 1. Build initial model (backbone frozen)
    logger.info("Building MobileNetV2 transfer learning model...")
    model = build_model(learning_rate=initial_lr)

    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=1,
    )

    # Phase 1: Train classification head
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING CLASSIFICATION HEAD (BACKBONE FROZEN)")
    print("=" * 70)
    logger.info(f"Starting Phase 1 for {initial_epochs} epochs...")

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1,
    )

    merged_history = {k: [float(v) for v in vals] for k, vals in history_phase1.history.items()}
    fine_tune_start_epoch = None

    # Phase 2: Fine-tuning upper layers
    if enable_fine_tuning and fine_tune_epochs > 0:
        fine_tune_start_epoch = len(merged_history["loss"])
        print("\n" + "=" * 70)
        print(f"PHASE 2: FINE-TUNING TOP {config.FINE_TUNE_LAYERS} LAYERS")
        print("=" * 70)
        model = unfreeze_for_fine_tuning(
            model=model,
            fine_tune_layers=config.FINE_TUNE_LAYERS,
            fine_tune_lr=fine_tune_lr,
        )

        total_epochs = fine_tune_start_epoch + fine_tune_epochs
        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=fine_tune_start_epoch,
            epochs=total_epochs,
            callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
            verbose=1,
        )

        # Merge history
        for k, vals in history_phase2.history.items():
            if k in merged_history:
                merged_history[k].extend([float(v) for v in vals])
            else:
                merged_history[k] = [float(v) for v in vals]

    total_training_time = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {total_training_time} seconds.")

    # Save final model explicitly if checkpoint didn't trigger
    if not config.BEST_MODEL_PATH.exists():
        model.save(config.BEST_MODEL_PATH)
        logger.info(f"Saved model to {config.BEST_MODEL_PATH}")

    # Save training history JSON
    save_json(merged_history, config.TRAINING_HISTORY_PATH)
    logger.info(f"Saved history to {config.TRAINING_HISTORY_PATH}")

    # Plot training curves
    plot_path = plot_training_history(
        merged_history, fine_tune_epoch=fine_tune_start_epoch
    )

    # Machine-readable summary
    final_train_acc = merged_history.get("accuracy", [0.0])[-1]
    final_val_acc = merged_history.get("val_accuracy", [0.0])[-1]
    final_train_loss = merged_history.get("loss", [0.0])[-1]
    final_val_loss = merged_history.get("val_loss", [0.0])[-1]

    training_summary = {
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": total_training_time,
        "total_epochs_trained": len(merged_history.get("loss", [])),
        "fine_tuning_applied": enable_fine_tuning and (fine_tune_epochs > 0),
        "fine_tune_start_epoch": fine_tune_start_epoch,
        "final_metrics": {
            "train_accuracy": round(float(final_train_acc), 4),
            "val_accuracy": round(float(final_val_acc), 4),
            "train_loss": round(float(final_train_loss), 4),
            "val_loss": round(float(final_val_loss), 4),
        },
        "sample_counts": {
            "train_samples": train_count,
            "train_distribution": train_dist,
            "val_samples": val_count,
            "val_distribution": val_dist,
        },
        "saved_artifacts": {
            "model_path": str(config.BEST_MODEL_PATH),
            "history_path": str(config.TRAINING_HISTORY_PATH),
            "curves_plot_path": str(plot_path),
        },
    }

    save_json(training_summary, config.TRAINING_SUMMARY_PATH)
    logger.info(f"Saved training summary to {config.TRAINING_SUMMARY_PATH}")

    print("\n" + "=" * 70)
    print("TRAINING PROCESS COMPLETED")
    print("=" * 70)
    print(f"Elapsed Time       : {total_training_time} s")
    print(f"Total Epochs       : {len(merged_history['loss'])}")
    print(f"Final Train Acc    : {final_train_acc * 100:.2f}%")
    print(f"Final Val Acc      : {final_val_acc * 100:.2f}%")
    print(f"Best Model File    : {config.BEST_MODEL_PATH}")
    print(f"Curves Plot File   : {plot_path}")
    print("=" * 70 + "\n")

    return training_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Deepfake Detector.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=config.INITIAL_EPOCHS, help="Phase 1 initial epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS, help="Phase 2 fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=config.INITIAL_LR, help="Phase 1 learning rate.")
    parser.add_argument("--no-fine-tune", action="store_true", help="Disable phase 2 fine-tuning.")
    args = parser.parse_args()

    train_model(
        batch_size=args.batch_size,
        initial_epochs=args.epochs,
        fine_tune_epochs=0 if args.no_fine_tune else args.fine_tune_epochs,
        initial_lr=args.lr,
        enable_fine_tuning=not args.no_fine_tune,
    )
