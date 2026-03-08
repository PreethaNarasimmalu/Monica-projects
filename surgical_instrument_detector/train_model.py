"""
train_model.py
--------------
Trains a YOLOv8n model on the downloaded surgical instruments dataset.

Usage:
    python train_model.py

Prerequisite:
    Run download_dataset.py first to prepare the dataset folder.
"""

import os
from ultralytics import YOLO

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_YAML      = "dataset/data.yaml"   # Path to the dataset config file
BASE_MODEL     = "yolov8n.pt"          # Pre-trained YOLOv8 nano weights
EPOCHS         = 50                    # Number of training epochs
IMAGE_SIZE     = 640                   # Input image size (pixels)
BATCH_SIZE     = 16                    # Training batch size
CONF_THRESHOLD = 0.5                   # Confidence threshold for evaluation
RUN_NAME       = "surgical_instrument_detector"  # Name used for results folder
# ──────────────────────────────────────────────────────────────────────────────


def train():
    """Load YOLOv8n and train on the surgical instruments dataset."""
    print("=" * 60)
    print("YOLOv8n Training — Surgical Instrument Detector")
    print("=" * 60)

    # Validate that the dataset config exists
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"Dataset config not found at '{DATA_YAML}'.\n"
            "Please run download_dataset.py first."
        )

    # Load the base YOLOv8n model (downloads weights on first run)
    print(f"\n[1/2] Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # Start training
    print(f"[2/2] Starting training for {EPOCHS} epochs...")
    print(f"      Dataset  : {os.path.abspath(DATA_YAML)}")
    print(f"      Image sz : {IMAGE_SIZE}px  |  Batch: {BATCH_SIZE}")
    print()

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=CONF_THRESHOLD,
        name=RUN_NAME
    )

    return model, results


def save_and_report(model, results):
    """Save the best weights and print training metrics."""
    # The best weights are saved automatically by Ultralytics under runs/
    best_weights = os.path.join(
        "runs", "detect", RUN_NAME, "weights", "best.pt"
    )

    print("\n─── Training Results ───────────────────────────────────────")

    if os.path.exists(best_weights):
        print(f"  Best weights saved : {os.path.abspath(best_weights)}")
    else:
        print("  Warning: best.pt not found at expected path.")
        print(f"  Check runs/detect/{RUN_NAME}/weights/")

    # Extract and display key metrics from the results object
    try:
        metrics = results.results_dict
        print(f"  mAP@50             : {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  Precision          : {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall             : {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    except Exception:
        # Fallback: metrics may be structured differently in some versions
        print("  (Could not parse detailed metrics — check runs/ folder for CSV logs)")

    print("───────────────────────────────────────────────────────────\n")
    print("Training complete. Run detect_and_count.py to test inference.")


if __name__ == "__main__":
    model, results = train()
    save_and_report(model, results)
