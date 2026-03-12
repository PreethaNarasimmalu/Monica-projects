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
import yaml
from ultralytics import YOLO

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Use absolute path so paths resolve correctly regardless of working directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
DATA_YAML      = "dataset/data.yaml"   # fallback; resolved dynamically below
BASE_MODEL     = "yolov8n.pt"          # Pre-trained YOLOv8 nano weights
EPOCHS         = 50                    # Number of training epochs
IMAGE_SIZE     = 640                   # Input image size (pixels)
BATCH_SIZE     = 16                    # Training batch size
CONF_THRESHOLD = 0.5                   # Confidence threshold for evaluation
RUN_NAME       = "surgical_instrument_detector"  # Name used for results folder
# ──────────────────────────────────────────────────────────────────────────────


def find_data_yaml():
    """Locate data.yaml inside OUTPUT_DIR, searching one level deep first.

    The subdirectory data.yaml (dataset/<project-version>/data.yaml) has correct
    absolute image paths, so it is preferred over the root dataset/data.yaml.
    """
    # Subdirectory path first: dataset/<project-version>/data.yaml
    if os.path.isdir(OUTPUT_DIR):
        for entry in sorted(os.listdir(OUTPUT_DIR)):
            candidate = os.path.join(OUTPUT_DIR, entry, "data.yaml")
            if os.path.exists(candidate):
                return candidate
    # Fallback: dataset/data.yaml
    direct = os.path.join(OUTPUT_DIR, "data.yaml")
    if os.path.exists(direct):
        return direct
    return None


def ensure_absolute_paths(yaml_path):
    """Rewrite any relative image paths in data.yaml to absolute paths.

    Roboflow sometimes writes relative paths that only work from inside
    the subdirectory. This ensures YOLO can find images from any CWD.
    """
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    changed = False
    for key in ("train", "val", "test"):
        if key not in cfg:
            continue
        raw = cfg[key]
        if raw and not os.path.isabs(raw):
            abs_path = os.path.normpath(os.path.join(yaml_dir, raw))
            cfg[key] = abs_path
            changed = True

    if changed:
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"  Rewrote relative paths to absolute in: {yaml_path}")

    # Verify the train path exists
    train_path = cfg.get("train", "")
    if train_path and not os.path.isdir(train_path):
        raise FileNotFoundError(
            f"Training images directory not found: {train_path}\n"
            "Please re-run download_dataset.py to fetch the dataset."
        )


def train():
    """Load YOLOv8n and train on the surgical instruments dataset."""
    print("=" * 60)
    print("YOLOv8n Training — Surgical Instrument Detector")
    print("=" * 60)

    # Locate data.yaml (handles Roboflow subdirectory layout)
    data_yaml = find_data_yaml()
    if not data_yaml:
        raise FileNotFoundError(
            f"data.yaml not found inside '{OUTPUT_DIR}'.\n"
            "Please run download_dataset.py first."
        )
    print(f"\nUsing dataset config: {os.path.abspath(data_yaml)}")
    ensure_absolute_paths(data_yaml)

    # Load the base YOLOv8n model (downloads weights on first run)
    print(f"\n[1/2] Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # Start training
    print(f"[2/2] Starting training for {EPOCHS} epochs...")
    print(f"      Dataset  : {os.path.abspath(data_yaml)}")
    print(f"      Image sz : {IMAGE_SIZE}px  |  Batch: {BATCH_SIZE}")
    print()

    results = model.train(
        data=data_yaml,
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
