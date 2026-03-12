"""
download_dataset.py
-------------------
Downloads a surgical instruments dataset from Roboflow in YOLOv8 format
and organises it into the required folder structure.

Usage:
    python download_dataset.py

Before running:
    1. Replace YOUR_API_KEY with your Roboflow API key.
    2. Replace WORKSPACE_NAME and PROJECT_NAME with the correct values
       from the Roboflow project URL.
    3. Adjust VERSION_NUMBER if needed.
"""

import os
import shutil
import yaml
from roboflow import Roboflow

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
API_KEY        = "YOUR_API_KEY"      # <-- Replace with your Roboflow API key
WORKSPACE_NAME = "WORKSPACE_NAME"    # <-- Replace with your workspace slug
PROJECT_NAME   = "PROJECT_NAME"      # <-- Replace with your project slug
VERSION_NUMBER = 1                   # Dataset version to download
DATASET_FORMAT = "yolov8"
# Use absolute path so Roboflow always downloads to the right place
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
# ──────────────────────────────────────────────────────────────────────────────


def download_dataset():
    """Connect to Roboflow and download the specified dataset version."""
    print("=" * 60)
    print("Surgical Instrument Dataset Downloader")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Authenticate with Roboflow
    print(f"\n[1/3] Authenticating with Roboflow...")
    rf = Roboflow(api_key=API_KEY)

    # Access the project
    print(f"[2/3] Accessing project '{PROJECT_NAME}' in workspace '{WORKSPACE_NAME}'...")
    project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)

    # Download the dataset in YOLOv8 format
    print(f"[3/3] Downloading version {VERSION_NUMBER} in '{DATASET_FORMAT}' format...")
    dataset = project.version(VERSION_NUMBER).download(
        model_format=DATASET_FORMAT,
        location=OUTPUT_DIR,
        overwrite=True
    )

    print(f"\nDownload complete. Checking contents of: {OUTPUT_DIR}")
    _list_dataset_contents(OUTPUT_DIR)
    return dataset


def _list_dataset_contents(directory):
    """Print the top-level contents of the dataset directory."""
    if not os.path.isdir(directory):
        print(f"  ERROR: directory does not exist: {directory}")
        return
    entries = os.listdir(directory)
    if not entries:
        print("  WARNING: directory is empty after download.")
        return
    for entry in sorted(entries):
        full = os.path.join(directory, entry)
        if os.path.isdir(full):
            n = sum(1 for f in os.listdir(full) if os.path.isfile(os.path.join(full, f)))
            print(f"  [dir]  {entry}/  ({n} files)")
        else:
            print(f"  [file] {entry}")


def find_roboflow_dir():
    """Find the subdirectory that Roboflow extracted the dataset into.

    Roboflow downloads to {OUTPUT_DIR}/{project}-{version}/ by default.
    Falls back to OUTPUT_DIR itself if train/valid exist there directly.
    """
    # Check one level deep for a subdirectory containing data.yaml
    if os.path.isdir(OUTPUT_DIR):
        for entry in sorted(os.listdir(OUTPUT_DIR)):
            candidate = os.path.join(OUTPUT_DIR, entry)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "data.yaml")):
                return candidate
    # Fallback: flat layout directly in OUTPUT_DIR
    if os.path.exists(os.path.join(OUTPUT_DIR, "data.yaml")):
        return OUTPUT_DIR
    return None


def fix_data_yaml():
    """Rewrite data.yaml so all image paths are absolute.

    Roboflow sometimes writes relative paths (../train/images) or paths
    that are only valid from inside the subdirectory. This rewrites them
    to absolute paths so they work regardless of working directory.
    """
    roboflow_dir = find_roboflow_dir()
    if not roboflow_dir:
        print("  WARNING: Could not find Roboflow dataset directory to fix data.yaml.")
        return None

    yaml_path = os.path.join(roboflow_dir, "data.yaml")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    changed = False
    for key in ("train", "val", "test"):
        if key not in cfg:
            continue
        raw = cfg[key]
        if not os.path.isabs(raw):
            # Resolve relative to the directory containing data.yaml
            abs_path = os.path.normpath(os.path.join(roboflow_dir, raw))
            cfg[key] = abs_path
            changed = True

    if changed:
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"  Fixed paths in: {yaml_path}")

    return yaml_path


def find_split_dir(split):
    """Find the images directory for a given split (train/valid/test)."""
    roboflow_dir = find_roboflow_dir()
    if not roboflow_dir:
        return None
    # Direct: <roboflow_dir>/train/images
    direct = os.path.join(roboflow_dir, split, "images")
    if os.path.isdir(direct):
        return direct
    return None


def count_images():
    """Count and display the number of training and validation images."""
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")

    def count_in(directory):
        if not directory or not os.path.isdir(directory):
            return 0
        return len([f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTS)])

    train_dir = find_split_dir("train")
    valid_dir = find_split_dir("valid")
    test_dir  = find_split_dir("test")

    train_count = count_in(train_dir)
    valid_count = count_in(valid_dir)
    test_count  = count_in(test_dir)

    print("\n─── Dataset Summary ───────────────────────────────────────")
    print(f"  Training images   : {train_count}  ({train_dir or 'not found'})")
    print(f"  Validation images : {valid_count}  ({valid_dir or 'not found'})")
    print(f"  Test images       : {test_count}  ({test_dir  or 'not found'})")
    print(f"  Total images      : {train_count + valid_count + test_count}")
    print("───────────────────────────────────────────────────────────\n")

    return train_count, valid_count


if __name__ == "__main__":
    download_dataset()
    fix_data_yaml()
    count_images()
    print("Dataset download complete. You can now run train_model.py")
