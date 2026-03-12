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
from roboflow import Roboflow

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
API_KEY        = "YOUR_API_KEY"      # <-- Replace with your Roboflow API key
WORKSPACE_NAME = "WORKSPACE_NAME"    # <-- Replace with your workspace slug
PROJECT_NAME   = "PROJECT_NAME"      # <-- Replace with your project slug
VERSION_NUMBER = 1                   # Dataset version to download
DATASET_FORMAT = "yolov8"
OUTPUT_DIR     = "dataset"
# ──────────────────────────────────────────────────────────────────────────────


def download_dataset():
    """Connect to Roboflow and download the specified dataset version."""
    print("=" * 60)
    print("Surgical Instrument Dataset Downloader")
    print("=" * 60)

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
        location=OUTPUT_DIR
    )

    print(f"\nDataset downloaded to: {os.path.abspath(OUTPUT_DIR)}")
    return dataset


def find_split_dir(split):
    """Find the images directory for a given split (train/valid/test).

    Roboflow sometimes downloads into a subdirectory named after the project,
    so we search one level deep before giving up.
    """
    # Direct path: dataset/train/images
    direct = os.path.join(OUTPUT_DIR, split, "images")
    if os.path.isdir(direct):
        return direct

    # Subdirectory path: dataset/<project-version>/train/images
    if os.path.isdir(OUTPUT_DIR):
        for entry in os.listdir(OUTPUT_DIR):
            candidate = os.path.join(OUTPUT_DIR, entry, split, "images")
            if os.path.isdir(candidate):
                return candidate

    return None


def count_images():
    """Count and display the number of training and validation images."""
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")

    def count_in(directory):
        if not directory:
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
    count_images()
    print("Dataset download complete. You can now run train_model.py")
