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


def count_images():
    """Count and display the number of training and validation images."""
    train_img_dir = os.path.join(OUTPUT_DIR, "train", "images")
    valid_img_dir = os.path.join(OUTPUT_DIR, "valid", "images")

    train_count = len([
        f for f in os.listdir(train_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]) if os.path.isdir(train_img_dir) else 0

    valid_count = len([
        f for f in os.listdir(valid_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]) if os.path.isdir(valid_img_dir) else 0

    print("\n─── Dataset Summary ───────────────────────────────────────")
    print(f"  Training images   : {train_count}")
    print(f"  Validation images : {valid_count}")
    print(f"  Total images      : {train_count + valid_count}")
    print("───────────────────────────────────────────────────────────\n")

    return train_count, valid_count


if __name__ == "__main__":
    download_dataset()
    count_images()
    print("Dataset download complete. You can now run train_model.py")
