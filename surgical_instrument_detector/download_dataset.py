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

import io
import os
import shutil
import time
import zipfile

import requests
import yaml

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
API_KEY        = "YOUR_API_KEY"      # <-- Replace with your Roboflow API key
WORKSPACE_NAME = "WORKSPACE_NAME"    # <-- Replace with your workspace slug
PROJECT_NAME   = "PROJECT_NAME"      # <-- Replace with your project slug
VERSION_NUMBER = 1                   # Dataset version to download
DATASET_FORMAT = "yolov8"
# Use absolute path so Roboflow always downloads to the right place
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
# ──────────────────────────────────────────────────────────────────────────────

_API_BASE = "https://api.roboflow.com"


def _get_export_link():
    """Call the Roboflow REST API and return the download URL for the zip.

    If the export has not been generated yet this function triggers generation
    and polls until it is ready (up to ~5 minutes).
    """
    url = (
        f"{_API_BASE}/{WORKSPACE_NAME}/{PROJECT_NAME}"
        f"/{VERSION_NUMBER}/{DATASET_FORMAT}?api_key={API_KEY}"
    )
    print(f"  GET {url.replace(API_KEY, '***')}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    info = resp.json()

    link = info.get("export", {}).get("link")
    if link:
        return link

    # Export not yet generated — trigger it
    print("  Export not ready. Triggering generation...")
    gen_url = (
        f"{_API_BASE}/{WORKSPACE_NAME}/{PROJECT_NAME}"
        f"/{VERSION_NUMBER}/{DATASET_FORMAT}/export?api_key={API_KEY}"
    )
    requests.post(gen_url, timeout=30)

    # Poll until ready (max 5 minutes)
    for attempt in range(30):
        time.sleep(10)
        print(f"  Waiting for export... ({(attempt + 1) * 10}s)")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        info = resp.json()
        link = info.get("export", {}).get("link")
        if link:
            return link

    raise RuntimeError(
        "Roboflow export did not become ready after 5 minutes. "
        "Try clicking 'Download Dataset' in the Roboflow UI to trigger it manually."
    )


def download_dataset():
    """Download the dataset zip directly via the Roboflow REST API and extract it."""
    print("=" * 60)
    print("Surgical Instrument Dataset Downloader")
    print("=" * 60)

    # Clean slate — remove any previous download
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing dataset directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # ── Step 1: get the download URL ──────────────────────────────────────────
    print("\n[1/3] Fetching export URL from Roboflow API...")
    download_url = _get_export_link()
    print(f"  Download URL obtained.")

    # ── Step 2: download the zip ──────────────────────────────────────────────
    print("\n[2/3] Downloading dataset zip (this may take a minute)...")
    zip_resp = requests.get(download_url, timeout=300, stream=True)
    zip_resp.raise_for_status()

    raw = zip_resp.content
    size_kb = len(raw) / 1024
    print(f"  Downloaded {size_kb:.1f} KB")

    if size_kb < 5:
        print("  WARNING: zip is very small — it may only contain data.yaml.")

    # ── Step 3: inspect and extract ───────────────────────────────────────────
    print(f"\n[3/3] Extracting to {OUTPUT_DIR} ...")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        print(f"  Zip contains {len(names)} entries")
        # Show a sample of what's inside
        sample = names[:10]
        for n in sample:
            print(f"    {n}")
        if len(names) > 10:
            print(f"    ... ({len(names) - 10} more)")
        zf.extractall(OUTPUT_DIR)

    print("\nExtraction complete. Contents of dataset directory:")
    _list_dataset_contents(OUTPUT_DIR)
    return OUTPUT_DIR


def _list_dataset_contents(directory):
    """Print the top-level contents of the dataset directory."""
    if not os.path.isdir(directory):
        print(f"  ERROR: directory does not exist: {directory}")
        return
    entries = os.listdir(directory)
    if not entries:
        print("  WARNING: directory is empty after extraction.")
        return
    for entry in sorted(entries):
        full = os.path.join(directory, entry)
        if os.path.isdir(full):
            n = sum(1 for f in os.listdir(full) if os.path.isfile(os.path.join(full, f)))
            print(f"  [dir]  {entry}/  ({n} files)")
        else:
            print(f"  [file] {entry}")


def find_roboflow_dir():
    """Find the directory that contains data.yaml (flat or one level deep)."""
    # Flat layout: data.yaml directly in OUTPUT_DIR
    if os.path.exists(os.path.join(OUTPUT_DIR, "data.yaml")):
        return OUTPUT_DIR
    # Nested layout: OUTPUT_DIR/{project}-{version}/data.yaml
    if os.path.isdir(OUTPUT_DIR):
        for entry in sorted(os.listdir(OUTPUT_DIR)):
            candidate = os.path.join(OUTPUT_DIR, entry)
            if os.path.isdir(candidate) and os.path.exists(
                os.path.join(candidate, "data.yaml")
            ):
                return candidate
    return None


def fix_data_yaml():
    """Rewrite data.yaml so all image paths are absolute."""
    roboflow_dir = find_roboflow_dir()
    if not roboflow_dir:
        print("  WARNING: Could not find data.yaml to fix paths.")
        return None

    yaml_path = os.path.join(roboflow_dir, "data.yaml")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    changed = False
    for key in ("train", "val", "test"):
        if key not in cfg:
            continue
        raw = cfg[key]
        if raw and not os.path.isabs(raw):
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
    direct = os.path.join(roboflow_dir, split, "images")
    if os.path.isdir(direct):
        return direct
    return None


def count_images():
    """Count and display the number of images per split."""
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
