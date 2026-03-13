"""
demo_detect.py
--------------
Demo mode: detects BOTH surgical instruments (trained model) AND
everyday cutlery — fork, knife, spoon — using YOLOv8's pretrained
COCO weights (no retraining required).

Surgical instruments are drawn with GREEN boxes.
Cutlery items are drawn with BLUE boxes.

Usage:
    # Upload / single image
    python demo_detect.py --image path/to/image.jpg

    # Live camera  (press SPACE to capture, Q to quit)
    python demo_detect.py --camera
    python demo_detect.py --camera 1          # use camera index 1
"""

import argparse
import os
import platform
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SURGICAL_MODEL_PATH = os.path.join(
    "runs", "detect", "surgical_instrument_detector", "weights", "best.pt"
)
COCO_MODEL_PATH = "yolov8n.pt"   # Downloaded automatically on first run

SURGICAL_CONF_THRESHOLD = 0.5   # Confidence for surgical model
COCO_CONF_THRESHOLD     = 0.35  # Lower threshold catches items COCO would miss

# COCO class IDs for cutlery
CUTLERY_CLASS_IDS = {42: "Fork", 43: "Knife", 44: "Spoon"}

# If a surgical box overlaps a COCO box by more than this, suppress the
# surgical detection (it's just the model mislabelling a utensil).
CROSS_MODEL_IOU_THRESHOLD = 0.30

# Drawing colours (BGR)
SURGICAL_COLOUR = (0, 200, 0)    # Green
CUTLERY_COLOUR  = (255, 100, 0)  # Blue
# ──────────────────────────────────────────────────────────────────────────────

# Load models once at module level (avoids reloading on every frame)
_surgical_model = None
_coco_model     = None


def _load_models():
    global _surgical_model, _coco_model
    if _coco_model is None:
        print(f"Loading COCO model: {COCO_MODEL_PATH}")
        _coco_model = YOLO(COCO_MODEL_PATH)
    if _surgical_model is None:
        if os.path.exists(SURGICAL_MODEL_PATH):
            print(f"Loading surgical model: {SURGICAL_MODEL_PATH}")
            _surgical_model = YOLO(SURGICAL_MODEL_PATH)
        else:
            print(
                f"[WARNING] Surgical model not found at '{SURGICAL_MODEL_PATH}'.\n"
                "          Only cutlery detection will run.\n"
                "          Train the model first with train_model.py to also detect instruments."
            )


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Return Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _run_detection(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Run detection on a numpy BGR image (OpenCV frame).

    Returns
    -------
    annotated : np.ndarray   Image with bounding boxes drawn on it.
    results   : dict         Detection counts and cutlery item list.
    """
    _load_models()
    annotated = image.copy()

    # ── 1. Cutlery (COCO) ─────────────────────────────────────────────────────
    c_res   = _coco_model.predict(source=image, conf=COCO_CONF_THRESHOLD,
                                   save=False, verbose=False,
                                   classes=list(CUTLERY_CLASS_IDS.keys()))
    c_boxes      = c_res[0].boxes
    coco_xyxy    = c_boxes.xyxy.cpu().numpy()   # keep for IoU filtering
    cutlery_items = []
    cutlery_count = len(c_boxes)

    for box, cls_id, conf in zip(coco_xyxy,
                                  c_boxes.cls.cpu().numpy(),
                                  c_boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        name = CUTLERY_CLASS_IDS[int(cls_id)]
        cutlery_items.append(name)
        _draw_box(annotated, x1, y1, x2, y2, f"{name} {conf:.2f}", CUTLERY_COLOUR)

    # ── 2. Surgical instruments (suppress boxes that overlap COCO hits) ────────
    surgical_count = 0
    if _surgical_model is not None:
        s_res   = _surgical_model.predict(source=image, conf=SURGICAL_CONF_THRESHOLD,
                                          save=False, verbose=False)
        s_boxes = s_res[0].boxes
        s_names = _surgical_model.names

        for box, cls_id, conf in zip(s_boxes.xyxy.cpu().numpy(),
                                     s_boxes.cls.cpu().numpy(),
                                     s_boxes.conf.cpu().numpy()):
            # Skip if this box significantly overlaps any COCO detection
            # (the surgical model is mis-labelling a utensil as an instrument)
            overlaps_cutlery = any(
                _iou(box, coco_box) > CROSS_MODEL_IOU_THRESHOLD
                for coco_box in coco_xyxy
            )
            if overlaps_cutlery:
                continue

            x1, y1, x2, y2 = map(int, box)
            _draw_box(annotated, x1, y1, x2, y2,
                      f"{s_names[int(cls_id)]} {conf:.2f}", SURGICAL_COLOUR)
            surgical_count += 1

    # ── 3. Summary overlay ────────────────────────────────────────────────────
    total = surgical_count + cutlery_count
    cv2.putText(annotated,
                f"Total Instruments : {total}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated, {
        "surgical_count": surgical_count,
        "cutlery_count":  cutlery_count,
        "total_count":    total,
        "cutlery_items":  cutlery_items,
    }


# ─── MODE 1: Single image (upload) ────────────────────────────────────────────

def demo_detect_image(image_path: str, show: bool = True) -> dict:
    """Run detection on a saved image file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: '{image_path}'")

    image = cv2.imread(image_path)
    annotated, results = _run_detection(image)

    _print_summary(results)

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_demo{ext}"
    cv2.imwrite(output_path, annotated)
    print(f"Annotated image saved: {output_path}")

    if show:
        _open_image(output_path)

    return results


# ─── MODE 2: Live camera ───────────────────────────────────────────────────────

def demo_detect_camera(camera_index: int = 0) -> None:
    """
    Open a live camera feed (terminal-controlled, no GUI window required).

    Controls  (type in this terminal, then press Enter)
    --------
    Enter  – capture current frame and run detection
    q      – quit
    """
    _load_models()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}.")
        return

    print(f"\nCamera {camera_index} opened.")
    print("  Press Enter       →  capture frame & run detection")
    print("  Type q + Enter    →  quit\n")

    capture_num = 0

    while True:
        try:
            user_input = input("Ready — press Enter to capture (or type q to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nQuitting.")
            break

        if user_input == "q":
            print("Quitting camera mode.")
            break

        # Flush camera buffer — read a few frames to get the freshest image
        for _ in range(5):
            cap.grab()
        ret, frame = cap.retrieve()

        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        capture_num += 1
        print(f"\n[Capture {capture_num}] Running detection...")
        annotated, results = _run_detection(frame)
        _print_summary(results)

        output_path = os.path.abspath(f"demo_capture_{capture_num:03d}.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"Saved: {output_path}")
        _open_image(output_path)

    cap.release()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _open_image(path: str):
    """Open an image file with the OS default viewer."""
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        print(f"Opened result in default image viewer.")
    except Exception as e:
        print(f"[INFO] Could not open viewer automatically: {e}")
        print(f"       Open manually: {path}")


def _draw_box(image, x1, y1, x2, y2, label, colour):
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw, y1), colour, -1)
    cv2.putText(image, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def _print_summary(results: dict):
    print("\n─── Summary ───────────────────────────────────────────────")
    if results["cutlery_items"]:
        from collections import Counter
        for name, n in Counter(results["cutlery_items"]).items():
            print(f"  {name:<20s}: {n}")
    if results["surgical_count"]:
        print(f"  Surgical instruments: {results['surgical_count']}")
    print(f"  ─────────────────────────────")
    print(f"  Total Instruments : {results['total_count']}")
    print("───────────────────────────────────────────────────────────\n")


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo: detect surgical instruments + cutlery (fork/knife/spoon)."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i",
                       help="Path to an image file (upload mode).")
    group.add_argument("--camera", "-c",
                       nargs="?", const=0, type=int, metavar="INDEX",
                       help="Live camera mode. Optionally specify camera index (default: 0).")

    parser.add_argument("--no-display", action="store_true",
                        help="Image mode only: skip the display window.")

    args = parser.parse_args()

    if args.image:
        results = demo_detect_image(args.image, show=not args.no_display)
    else:
        demo_detect_camera(camera_index=args.camera)
