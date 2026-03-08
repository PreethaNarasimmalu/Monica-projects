"""
detect_and_count.py
--------------------
Loads the trained YOLOv8n model, runs inference on a given image,
counts all detected surgical instruments, draws bounding boxes,
and displays the annotated result.

Usage:
    python detect_and_count.py --image path/to/image.jpg

    or import and call detect_and_count() directly from other scripts.
"""

import argparse
import os
import cv2
from ultralytics import YOLO

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("runs", "detect", "surgical_instrument_detector",
                               "weights", "best.pt")  # Trained model weights
CONF_THRESHOLD = 0.5    # Minimum confidence to count a detection
SHOW_WINDOW    = True   # Set False when running headless (no display)
# ──────────────────────────────────────────────────────────────────────────────


def detect_and_count(image_path: str, show: bool = SHOW_WINDOW) -> int:
    """
    Run YOLOv8 detection on the given image and return instrument count.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    show : bool
        Whether to open a window displaying the annotated image.

    Returns
    -------
    int
        Number of detected instruments (N).
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'.\n"
            "Run train_model.py first."
        )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: '{image_path}'")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"Running detection on: {image_path}")
    results = model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        save=False,      # We handle saving/display ourselves
        verbose=False
    )

    # ── Parse detections ──────────────────────────────────────────────────────
    result     = results[0]                  # Single image → first result
    boxes      = result.boxes
    class_ids  = boxes.cls.cpu().numpy()     # Detected class indices
    confidences= boxes.conf.cpu().numpy()    # Confidence scores
    xyxy       = boxes.xyxy.cpu().numpy()    # Bounding boxes [x1,y1,x2,y2]
    names      = model.names                 # {id: class_name}

    instrument_count = len(boxes)

    # ── Print detection details ───────────────────────────────────────────────
    print("\n─── Detections ────────────────────────────────────────────")
    if instrument_count == 0:
        print("  No instruments detected above confidence threshold.")
    else:
        for i, (cls_id, conf) in enumerate(zip(class_ids, confidences)):
            class_name = names[int(cls_id)]
            print(f"  [{i+1}] {class_name:<30s}  confidence: {conf:.2f}")
    print(f"\n  Total instruments detected: {instrument_count}")
    print("───────────────────────────────────────────────────────────\n")

    # ── Draw bounding boxes on the image ──────────────────────────────────────
    image = cv2.imread(image_path)

    for i, (box, cls_id, conf) in enumerate(zip(xyxy, class_ids, confidences)):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls_id)]} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 8), (x1 + text_w, y1), (0, 255, 0), -1)

        # Draw label text
        cv2.putText(image, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Overlay total count ───────────────────────────────────────────────────
    count_text = f"Instruments detected: {instrument_count}"
    cv2.putText(image, count_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # ── Display / save annotated image ────────────────────────────────────────
    if show:
        cv2.imshow("Surgical Instrument Detection", image)
        print("Press any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save annotated image alongside original
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_detected{ext}"
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved: {output_path}")

    return instrument_count


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and count surgical instruments in an image."
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input image file."
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the image display window (headless mode)."
    )
    args = parser.parse_args()

    count = detect_and_count(args.image, show=not args.no_display)
    print(f"Instrument count (N) = {count}")
