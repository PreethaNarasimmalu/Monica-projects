"""
demo_detect.py
--------------
Demo mode: detects BOTH surgical instruments (trained model) AND
everyday cutlery — fork, knife, spoon — using YOLOv8's pretrained
COCO weights (no retraining required).

Surgical instruments are drawn with GREEN boxes.
Cutlery items are drawn with BLUE boxes.

Usage:
    python demo_detect.py --image path/to/image.jpg
    python demo_detect.py --image path/to/image.jpg --no-display
"""

import argparse
import os
import cv2
from ultralytics import YOLO

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SURGICAL_MODEL_PATH = os.path.join(
    "runs", "detect", "surgical_instrument_detector", "weights", "best.pt"
)
COCO_MODEL_PATH = "yolov8n.pt"   # Downloaded automatically on first run

CONF_THRESHOLD = 0.5

# COCO class IDs for cutlery
CUTLERY_CLASS_IDS = {42: "Fork", 43: "Knife", 44: "Spoon"}

# Drawing colours  (BGR)
SURGICAL_COLOUR = (0, 200, 0)    # Green
CUTLERY_COLOUR  = (255, 100, 0)  # Blue
# ──────────────────────────────────────────────────────────────────────────────


def demo_detect(image_path: str, show: bool = True) -> dict:
    """
    Run detection for both surgical instruments and cutlery on one image.

    Returns
    -------
    dict with keys:
        'surgical_count'  – number of surgical instruments detected
        'cutlery_count'   – number of cutlery items detected
        'total_count'     – combined total
        'cutlery_items'   – list of detected cutlery names
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: '{image_path}'")

    image = cv2.imread(image_path)

    # ── 1. Surgical instruments (trained model) ───────────────────────────────
    surgical_count = 0
    if os.path.exists(SURGICAL_MODEL_PATH):
        print(f"Loading surgical model: {SURGICAL_MODEL_PATH}")
        surgical_model = YOLO(SURGICAL_MODEL_PATH)
        s_results = surgical_model.predict(
            source=image_path, conf=CONF_THRESHOLD, save=False, verbose=False
        )
        s_boxes      = s_results[0].boxes
        s_class_ids  = s_boxes.cls.cpu().numpy()
        s_confidences= s_boxes.conf.cpu().numpy()
        s_xyxy       = s_boxes.xyxy.cpu().numpy()
        s_names      = surgical_model.names
        surgical_count = len(s_boxes)

        print("\n─── Surgical Instruments ──────────────────────────────────")
        if surgical_count == 0:
            print("  None detected.")
        for i, (box, cls_id, conf) in enumerate(zip(s_xyxy, s_class_ids, s_confidences)):
            x1, y1, x2, y2 = map(int, box)
            label = f"{s_names[int(cls_id)]} {conf:.2f}"
            print(f"  [{i+1}] {s_names[int(cls_id)]:<30s}  conf: {conf:.2f}")
            _draw_box(image, x1, y1, x2, y2, label, SURGICAL_COLOUR)
    else:
        print(
            f"[WARNING] Surgical model not found at '{SURGICAL_MODEL_PATH}'.\n"
            "          Only cutlery detection will run.\n"
            "          Train the model first with train_model.py to also detect instruments."
        )

    # ── 2. Cutlery — fork / knife / spoon (COCO pretrained) ──────────────────
    print("\n─── Cutlery (fork / knife / spoon) ────────────────────────")
    print(f"Loading COCO model: {COCO_MODEL_PATH}")
    coco_model = YOLO(COCO_MODEL_PATH)
    c_results = coco_model.predict(
        source=image_path, conf=CONF_THRESHOLD, save=False, verbose=False,
        classes=list(CUTLERY_CLASS_IDS.keys())   # Only detect fork/knife/spoon
    )
    c_boxes       = c_results[0].boxes
    c_class_ids   = c_boxes.cls.cpu().numpy()
    c_confidences = c_boxes.conf.cpu().numpy()
    c_xyxy        = c_boxes.xyxy.cpu().numpy()

    cutlery_items = []
    cutlery_count = len(c_boxes)

    if cutlery_count == 0:
        print("  None detected.")
    for i, (box, cls_id, conf) in enumerate(zip(c_xyxy, c_class_ids, c_confidences)):
        x1, y1, x2, y2 = map(int, box)
        name  = CUTLERY_CLASS_IDS[int(cls_id)]
        label = f"{name} {conf:.2f}"
        print(f"  [{i+1}] {name:<30s}  conf: {conf:.2f}")
        cutlery_items.append(name)
        _draw_box(image, x1, y1, x2, y2, label, CUTLERY_COLOUR)

    # ── 3. Summary overlay ────────────────────────────────────────────────────
    total_count = surgical_count + cutlery_count

    print("\n─── Summary ───────────────────────────────────────────────")
    print(f"  Surgical instruments : {surgical_count}")
    print(f"  Cutlery items        : {cutlery_count}")
    print(f"  TOTAL                : {total_count}")
    print("───────────────────────────────────────────────────────────\n")

    cv2.putText(image, f"Instruments: {surgical_count}  Cutlery: {cutlery_count}  Total: {total_count}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Legend
    cv2.putText(image, "Green = Surgical   Blue = Cutlery",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # ── 4. Display / save ─────────────────────────────────────────────────────
    if show:
        cv2.imshow("Demo Detection — Instruments & Cutlery", image)
        print("Press any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_demo{ext}"
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved: {output_path}")

    return {
        "surgical_count": surgical_count,
        "cutlery_count":  cutlery_count,
        "total_count":    total_count,
        "cutlery_items":  cutlery_items,
    }


# ─── Helper ───────────────────────────────────────────────────────────────────

def _draw_box(image, x1, y1, x2, y2, label, colour):
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw, y1), colour, -1)
    cv2.putText(image, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo: detect surgical instruments + cutlery (fork/knife/spoon)."
    )
    parser.add_argument("--image", "-i", required=True,
                        help="Path to the input image file.")
    parser.add_argument("--no-display", action="store_true",
                        help="Suppress the display window (headless mode).")
    args = parser.parse_args()

    results = demo_detect(args.image, show=not args.no_display)
    print(f"Total objects counted: {results['total_count']}")
