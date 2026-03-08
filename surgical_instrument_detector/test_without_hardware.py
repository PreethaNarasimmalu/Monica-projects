"""
test_without_hardware.py
-------------------------
Validates the AI detection and duration-calculation pipeline
WITHOUT requiring Arduino, ESP32, webcam, or door switch.

Use this script to:
  • Verify that detection works on a test image
  • Check that duration formulas produce correct results
  • Debug issues before connecting hardware

Usage:
    python test_without_hardware.py --image path/to/test_image.jpg

    Optional: override instrument count manually (skip detection)
    python test_without_hardware.py --count 5
"""

import argparse
import os

from detect_and_count   import detect_and_count
from calculate_duration import calculate_durations

# ──────────────────────────────────────────────────────────────────────────────


def test_pipeline(image_path: str = None, manual_count: int = None):
    """
    Run the detection + calculation pipeline without any serial communication.

    Parameters
    ----------
    image_path   : str | None   Path to a test image (used for AI detection).
    manual_count : int | None   Skip detection and use this instrument count.
    """
    print("=" * 60)
    print("Surgical Instrument System — Hardware-Free Test")
    print("=" * 60)

    # ── Step 1: Determine instrument count ────────────────────────────────────
    if manual_count is not None:
        # Manual override — skip model inference
        instrument_count = manual_count
        print(f"\n[Mode] Manual count override: N = {instrument_count}")

    elif image_path is not None:
        # Run AI detection on the provided image
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: '{image_path}'")
            return

        print(f"\n[Mode] AI detection on image: {image_path}")
        instrument_count = detect_and_count(image_path, show=True)

    else:
        print("[ERROR] Provide either --image or --count argument.")
        return

    # ── Step 2: Calculate durations ───────────────────────────────────────────
    print("\n[Calculating durations...]")
    durations = calculate_durations(instrument_count)

    # ── Step 3: Final summary (no serial communication) ───────────────────────
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Instrument count (N) : {instrument_count}")
    print(f"  T_UV                 : {durations['T_UV']} seconds")
    print(f"  T_spray              : {durations['T_spray']} seconds")
    print(f"  Serial transmission  : SKIPPED (hardware-free mode)")
    print("=" * 60)
    print("\nTest complete. No hardware was required.\n")

    return durations


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the detection + calculation pipeline without hardware."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", "-i",
        type=str,
        help="Path to a test image for AI-based instrument detection."
    )
    group.add_argument(
        "--count", "-n",
        type=int,
        help="Manually specify instrument count (skips AI detection)."
    )

    args = parser.parse_args()
    test_pipeline(image_path=args.image, manual_count=args.count)
