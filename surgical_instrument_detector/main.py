"""
main.py
--------
Master script for the Surgical Instrument Sterilization System.

Pipeline (triggered each time the UV chamber door closes):
  1. Monitor door-switch serial signal
  2. Capture image from webcam
  3. Detect & count instruments (detect_and_count.py)
  4. Calculate sterilization durations (calculate_duration.py)
  5. Send durations to Arduino & ESP32 (serial_communication.py)
  6. Print cycle summary, then wait for next trigger

Usage:
    python main.py

Stop the script cleanly at any time with Ctrl+C.
"""

import os
import time
import cv2
import serial

from detect_and_count       import detect_and_count
from calculate_duration     import calculate_durations
from serial_communication   import send_to_hardware

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DOOR_SWITCH_PORT  = "COM5"   # <-- Change to your door-switch serial port
DOOR_SWITCH_BAUD  = 9600     # Baud rate for door switch serial
DOOR_CLOSED_MSG   = "DOOR_CLOSED"  # Expected string from door switch
WEBCAM_INDEX      = 0        # Webcam device index (0 = default camera)
CAPTURE_PATH      = "captured_frame.jpg"   # Temporary image file
POLL_INTERVAL_S   = 0.5      # How often to check for door signal (seconds)
# ──────────────────────────────────────────────────────────────────────────────


def capture_image(webcam_index: int, save_path: str) -> str:
    """
    Capture a single frame from the webcam and save it to disk.

    Parameters
    ----------
    webcam_index : int  OpenCV camera index.
    save_path    : str  File path to save the captured image.

    Returns
    -------
    str  Path of the saved image, or empty string on failure.
    """
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index {webcam_index}).")
        return ""

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Failed to capture frame from webcam.")
        return ""

    cv2.imwrite(save_path, frame)
    print(f"  Image captured and saved: {os.path.abspath(save_path)}")
    return save_path


def run_cycle(image_path: str) -> dict:
    """
    Execute the full detection → calculation → transmission pipeline.

    Returns a summary dict for logging.
    """
    # Step 3 — Detect and count instruments
    print("\n[STEP 3] Detecting instruments...")
    instrument_count = detect_and_count(image_path, show=False)

    # Step 4 — Calculate sterilization durations
    print("[STEP 4] Calculating sterilization durations...")
    durations = calculate_durations(instrument_count)

    # Step 5 — Send to hardware
    print("[STEP 5] Sending to hardware controllers...")
    tx_success = send_to_hardware(durations["T_UV"], durations["T_spray"])

    # Step 6 — Summary
    print("\n" + "=" * 60)
    print("CYCLE SUMMARY")
    print("=" * 60)
    print(f"  Instruments detected : {instrument_count}")
    print(f"  T_UV calculated      : {durations['T_UV']} seconds")
    print(f"  T_spray calculated   : {durations['T_spray']} seconds")
    print(f"  Serial transmission  : {'SUCCESS' if tx_success else 'FAILED'}")
    print("=" * 60)

    return {
        "instrument_count": instrument_count,
        "T_UV": durations["T_UV"],
        "T_spray": durations["T_spray"],
        "tx_success": tx_success,
    }


def monitor_door_switch():
    """
    Open the door-switch serial port and block until a DOOR_CLOSED signal
    is received, then return True.  Returns False on connection error.
    """
    try:
        ser = serial.Serial(DOOR_SWITCH_PORT, DOOR_SWITCH_BAUD, timeout=1)
        print(f"  Monitoring door switch on {DOOR_SWITCH_PORT}...")

        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line == DOOR_CLOSED_MSG:
                print(f"  Door closed signal received: '{line}'")
                ser.close()
                return True
            time.sleep(POLL_INTERVAL_S)

    except serial.SerialException as e:
        print(f"[ERROR] Door switch port ({DOOR_SWITCH_PORT}): {e}")
        return False


def main():
    """Entry point: continuous monitoring loop."""
    print("=" * 60)
    print("Surgical Instrument Sterilization System — RUNNING")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    cycle_number = 0

    try:
        while True:
            print(f"\n[Waiting] Ready for next sterilization cycle...")

            # Step 1 — Wait for door closed signal
            print("\n[STEP 1] Monitoring door switch...")
            door_closed = monitor_door_switch()

            if not door_closed:
                print("  Could not read door switch. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            cycle_number += 1
            print(f"\n{'─'*60}")
            print(f"  CYCLE #{cycle_number} STARTED")
            print(f"{'─'*60}")

            # Step 2 — Capture image
            print("\n[STEP 2] Capturing image from webcam...")
            image_path = capture_image(WEBCAM_INDEX, CAPTURE_PATH)

            if not image_path:
                print("  Image capture failed. Skipping this cycle.")
                continue

            # Run full pipeline
            run_cycle(image_path)

            print(f"\n  Cycle #{cycle_number} complete. Returning to monitoring mode.\n")

    except KeyboardInterrupt:
        print("\n\n[Shutdown] Ctrl+C received. System stopped cleanly.")


if __name__ == "__main__":
    main()
