"""
serial_communication.py
------------------------
Sends calculated sterilization durations to hardware controllers
via serial (UART) communication:

  • Arduino Uno  → receives T_UV   (UV lamp controller)
  • ESP32        → receives T_spray (spray nozzle controller)

Usage:
    python serial_communication.py --uv 65 --spray 29

    or import and call send_to_hardware() directly from other scripts.
"""

import argparse
import time
import serial

# ─── SERIAL PORT CONFIGURATION ───────────────────────────────────────────────
# Change these port names to match your system:
#   Windows : "COM3", "COM4", ...
#   Linux   : "/dev/ttyUSB0", "/dev/ttyACM0", ...
#   macOS   : "/dev/cu.usbserial-XXXX", ...

ARDUINO_PORT  = "COM3"    # <-- Change to your Arduino Uno port
ESP32_PORT    = "COM4"    # <-- Change to your ESP32 port
BAUD_RATE     = 9600      # Must match the baud rate set on each device
TIMEOUT_S     = 2         # Serial read timeout in seconds
POST_SEND_DELAY = 0.1     # Short delay after each send (seconds)
# ──────────────────────────────────────────────────────────────────────────────


def send_to_hardware(t_uv: float, t_spray: float) -> bool:
    """
    Open serial connections to Arduino Uno and ESP32,
    transmit the duration values, then close both connections.

    Parameters
    ----------
    t_uv    : float  UV exposure duration in seconds.
    t_spray : float  Spray duration in seconds.

    Returns
    -------
    bool  True if both transmissions succeeded, False otherwise.
    """
    print("\n─── Serial Transmission ────────────────────────────────────")
    success = True

    # ── Send T_UV to Arduino Uno ──────────────────────────────────────────────
    try:
        print(f"  Opening Arduino port  : {ARDUINO_PORT} @ {BAUD_RATE} baud")
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=TIMEOUT_S)
        time.sleep(2)  # Allow Arduino to reset after serial connection opens

        message_uv = f"{t_uv}\n"
        arduino.write(message_uv.encode("utf-8"))
        time.sleep(POST_SEND_DELAY)

        print(f"  Sent to Arduino       : T_UV = {t_uv} s  ✓")
        arduino.close()

    except serial.SerialException as e:
        print(f"  [ERROR] Arduino ({ARDUINO_PORT}) : {e}")
        success = False

    # ── Send T_spray to ESP32 ─────────────────────────────────────────────────
    try:
        print(f"  Opening ESP32 port    : {ESP32_PORT} @ {BAUD_RATE} baud")
        esp32 = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=TIMEOUT_S)
        time.sleep(2)  # Allow ESP32 to stabilise

        message_spray = f"{t_spray}\n"
        esp32.write(message_spray.encode("utf-8"))
        time.sleep(POST_SEND_DELAY)

        print(f"  Sent to ESP32         : T_spray = {t_spray} s  ✓")
        esp32.close()

    except serial.SerialException as e:
        print(f"  [ERROR] ESP32 ({ESP32_PORT}) : {e}")
        success = False

    # ── Summary ───────────────────────────────────────────────────────────────
    if success:
        print("  All transmissions successful.")
    else:
        print("  One or more transmissions failed. Check port names and connections.")
    print("───────────────────────────────────────────────────────────\n")

    return success


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send sterilization durations to Arduino and ESP32 via serial."
    )
    parser.add_argument("--uv",    type=float, required=True, help="T_UV value in seconds.")
    parser.add_argument("--spray", type=float, required=True, help="T_spray value in seconds.")
    args = parser.parse_args()

    send_to_hardware(args.uv, args.spray)
