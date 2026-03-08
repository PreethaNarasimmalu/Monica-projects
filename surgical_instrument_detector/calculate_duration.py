"""
calculate_duration.py
----------------------
Calculates UV sterilization duration and spray duration based on the
number of detected surgical instruments.

Formulas
--------
  T_UV    = T_BASE_UV   + (N × k_UV)
  T_spray = T_BASE_SPRAY + (N × k_spray)

Usage:
    python calculate_duration.py --count <N>

    or import and call calculate_durations() directly from other scripts.
"""

import argparse

# ─── CALIBRATION CONSTANTS ────────────────────────────────────────────────────
# Adjust these values after experimental testing with real equipment.

T_BASE_UV    = 40    # Base UV exposure time in seconds (always applied)
k_UV         = 5     # Additional UV seconds per instrument  <-- tune this

T_BASE_SPRAY = 20    # Base spray duration in seconds (always applied)
k_spray      = 3     # Additional spray seconds per instrument  <-- tune this
# ──────────────────────────────────────────────────────────────────────────────


def calculate_durations(instrument_count: int) -> dict:
    """
    Calculate sterilization durations from instrument count.

    Parameters
    ----------
    instrument_count : int
        Number of surgical instruments detected (N).

    Returns
    -------
    dict with keys:
        'N'       – instrument count
        'T_UV'    – UV exposure duration (seconds)
        'T_spray' – spray duration (seconds)
    """
    if instrument_count < 0:
        raise ValueError("Instrument count cannot be negative.")

    T_UV    = T_BASE_UV    + (instrument_count * k_UV)
    T_spray = T_BASE_SPRAY + (instrument_count * k_spray)

    print("\n─── Sterilization Duration Calculation ────────────────────")
    print(f"  Instrument count (N)  : {instrument_count}")
    print()
    print(f"  UV Exposure Duration")
    print(f"    Formula  : T_UV = {T_BASE_UV} + ({instrument_count} × {k_UV})")
    print(f"    T_UV     = {T_UV} seconds")
    print()
    print(f"  Spray Duration")
    print(f"    Formula  : T_spray = {T_BASE_SPRAY} + ({instrument_count} × {k_spray})")
    print(f"    T_spray  = {T_spray} seconds")
    print("───────────────────────────────────────────────────────────\n")

    return {"N": instrument_count, "T_UV": T_UV, "T_spray": T_spray}


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate UV and spray durations from instrument count."
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        required=True,
        help="Number of detected surgical instruments."
    )
    args = parser.parse_args()

    durations = calculate_durations(args.count)
    print(f"T_UV    = {durations['T_UV']} s")
    print(f"T_spray = {durations['T_spray']} s")
