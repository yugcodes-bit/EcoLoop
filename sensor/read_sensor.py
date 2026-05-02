# ============================================================
# ECO-LOOP — Sensor Reading Module
# sensor/read_sensor.py
#
# TWO MODES:
# 1. SIMULATION MODE (current) — runs on laptop
#    uses fake sensor values to test full pipeline
#
# 2. REAL MODE (when sensor arrives) — runs on Raspberry Pi
#    uncomment the smbus2 section to activate
#    connects to AS7265x via I2C at address 0x49
# ============================================================

import requests
import json
import time
import random
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────
API_URL  = "http://localhost:8000"   # FastAPI server address
MODE     = "simulation"              # change to "real" when sensor arrives

# ── AS7265x channel wavelengths in order ───────────────────────────────────
CHANNELS = [
    410, 435, 460, 485, 510, 535,
    560, 585, 610, 645, 680, 705,
    730, 760, 810, 860, 900, 940
]

# ══════════════════════════════════════════════════════════════════════════
# SIMULATION MODE
# Generates realistic fake sensor readings for testing
# Based on the spectral profiles from our synthetic dataset
# ══════════════════════════════════════════════════════════════════════════

# Base reflectance profiles matching our synthetic data generator
SIMULATED_PROFILES = {
    "fresh": [
        0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
        0.58, 0.60, 0.62, 0.58, 0.30, 0.55,
        0.65, 0.70, 0.80, 0.75, 0.70, 0.65
    ],
    "early_rot": [
        0.25, 0.30, 0.35, 0.38, 0.42, 0.46,
        0.48, 0.50, 0.52, 0.49, 0.38, 0.48,
        0.52, 0.55, 0.58, 0.53, 0.48, 0.43
    ],
    "severe_rot": [
        0.18, 0.22, 0.26, 0.29, 0.32, 0.36,
        0.38, 0.40, 0.41, 0.40, 0.45, 0.40,
        0.38, 0.40, 0.35, 0.30, 0.28, 0.22
    ]
}

def simulate_reading(apple_type="fresh"):
    """
    Generates a simulated normalized reading
    as if it came from the real AS7265x sensor.

    Parameters:
        apple_type : str — "fresh", "early_rot", or "severe_rot"

    Returns:
        dict — channel names to normalized values
    """
    base = SIMULATED_PROFILES[apple_type]

    # Add small random noise — simulates natural apple variation
    noisy = [
        round(max(0.0, min(1.0, v + random.uniform(-0.03, 0.03))), 4)
        for v in base
    ]

    return {f"ch_{w}nm": v for w, v in zip(CHANNELS, noisy)}


# ══════════════════════════════════════════════════════════════════════════
# REAL SENSOR MODE
# Uncomment this entire section when AS7265x is connected to Pi
# ══════════════════════════════════════════════════════════════════════════

# import smbus2
# import sys
#
# # AS7265x I2C address (fixed — always 0x49)
# AS7265X_ADDR = 0x49
# BUS_NUMBER   = 1   # Raspberry Pi uses bus 1
#
# # AS7265x register addresses for each channel
# # These are the exact register addresses from the AS7265x datasheet
# CHANNEL_REGISTERS = [
#     0x08, 0x0A, 0x0C, 0x0E, 0x10, 0x12,   # AS72652 channels (UV/Vis)
#     0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E,   # AS72651 channels (Visible)
#     0x20, 0x22, 0x24, 0x26, 0x28, 0x2A,   # AS72653 channels (NIR)
# ]
#
# def read_real_sensor():
#     """
#     Reads all 18 channels from the physical AS7265x sensor.
#     Returns raw integer values — normalize separately.
#     Run: i2cdetect -y 1 first to confirm sensor at 0x49
#     """
#     try:
#         bus = smbus2.SMBus(BUS_NUMBER)
#
#         raw_values = []
#         for reg in CHANNEL_REGISTERS:
#             # Each channel is 2 bytes — read word (16-bit value)
#             raw = bus.read_word_data(AS7265X_ADDR, reg)
#             raw_values.append(raw)
#
#         bus.close()
#         return raw_values
#
#     except Exception as e:
#         print(f"Sensor read error: {e}")
#         print("Check wiring: SDA→Pin3, SCL→Pin5, 3.3V→Pin1, GND→Pin6")
#         sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# CALIBRATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def run_calibration_simulation():
    """
    Sends simulated dark and white reference readings
    to the FastAPI /calibrate endpoint.
    In real mode: replace with actual sensor readings.
    """
    print("\nRunning calibration...")

    # Simulated dark reference — low values, LEDs off
    dark_ref  = [10, 11, 10, 12, 11, 10,
                 11, 12, 10, 11, 10, 11,
                 10, 11, 10, 11, 10, 11]

    # Simulated white reference — high values, PTFE tile
    white_ref = [900, 880, 870, 860, 850, 840,
                 830, 820, 810, 800, 790, 780,
                 850, 870, 920, 890, 870, 840]

    response = requests.post(
        f"{API_URL}/calibrate",
        json={
            "dark_readings":  dark_ref,
            "white_readings": white_ref
        }
    )

    if response.status_code == 200:
        print(f"  ✅ {response.json()['status']}")
        return True
    else:
        print(f"  ❌ Calibration failed: {response.text}")
        return False


# ══════════════════════════════════════════════════════════════════════════
# MAIN SCAN FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def scan_apple(apple_type="fresh", scan_number=1):
    """
    Simulates scanning an apple and sends reading
    to FastAPI server for prediction.

    Parameters:
        apple_type  : str — "fresh", "early_rot", "severe_rot"
        scan_number : int — scan identifier for display
    """

    print(f"\n{'='*55}")
    print(f"SCAN #{scan_number} — Simulating: {apple_type.upper()} apple")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}")

    # Get simulated normalized reading
    normalized = simulate_reading(apple_type)

    print("\nSpectral readings (normalized 0.0-1.0):")
    for ch, val in normalized.items():
        # Simple bar visualization in terminal
        bar_length = int(val * 30)
        bar = "█" * bar_length
        print(f"  {ch:12s}: {val:.4f} {bar}")

    # Send to FastAPI server
    print(f"\nSending to ECO-LOOP API...")
    start = time.time()

    response = requests.post(
        f"{API_URL}/predict/normalized",
        json=normalized
    )

    elapsed = time.time() - start

    if response.status_code != 200:
        print(f"❌ API Error: {response.text}")
        return

    result = response.json()

    # ── Display results ────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"ECO-LOOP RESULT")
    print(f"{'─'*55}")
    print(f"  Rot Class    : {result['rot_class_name']}")
    print(f"  Predicted pH : {result['predicted_pH']}")
    print(f"  Confidence   : {result['confidence']}")
    print(f"  Response Time: {elapsed:.3f} seconds")

    rec = result["compost_recommendation"]
    print(f"\n{'─'*55}")
    print(f"COMPOST PRESCRIPTION")
    print(f"{'─'*55}")
    print(f"  Rot Stage    : {rec['rot_stage']}")
    print(f"  C:N Ratio    : {rec['cn_ratio']}")
    print(f"  Amendment    : {rec['primary_amendment']}")
    print(f"  Action       : {rec['action']}")
    print(f"  Ready In     : {rec['expected_compost_ready']}")
    print(f"  ⚠️  Warning  : {rec['warning']}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# RUN DEMO — tests all 3 apple types in sequence
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*55)
    print("ECO-LOOP — Sensor Module Test")
    print(f"Mode: {MODE.upper()}")
    print("="*55)

    # Step 1 — Check server is running
    print("\nChecking API server...")
    try:
        health = requests.get(f"{API_URL}/")
        print(f"  ✅ Server is running — {health.json()['status']}")
    except:
        print("  ❌ Server not running!")
        print("  Start it with: uvicorn api.main:app --reload")
        exit()

    # Step 2 — Calibrate
    calibrated = run_calibration_simulation()
    if not calibrated:
        exit()

    # Step 3 — Scan all 3 apple types
    print("\nStarting demo scans...")
    time.sleep(1)

    scan_apple("fresh",       scan_number=1)
    time.sleep(1)
    scan_apple("early_rot",   scan_number=2)
    time.sleep(1)
    scan_apple("severe_rot",  scan_number=3)

    print(f"\n{'='*55}")
    print("Demo complete — all 3 apple types scanned successfully")
    print("System ready for real sensor connection")
    print("="*55)