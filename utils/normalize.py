# ============================================================
# ECO-LOOP — Calibration Normalization Helper
# utils/normalize.py
#
# Converts raw AS7265x integer readings to 0.0-1.0
# reflectance values using dark and white reference scans
#
# Formula: reflectance = (raw - dark) / (white - dark)
# ============================================================

# ── The 18 AS7265x channel wavelengths in order ────────────────────────────
CHANNEL_WAVELENGTHS = [
    410, 435, 460, 485, 510, 535,
    560, 585, 610, 645, 680, 705,
    730, 760, 810, 860, 900, 940
]

CHANNEL_NAMES = [f"ch_{w}nm" for w in CHANNEL_WAVELENGTHS]


def normalize_reading(raw: list,
                      dark: list,
                      white: list) -> dict:
    """
    Normalizes raw AS7265x readings using dark and white references.

    Parameters:
        raw   : list of 18 raw integer readings from apple scan
        dark  : list of 18 dark reference readings (box sealed, LEDs off)
        white : list of 18 white reference readings (PTFE tile, LEDs on)

    Returns:
        dict with channel names as keys and normalized
        reflectance values (0.0 to 1.0) as values
    """

    if len(raw) != 18 or len(dark) != 18 or len(white) != 18:
        raise ValueError(
            f"All inputs must have exactly 18 values. "
            f"Got raw={len(raw)}, dark={len(dark)}, white={len(white)}"
        )

    normalized = {}

    for i, channel_name in enumerate(CHANNEL_NAMES):

        raw_val   = float(raw[i])
        dark_val  = float(dark[i])
        white_val = float(white[i])

        # Avoid division by zero if white = dark
        denominator = white_val - dark_val
        if denominator == 0:
            normalized[channel_name] = 0.0
            continue

        # Apply normalization formula
        value = (raw_val - dark_val) / denominator

        # Clip to 0.0-1.0 range
        # Values outside this range indicate calibration error
        value = max(0.0, min(1.0, value))

        normalized[channel_name] = round(value, 4)

    return normalized


def validate_calibration(dark: list, white: list) -> dict:
    """
    Checks if dark and white reference readings are valid
    before starting a scanning session.

    Returns a validation report.
    """

    issues = []

    for i, ch in enumerate(CHANNEL_NAMES):
        if white[i] <= dark[i]:
            issues.append(
                f"{ch}: white reference ({white[i]}) is not "
                f"greater than dark reference ({dark[i]}) — "
                f"recalibrate this channel"
            )

        if dark[i] < 0:
            issues.append(f"{ch}: negative dark reference — "
                          f"sensor error")

    return {
        "calibration_valid": len(issues) == 0,
        "issues_found": len(issues),
        "details": issues if issues else ["All 18 channels calibrated correctly"]
    }


def reading_to_list(normalized_dict: dict) -> list:
    """
    Converts normalized dict back to ordered list
    for passing into scikit-learn models.

    Models expect a list in the exact channel order,
    not a dictionary.
    """
    return [normalized_dict[ch] for ch in CHANNEL_NAMES]


# ══════════════════════════════════════════════════════════════════════════
# TEST — run directly to verify normalization works
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 55)
    print("ECO-LOOP Normalization Helper — Test Run")
    print("=" * 55)

    # Simulated dark reference (low values — noise floor)
    dark_ref = [10, 11, 10, 12, 11, 10,
                11, 12, 10, 11, 10, 11,
                10, 11, 10, 11, 10, 11]

    # Simulated white reference (high values — PTFE tile)
    white_ref = [900, 880, 870, 860, 850, 840,
                 830, 820, 810, 800, 790, 780,
                 850, 870, 920, 890, 870, 840]

    # Simulated raw apple reading (fresh apple)
    raw_fresh = [280, 310, 360, 400, 440, 490,
                 510, 530, 540, 510, 260, 490,
                 570, 620, 750, 680, 620, 560]

    # Simulated raw apple reading (severe rot apple)
    raw_rotten = [160, 190, 220, 250, 270, 300,
                  320, 340, 350, 345, 380, 340,
                  330, 350, 310, 270, 250, 195]

    print("\n--- Validating calibration references ---")
    validation = validate_calibration(dark_ref, white_ref)
    print(f"  Calibration valid : {validation['calibration_valid']}")
    print(f"  Details           : {validation['details'][0]}")

    print("\n--- Normalizing Fresh Apple Reading ---")
    fresh_normalized = normalize_reading(raw_fresh, dark_ref, white_ref)
    for ch, val in fresh_normalized.items():
        print(f"  {ch:12s} : {val:.4f}")

    print("\n--- Normalizing Rotten Apple Reading ---")
    rotten_normalized = normalize_reading(raw_rotten, dark_ref, white_ref)
    for ch, val in rotten_normalized.items():
        print(f"  {ch:12s} : {val:.4f}")

    print("\n--- Comparing key diagnostic bands ---")
    print(f"  680nm  — Fresh: {fresh_normalized['ch_680nm']:.4f} "
          f"| Rotten: {rotten_normalized['ch_680nm']:.4f} "
          f"| Rotten HIGHER = chlorophyll breakdown ✓")

    print(f"  810nm  — Fresh: {fresh_normalized['ch_810nm']:.4f} "
          f"| Rotten: {rotten_normalized['ch_810nm']:.4f} "
          f"| Fresh HIGHER = healthy cells ✓")

    print(f"  940nm  — Fresh: {fresh_normalized['ch_940nm']:.4f} "
          f"| Rotten: {rotten_normalized['ch_940nm']:.4f} "
          f"| Fresh HIGHER = more water content ✓")

    print("\n--- Converting to list for ML model input ---")
    fresh_list = reading_to_list(fresh_normalized)
    print(f"  List length : {len(fresh_list)} values")
    print(f"  First 3     : {fresh_list[:3]}")
    print(f"  Last 3      : {fresh_list[-3:]}")