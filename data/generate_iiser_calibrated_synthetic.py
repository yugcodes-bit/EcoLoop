# ============================================================
# ECO-LOOP — IISER-Calibrated Synthetic Data Generator
# data/generate_iiser_calibrated_synthetic.py
#
# Builds synthetic training data calibrated to REAL IISER
# pilot measurements at the 18 AS7265x wavelengths.
#
# Replaces theoretical baseline values with actual lab data.
# ============================================================

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
IISER_MASTER = r"D:\ECOLOOP\ecoloop\iiser_processing\master_iiser_dataset.csv"
OUTPUT_CSV   = r"D:\ECOLOOP\ecoloop\data\synthetic_apple_data.csv"

# ── AS7265x channel wavelengths (the 18 our sensor reads) ──────────────────
AS7265X_WAVELENGTHS = [
    410, 435, 460, 485, 510, 535,
    560, 585, 610, 645, 680, 705,
    730, 760, 810, 860, 900, 940
]
CHANNEL_NAMES = [f"ch_{w}nm" for w in AS7265X_WAVELENGTHS]

# ── Outlier scan to exclude (instrument error from Phase 1) ────────────────
OUTLIER_SCAN = "rotten_apple_1"

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load REAL IISER data and extract 18 AS7265x wavelengths
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("ECO-LOOP — IISER-Calibrated Synthetic Data Generator")
print("="*60)
print(f"\nLoading real IISER data from: {IISER_MASTER}")

df = pd.read_csv(IISER_MASTER)

# Use reflectance only (matches what AS7265x measures)
# Use with-skin only (matches deployment scenario)
df = df[df["measurement"] == "reflectance"].copy()
df = df[~df["sample_name"].str.lower().str.contains("without skin")]
df = df[df["folder"] != OUTLIER_SCAN]

print(f"  Clean scans loaded: {df['folder'].nunique()} scans")
print(f"  Fresh scans       : {df[df['apple_type']=='Fresh']['folder'].nunique()}")
print(f"  Rotten scans      : {df[df['apple_type']=='Rotten']['folder'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Extract real reflectance values at the 18 AS7265x wavelengths
# ══════════════════════════════════════════════════════════════════════════

def extract_18_channels(scan_df):
    """
    For one scan, extract values at exactly the 18 AS7265x wavelengths.
    Uses nearest-wavelength matching (real data is at 1nm intervals, 
    so closest match is essentially exact).
    Converts from R% (0-100) to normalized 0-1 scale for ML compatibility.
    """
    values = []
    for wl in AS7265X_WAVELENGTHS:
        # Find row with wavelength closest to target
        nearest = scan_df.iloc[(scan_df['wavelength_nm'] - wl).abs().argsort()[:1]]
        val = float(nearest['intensity'].values[0])
        # Convert from R% to 0-1 scale (IISER gives 0-100, AS7265x gives 0-1)
        normalized = val / 100.0
        # Clip just in case
        normalized = max(0.0, min(1.0, normalized))
        values.append(normalized)
    return values


# Extract real 18-channel readings for each clean IISER scan
real_readings = {"Fresh": [], "Rotten": []}

for folder in df["folder"].unique():
    scan = df[df["folder"] == folder]
    apple_type = scan["apple_type"].iloc[0]
    values = extract_18_channels(scan)
    real_readings[apple_type].append(values)

real_readings["Fresh"]  = np.array(real_readings["Fresh"])
real_readings["Rotten"] = np.array(real_readings["Rotten"])

# Compute REAL means and standard deviations (these are our calibration anchors)
fresh_mean  = real_readings["Fresh"].mean(axis=0)
fresh_std   = real_readings["Fresh"].std(axis=0)
rotten_mean = real_readings["Rotten"].mean(axis=0)
rotten_std  = real_readings["Rotten"].std(axis=0)

# For Early-Rot, interpolate between Fresh and Rotten (60% Fresh + 40% Rotten)
early_mean = 0.6 * fresh_mean + 0.4 * rotten_mean
early_std  = np.maximum(fresh_std, rotten_std) * 0.8   # slightly less variance

print(f"\n  Extracted 18-channel signatures from real IISER data ✓")
print(f"  Fresh mean   @ 810nm: {fresh_mean[14]:.3f}")
print(f"  Rotten mean  @ 810nm: {rotten_mean[14]:.3f}")
print(f"  Early mean   @ 810nm: {early_mean[14]:.3f} (interpolated)")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Generate synthetic samples around the REAL anchors
# ══════════════════════════════════════════════════════════════════════════

# Class profiles built from real IISER measurements (not theoretical values)
CLASS_PROFILES = {
    "Fresh": {
        "mean": fresh_mean,
        "std":  fresh_std + 0.02,        # slight noise injection
        "label": 0,
        "ph_mean": 4.2,
        "ph_std":  0.2
    },
    "Early-Rot": {
        "mean": early_mean,
        "std":  early_std + 0.02,
        "label": 1,
        "ph_mean": 3.5,
        "ph_std":  0.15
    },
    "Severe-Rot": {
        "mean": rotten_mean,
        "std":  rotten_std + 0.02,
        "label": 2,
        "ph_mean": 2.9,
        "ph_std":  0.15
    }
}

# Generate samples
SAMPLES_PER_CLASS = 30
np.random.seed(42)   # reproducibility

all_rows = []

for class_name, profile in CLASS_PROFILES.items():
    print(f"\n  Generating {SAMPLES_PER_CLASS} {class_name} samples...")

    for i in range(SAMPLES_PER_CLASS):
        # Sample 18 spectral values around the real mean with realistic noise
        spectral = np.random.normal(profile["mean"], profile["std"])
        # Clip to valid 0-1 range
        spectral = np.clip(spectral, 0.0, 1.0)

        # Sample pH value
        ph = np.random.normal(profile["ph_mean"], profile["ph_std"])
        # Clip to valid pH range
        ph = max(2.5, min(5.0, ph))

        # Build row
        row = {ch: round(val, 4) for ch, val in zip(CHANNEL_NAMES, spectral)}
        row["pH"] = round(ph, 2)
        row["class_label"] = profile["label"]
        row["class_name"]  = class_name

        all_rows.append(row)


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Save to CSV
# ══════════════════════════════════════════════════════════════════════════

synth_df = pd.DataFrame(all_rows)
synth_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n" + "="*60)
print(f"SYNTHETIC DATASET BUILT (IISER-CALIBRATED)")
print(f"="*60)
print(f"  Total samples : {len(synth_df)}")
print(f"  Per class     : {SAMPLES_PER_CLASS} (Fresh / Early-Rot / Severe-Rot)")
print(f"  Output file   : {OUTPUT_CSV}")

# Quick sanity check on key bands
print(f"\n  Sanity check at key bands:")
for class_name in ["Fresh", "Early-Rot", "Severe-Rot"]:
    subset = synth_df[synth_df["class_name"] == class_name]
    print(f"    {class_name:11s}  680nm: {subset['ch_680nm'].mean():.3f} | "
          f"810nm: {subset['ch_810nm'].mean():.3f} | "
          f"940nm: {subset['ch_940nm'].mean():.3f} | "
          f"pH: {subset['pH'].mean():.2f}")

print(f"\nNext steps:")
print(f"  1. Retrain models on this new data:")
print(f"     python models/train_models.py")
print(f"  2. Verify charts still work with new data:")
print(f"     python notebooks/chart6_pca_plot.py")
print()