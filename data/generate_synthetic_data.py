import pandas as pd
import numpy as np

# Set seed so results are reproducible (same data every run)
np.random.seed(42)

# ── AS7265x channel wavelengths (nm) ──────────────────────────────────────
# These are the exact 18 wavelengths the real sensor measures
CHANNELS = [
    410, 435, 460, 485, 510, 535,   # AS72652 - UV/Visible chip
    560, 585, 610, 645, 680, 705,   # AS72651 - Visible chip
    730, 760, 810, 860, 900, 940    # AS72653 - NIR chip
]

# ── How many samples per class ─────────────────────────────────────────────
SAMPLES_PER_CLASS = 25

# ── Base reflectance profiles per class ───────────────────────────────────
# These values come from NIR apple literature (Huang et al. 2021, Fan et al. 2020)
# Fresh apple has HIGH reflectance at 810nm (healthy cells)
# Rotten apple has LOW reflectance at 810nm (cell collapse)
# Water band at 940nm drops as rot increases (water leaks out of cells)

BASE_PROFILES = {
    # class_label: [reflectance value for each of the 18 channels]
    0: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55,   # Fresh - strong visible reflection
        0.58, 0.60, 0.62, 0.58, 0.30, 0.55,   # 680nm = low (chlorophyll absorbs here)
        0.65, 0.70, 0.80, 0.75, 0.70, 0.65],  # 810nm = HIGH (healthy cells)

    1: [0.25, 0.30, 0.35, 0.38, 0.42, 0.46,   # Early-Rot - slightly lower overall
        0.48, 0.50, 0.52, 0.49, 0.38, 0.48,   # 680nm rises (chlorophyll breaking down)
        0.52, 0.55, 0.58, 0.53, 0.48, 0.43],  # 810nm drops (cells starting to collapse)

    2: [0.18, 0.22, 0.26, 0.29, 0.32, 0.36,   # Severe-Rot - low across all channels
        0.38, 0.40, 0.41, 0.40, 0.45, 0.40,   # 680nm HIGH (chlorophyll fully gone)
        0.38, 0.40, 0.35, 0.30, 0.28, 0.22],  # 810nm = LOW (cells collapsed)
}

# ── pH ranges per class ────────────────────────────────────────────────────
PH_RANGES = {
    0: (3.8, 4.5),   # Fresh apple pH
    1: (3.2, 3.8),   # Early-Rot pH
    2: (2.8, 3.2),   # Severe-Rot pH
}

# ── Class names for readability ────────────────────────────────────────────
CLASS_NAMES = {
    0: "Fresh",
    1: "Early-Rot",
    2: "Severe-Rot"
}

# ── Generate the data ──────────────────────────────────────────────────────
all_rows = []

for class_label, base in BASE_PROFILES.items():
    for i in range(SAMPLES_PER_CLASS):

        # Add small random noise to each channel (±0.03)
        # This simulates natural variation between different apples
        noise = np.random.normal(0, 0.03, size=18)
        channels = np.clip(np.array(base) + noise, 0.0, 1.0)

        # Generate a pH value within the class range
        ph_min, ph_max = PH_RANGES[class_label]
        ph = round(np.random.uniform(ph_min, ph_max), 2)

        # Build one row: sample_id + 18 channel values + pH + class
        row = {"sample_id": f"{CLASS_NAMES[class_label]}_{i+1:02d}"}
        for j, wavelength in enumerate(CHANNELS):
            row[f"ch_{wavelength}nm"] = round(float(channels[j]), 4)
        row["pH"] = ph
        row["class_label"] = class_label
        row["class_name"] = CLASS_NAMES[class_label]

        all_rows.append(row)

# ── Save to CSV ────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
df.to_csv("data/synthetic_apple_data.csv", index=False)

print(f"Dataset generated: {len(df)} samples")
print(f"\nClass distribution:")
print(df["class_name"].value_counts())
print(f"\npH statistics per class:")
print(df.groupby("class_name")["pH"].describe().round(2))
print(f"\nFirst 3 rows preview:")
print(df.head(3).to_string())