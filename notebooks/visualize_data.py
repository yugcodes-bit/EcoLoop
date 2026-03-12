import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load the synthetic data ────────────────────────────────────────────────
df = pd.read_csv("data/synthetic_apple_data.csv")

# Channel wavelengths (x-axis for spectral plots)
CHANNELS = [410, 435, 460, 485, 510, 535,
            560, 585, 610, 645, 680, 705,
            730, 760, 810, 860, 900, 940]

CHANNEL_COLS = [f"ch_{w}nm" for w in CHANNELS]

# Colors for each class
COLORS = {
    "Fresh":       "#2ecc71",   # green
    "Early-Rot":   "#f39c12",   # orange
    "Severe-Rot":  "#e74c3c"    # red
}

# ── Figure with 3 subplots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Eco-Loop — Synthetic Apple Spectral Data Overview",
             fontsize=14, fontweight="bold")

# ────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Average spectral signature per class
# This is the most important plot — shows HOW the 3 classes differ
# ────────────────────────────────────────────────────────────────────────────
ax1 = axes[0]

for class_name, color in COLORS.items():
    class_df = df[df["class_name"] == class_name]
    
    # Average reflectance per channel for this class
    mean_vals = class_df[CHANNEL_COLS].mean().values
    std_vals  = class_df[CHANNEL_COLS].std().values

    # Plot average line
    ax1.plot(CHANNELS, mean_vals, color=color, linewidth=2.5, label=class_name)
    
    # Plot shaded area = natural variation between apples
    ax1.fill_between(CHANNELS,
                     mean_vals - std_vals,
                     mean_vals + std_vals,
                     color=color, alpha=0.15)

# Mark the 3 key diagnostic bands we studied
ax1.axvline(x=680, color="purple", linestyle="--", alpha=0.6, linewidth=1.2)
ax1.axvline(x=810, color="blue",   linestyle="--", alpha=0.6, linewidth=1.2)
ax1.axvline(x=940, color="gray",   linestyle="--", alpha=0.6, linewidth=1.2)

ax1.text(680, 0.82, "680nm\nChloro", fontsize=7, color="purple", ha="center")
ax1.text(810, 0.82, "810nm\nCell",   fontsize=7, color="blue",   ha="center")
ax1.text(940, 0.82, "940nm\nWater",  fontsize=7, color="gray",   ha="center")

ax1.set_title("Spectral Signatures by Rot Class", fontweight="bold")
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Reflectance (0–1)")
ax1.legend()
ax1.set_ylim(0, 0.95)
ax1.grid(True, alpha=0.3)

# ────────────────────────────────────────────────────────────────────────────
# PLOT 2 — pH distribution per class (box plot)
# Shows that pH clearly separates the 3 classes
# ────────────────────────────────────────────────────────────────────────────
ax2 = axes[1]

class_order = ["Fresh", "Early-Rot", "Severe-Rot"]
ph_data     = [df[df["class_name"] == c]["pH"].values for c in class_order]
colors_list = [COLORS[c] for c in class_order]

bp = ax2.boxplot(ph_data, patch_artist=True, labels=class_order)

for patch, color in zip(bp["boxes"], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_title("pH Distribution per Class", fontweight="bold")
ax2.set_xlabel("Rot Class")
ax2.set_ylabel("pH Value")
ax2.grid(True, alpha=0.3, axis="y")

# Add horizontal lines showing pH zones
ax2.axhline(y=3.5, color="gray", linestyle=":", alpha=0.5)
ax2.axhline(y=3.8, color="gray", linestyle=":", alpha=0.5)

# ────────────────────────────────────────────────────────────────────────────
# PLOT 3 — 810nm vs 680nm scatter plot
# The 2 most diagnostic channels plotted against each other
# Good separation here = model will classify well
# ────────────────────────────────────────────────────────────────────────────
ax3 = axes[2]

for class_name, color in COLORS.items():
    class_df = df[df["class_name"] == class_name]
    ax3.scatter(
        class_df["ch_680nm"],
        class_df["ch_810nm"],
        color=color,
        label=class_name,
        alpha=0.75,
        edgecolors="white",
        linewidth=0.5,
        s=60
    )

ax3.set_title("Key Bands: 680nm vs 810nm", fontweight="bold")
ax3.set_xlabel("Reflectance at 680nm (Chlorophyll Band)")
ax3.set_ylabel("Reflectance at 810nm (Cell Structure Band)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── Save and show ──────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig("data/spectral_overview.png", dpi=150, bbox_inches="tight")
plt.show()

print("Plot saved to data/spectral_overview.png")