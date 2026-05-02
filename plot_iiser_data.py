# ============================================================
# ECO-LOOP — IISER Data Visualization
# plot_iiser_data.py
#
# 4 critical plots from real IISER pilot data:
# 1. All reflectance scans overlaid
# 2. All absorbance scans overlaid
# 3. Mean Fresh vs Mean Rotten — Reflectance
# 4. Mean Fresh vs Mean Rotten — Absorbance
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
INPUT_FILE = r"D:\ECOLOOP\ecoloop\iiser_processing\master_iiser_dataset.csv"
OUTPUT_DIR = r"D:\ECOLOOP\ecoloop\iiser_processing\plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colors ─────────────────────────────────────────────────────────────────
COLORS = {
    "Fresh":  "#2ecc71",   # green
    "Rotten": "#e74c3c",   # red
}

# ── Key diagnostic wavelengths ─────────────────────────────────────────────
KEY_BANDS = {
    680:  ("Chlorophyll Band",   "#8e44ad"),
    810:  ("Cell Structure Band","#2980b9"),
    940:  ("Water Band",         "#16a085"),
}

# ── AS7265x channel wavelengths (for marking on plot) ──────────────────────
AS7265X_CHANNELS = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645,
                     680, 705, 730, 760, 810, 860, 900, 940]


# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("ECO-LOOP — IISER Data Visualization")
print("="*60)

print(f"\nLoading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# Filter to scan range we care about (300-1000nm matches our sensor zone + buffer)
df = df[(df["wavelength_nm"] >= 300) & (df["wavelength_nm"] <= 1000)].copy()

print(f"  Total rows           : {len(df):,}")
print(f"  Reflectance scans    : {df[df['measurement']=='reflectance']['folder'].nunique()}")
print(f"  Absorbance scans     : {df[df['measurement']=='absorbance']['folder'].nunique()}")
print(f"  Wavelength range     : {df['wavelength_nm'].min():.0f}-{df['wavelength_nm'].max():.0f}nm")


# ══════════════════════════════════════════════════════════════════════════
# HELPER — Mark key bands on a plot
# ══════════════════════════════════════════════════════════════════════════
def mark_key_bands(ax, y_max):
    for wl, (label, color) in KEY_BANDS.items():
        ax.axvline(x=wl, color=color, linestyle="--", alpha=0.6, linewidth=1.2)
        ax.text(wl, y_max * 0.97, f"{wl}nm\n{label}",
                fontsize=8, color=color, ha="center", va="top",
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.9, pad=2))


# ══════════════════════════════════════════════════════════════════════════
# PLOT 1 — All Reflectance Scans Overlaid
# ══════════════════════════════════════════════════════════════════════════
def plot_all_scans_reflectance():
    fig, ax = plt.subplots(figsize=(14, 7))

    refl = df[df["measurement"] == "reflectance"]

    for folder in refl["folder"].unique():
        scan = refl[refl["folder"] == folder].sort_values("wavelength_nm")
        apple_type = scan["apple_type"].iloc[0]
        position   = scan["position"].iloc[0]

        ax.plot(scan["wavelength_nm"], scan["intensity"],
                color=COLORS[apple_type], alpha=0.55, linewidth=1.3,
                label=f"{apple_type} - {folder} ({position})")

    y_max = refl["intensity"].max()
    mark_key_bands(ax, y_max)

    # Custom legend — one entry per type
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["Fresh"],  linewidth=2.5,
               label=f"Fresh Apple ({refl[refl['apple_type']=='Fresh']['folder'].nunique()} scans)"),
        Line2D([0], [0], color=COLORS["Rotten"], linewidth=2.5,
               label=f"Rotten Apple ({refl[refl['apple_type']=='Rotten']['folder'].nunique()} scans)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

    ax.set_title("All IISER Reflectance Scans — Fresh vs Rotten Apples\n"
                 "Real Shimadzu UV-3600i Plus Data (300-1000nm)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance (R%)", fontsize=11)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, "plot1_all_reflectance_scans.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 1 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 2 — All Absorbance Scans Overlaid
# ══════════════════════════════════════════════════════════════════════════
def plot_all_scans_absorbance():
    fig, ax = plt.subplots(figsize=(14, 7))

    absb = df[df["measurement"] == "absorbance"]

    for folder in absb["folder"].unique():
        scan = absb[absb["folder"] == folder].sort_values("wavelength_nm")
        apple_type = scan["apple_type"].iloc[0]
        position   = scan["position"].iloc[0]

        ax.plot(scan["wavelength_nm"], scan["intensity"],
                color=COLORS[apple_type], alpha=0.55, linewidth=1.3,
                label=f"{apple_type} - {folder} ({position})")

    y_max = absb["intensity"].max()
    mark_key_bands(ax, y_max)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["Fresh"],  linewidth=2.5,
               label=f"Fresh Apple ({absb[absb['apple_type']=='Fresh']['folder'].nunique()} scans)"),
        Line2D([0], [0], color=COLORS["Rotten"], linewidth=2.5,
               label=f"Rotten Apple ({absb[absb['apple_type']=='Rotten']['folder'].nunique()} scans)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

    ax.set_title("All IISER Absorbance Scans — Fresh vs Rotten Apples\n"
                 "Real Shimadzu UV-3600i Plus Data (300-1000nm)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Absorbance (Abs)", fontsize=11)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, "plot2_all_absorbance_scans.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 2 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 3 — Mean Fresh vs Mean Rotten (Reflectance)
# ══════════════════════════════════════════════════════════════════════════
def plot_mean_reflectance():
    fig, ax = plt.subplots(figsize=(14, 7))

    refl = df[df["measurement"] == "reflectance"]

    for apple_type in ["Fresh", "Rotten"]:
        subset = refl[refl["apple_type"] == apple_type]

        # Mean and std across all scans of this type at each wavelength
        grouped = subset.groupby("wavelength_nm")["intensity"]
        mean = grouped.mean()
        std  = grouped.std()

        ax.plot(mean.index, mean.values, color=COLORS[apple_type],
                linewidth=2.8, label=f"{apple_type} (mean)")

        ax.fill_between(mean.index, mean - std, mean + std,
                        color=COLORS[apple_type], alpha=0.2,
                        label=f"{apple_type} (±1 std)")

    y_max = refl["intensity"].max()
    mark_key_bands(ax, y_max)

    ax.set_title("Mean Spectral Signature — Fresh vs Rotten (Reflectance)\n"
                 "Real IISER Data — Shaded Region = Natural Variance Across Scans",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance (R%)", fontsize=11)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, "plot3_mean_reflectance.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 3 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 4 — Mean Fresh vs Mean Rotten (Absorbance)
# ══════════════════════════════════════════════════════════════════════════
def plot_mean_absorbance():
    fig, ax = plt.subplots(figsize=(14, 7))

    absb = df[df["measurement"] == "absorbance"]

    for apple_type in ["Fresh", "Rotten"]:
        subset = absb[absb["apple_type"] == apple_type]

        grouped = subset.groupby("wavelength_nm")["intensity"]
        mean = grouped.mean()
        std  = grouped.std()

        ax.plot(mean.index, mean.values, color=COLORS[apple_type],
                linewidth=2.8, label=f"{apple_type} (mean)")

        ax.fill_between(mean.index, mean - std, mean + std,
                        color=COLORS[apple_type], alpha=0.2,
                        label=f"{apple_type} (±1 std)")

    y_max = absb["intensity"].max()
    mark_key_bands(ax, y_max)

    ax.set_title("Mean Spectral Signature — Fresh vs Rotten (Absorbance)\n"
                 "Real IISER Data — Shaded Region = Natural Variance Across Scans",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Absorbance (Abs)", fontsize=11)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, "plot4_mean_absorbance.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 4 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# KEY FINDINGS REPORT
# ══════════════════════════════════════════════════════════════════════════
def print_key_findings():
    """
    Print numerical findings at the 3 key diagnostic wavelengths.
    These are what you tell your guide.
    """
    print("\n" + "="*60)
    print("KEY FINDINGS AT DIAGNOSTIC WAVELENGTHS")
    print("="*60)

    for measurement_type in ["reflectance", "absorbance"]:
        print(f"\n--- {measurement_type.upper()} ---")
        subset = df[df["measurement"] == measurement_type]

        for wl, (label, _) in KEY_BANDS.items():
            # Find values closest to this wavelength
            near = subset[subset["wavelength_nm"].between(wl - 1, wl + 1)]

            fresh_mean  = near[near["apple_type"]=="Fresh"]["intensity"].mean()
            rotten_mean = near[near["apple_type"]=="Rotten"]["intensity"].mean()
            diff        = fresh_mean - rotten_mean
            pct_diff    = (diff / rotten_mean) * 100 if rotten_mean else 0

            print(f"  {wl}nm ({label}):")
            print(f"     Fresh  mean : {fresh_mean:.3f}")
            print(f"     Rotten mean : {rotten_mean:.3f}")
            print(f"     Difference  : {diff:+.3f} ({pct_diff:+.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# RUN ALL PLOTS
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGenerating plots...\n")

    plot_all_scans_reflectance()
    plot_all_scans_absorbance()
    plot_mean_reflectance()
    plot_mean_absorbance()

    print_key_findings()

    print("\n" + "="*60)
    print("ALL PLOTS GENERATED")
    print("="*60)
    print(f"  Output directory: {OUTPUT_DIR}")
    print("\nNext: Open each PNG and verify Fresh vs Rotten separation.")
    print("      The numbers at 680nm, 810nm, 940nm are what you ")
    print("      show your guide on Monday.\n")