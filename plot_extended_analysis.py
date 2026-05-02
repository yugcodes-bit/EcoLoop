# ============================================================
# ECO-LOOP — Extended Analysis
# plot_extended_analysis.py
#
# 1. Excludes outlier (rotten_apple_1 instrument error)
# 2. Clean main plots — with-skin samples only
# 3. Paired comparison — skin influence analysis
# 4. Extended range 1000-1500nm — deep NIR discovery
# 5. Updated numerical findings at 6 key wavelengths
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
INPUT_FILE = r"D:\ECOLOOP\ecoloop\iiser_processing\master_iiser_dataset.csv"
OUTPUT_DIR = r"D:\ECOLOOP\ecoloop\iiser_processing\plots_extended"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────
COLORS = {
    "Fresh":  "#2ecc71",
    "Rotten": "#e74c3c",
}

KEY_BANDS = {
    680:  ("Chlorophyll",      "#8e44ad"),
    810:  ("Cell Structure",   "#2980b9"),
    940:  ("Water (NIR-1)",    "#16a085"),
    1200: ("Sugar/Carb",       "#f39c12"),
    1300: ("Deep Tissue",      "#d35400"),
    1450: ("Water (NIR-2)",    "#c0392b"),
}

OUTLIER_FOLDER = "rotten_apple_1"  # excluded — instrument error

# ══════════════════════════════════════════════════════════════════════════
# LOAD AND PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("ECO-LOOP — Extended Analysis (300-1500nm)")
print("="*60)

df = pd.read_csv(INPUT_FILE)

# Detect skin status
def detect_skin(name):
    name_lower = str(name).lower()
    if "without skin" in name_lower or "no skin" in name_lower:
        return "WITHOUT_SKIN"
    return "WITH_SKIN"

df["skin_status"] = df["sample_name"].apply(detect_skin)

# Exclude the outlier
df_clean = df[df["folder"] != OUTLIER_FOLDER].copy()

# Restrict to 300-1500nm
df_clean = df_clean[(df_clean["wavelength_nm"] >= 300) &
                    (df_clean["wavelength_nm"] <= 1500)]

print(f"\n  Outlier excluded: {OUTLIER_FOLDER}")
print(f"  Clean dataset    : {df_clean['folder'].nunique()} unique scans")
print(f"  Wavelength range : 300-1500nm")


# ══════════════════════════════════════════════════════════════════════════
# HELPER — Mark key bands
# ══════════════════════════════════════════════════════════════════════════
def mark_bands(ax, y_top, bands_to_show=None):
    if bands_to_show is None:
        bands_to_show = list(KEY_BANDS.keys())
    for wl in bands_to_show:
        label, color = KEY_BANDS[wl]
        ax.axvline(x=wl, color=color, linestyle="--", alpha=0.55, linewidth=1.1)
        ax.text(wl, y_top * 0.97, f"{wl}\n{label}",
                fontsize=7, color=color, ha="center", va="top",
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.9, pad=1.5))


# ══════════════════════════════════════════════════════════════════════════
# PLOT 5 — Clean with-skin scans only, full 300-1500nm
# ══════════════════════════════════════════════════════════════════════════
def plot_clean_full_range():
    fig, ax = plt.subplots(figsize=(15, 7))

    refl = df_clean[(df_clean["measurement"] == "reflectance") &
                    (df_clean["skin_status"] == "WITH_SKIN")]

    for folder in refl["folder"].unique():
        scan = refl[refl["folder"] == folder].sort_values("wavelength_nm")
        apple_type = scan["apple_type"].iloc[0]
        ax.plot(scan["wavelength_nm"], scan["intensity"],
                color=COLORS[apple_type], alpha=0.55, linewidth=1.4)

    y_max = refl["intensity"].max()
    mark_bands(ax, y_max)

    from matplotlib.lines import Line2D
    n_fresh  = refl[refl['apple_type']=='Fresh']['folder'].nunique()
    n_rotten = refl[refl['apple_type']=='Rotten']['folder'].nunique()
    legend_elements = [
        Line2D([0], [0], color=COLORS["Fresh"],  linewidth=2.5,
               label=f"Fresh ({n_fresh} with-skin scans)"),
        Line2D([0], [0], color=COLORS["Rotten"], linewidth=2.5,
               label=f"Rotten ({n_rotten} with-skin scans)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

    ax.set_title("Clean Reflectance Scans — With-Skin Samples Only (300-1500nm)\n"
                 "Outlier Excluded | Real IISER Shimadzu UV-3600i Plus Data",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance (R%)", fontsize=11)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, "plot5_clean_full_range.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 5 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 6 — Mean comparison clean (with-skin only)
# ══════════════════════════════════════════════════════════════════════════
def plot_clean_mean():
    fig, ax = plt.subplots(figsize=(15, 7))

    refl = df_clean[(df_clean["measurement"] == "reflectance") &
                    (df_clean["skin_status"] == "WITH_SKIN")]

    for apple_type in ["Fresh", "Rotten"]:
        subset = refl[refl["apple_type"] == apple_type]
        grouped = subset.groupby("wavelength_nm")["intensity"]
        mean = grouped.mean()
        std  = grouped.std()

        ax.plot(mean.index, mean.values, color=COLORS[apple_type],
                linewidth=3, label=f"{apple_type} (mean)")
        ax.fill_between(mean.index, mean - std, mean + std,
                        color=COLORS[apple_type], alpha=0.2)

    y_max = refl["intensity"].max()
    mark_bands(ax, y_max)

    ax.set_title("Mean Spectral Signature — Clean With-Skin Samples (300-1500nm)\n"
                 "Real IISER Data | Shaded = Variance Across Scans | Outlier Excluded",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance (R%)", fontsize=11)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, "plot6_clean_mean.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 6 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 7 — Skin Influence (paired comparison)
# ══════════════════════════════════════════════════════════════════════════
def plot_skin_influence():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    refl = df_clean[df_clean["measurement"] == "reflectance"]

    for idx, apple_type in enumerate(["Fresh", "Rotten"]):
        ax = axes[idx]
        subset = refl[refl["apple_type"] == apple_type]

        # Mean of with-skin
        with_skin    = subset[subset["skin_status"] == "WITH_SKIN"]
        without_skin = subset[subset["skin_status"] == "WITHOUT_SKIN"]

        if len(with_skin) > 0:
            mean_ws = with_skin.groupby("wavelength_nm")["intensity"].mean()
            ax.plot(mean_ws.index, mean_ws.values,
                    color=COLORS[apple_type], linewidth=2.8,
                    linestyle="-", label=f"{apple_type} WITH skin")

        if len(without_skin) > 0:
            mean_ns = without_skin.groupby("wavelength_nm")["intensity"].mean()
            ax.plot(mean_ns.index, mean_ns.values,
                    color=COLORS[apple_type], linewidth=2.8,
                    linestyle="--", label=f"{apple_type} WITHOUT skin")

        y_max = subset["intensity"].max()
        mark_bands(ax, y_max, bands_to_show=[680, 810, 940])

        ax.set_title(f"{apple_type} Apple — Skin Influence\n"
                     f"Solid = with skin | Dashed = without skin",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)", fontsize=10)
        ax.set_ylabel("Reflectance (R%)", fontsize=10)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Paired Skin Influence Analysis — Same Physical Apple With and Without Skin",
                 fontsize=13, fontweight="bold", y=1.02)

    out = os.path.join(OUTPUT_DIR, "plot7_skin_influence.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 7 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 8 — Deep NIR Zoom (1000-1500nm)
# ══════════════════════════════════════════════════════════════════════════
def plot_deep_nir_zoom():
    fig, ax = plt.subplots(figsize=(15, 7))

    refl = df_clean[(df_clean["measurement"] == "reflectance") &
                    (df_clean["skin_status"] == "WITH_SKIN") &
                    (df_clean["wavelength_nm"] >= 1000) &
                    (df_clean["wavelength_nm"] <= 1500)]

    for apple_type in ["Fresh", "Rotten"]:
        subset = refl[refl["apple_type"] == apple_type]
        grouped = subset.groupby("wavelength_nm")["intensity"]
        mean = grouped.mean()
        std  = grouped.std()

        ax.plot(mean.index, mean.values, color=COLORS[apple_type],
                linewidth=3, label=f"{apple_type} (mean)")
        ax.fill_between(mean.index, mean - std, mean + std,
                        color=COLORS[apple_type], alpha=0.2,
                        label=f"{apple_type} (±1 std)")

    # Mark deep NIR specific bands
    y_max = refl["intensity"].max()
    for wl in [1200, 1300, 1450]:
        label, color = KEY_BANDS[wl]
        ax.axvline(x=wl, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(wl, y_max * 0.95, f"{wl}nm\n{label}",
                fontsize=9, color=color, ha="center", va="top",
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.9, pad=2))

    # Mark AS7265x sensor cutoff
    ax.axvline(x=940, color="black", linestyle=":", alpha=0.6, linewidth=1.5)
    ax.text(960, y_max * 0.5, "AS7265x sensor\nrange ends here →",
            fontsize=9, color="black", ha="left", va="center", style="italic")

    ax.set_title("DEEP NIR REGION (1000-1500nm) — Beyond Current Sensor Range\n"
                 "Discovery: Strong Fresh vs Rotten Discrimination in Extended NIR",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance (R%)", fontsize=11)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, "plot8_deep_nir_zoom.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 8 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# UPDATED FINDINGS — clean numbers at all 6 wavelengths
# ══════════════════════════════════════════════════════════════════════════
def print_clean_findings():
    print("\n" + "="*60)
    print("CLEAN FINDINGS — WITH-SKIN ONLY, OUTLIER EXCLUDED")
    print("="*60)

    refl = df_clean[(df_clean["measurement"] == "reflectance") &
                    (df_clean["skin_status"] == "WITH_SKIN")]

    print(f"\n  Sample basis: {refl[refl['apple_type']=='Fresh']['folder'].nunique()} Fresh, "
          f"{refl[refl['apple_type']=='Rotten']['folder'].nunique()} Rotten (with-skin scans)\n")

    print(f"  {'Wavelength':<12} {'Fresh Mean':<12} {'Rotten Mean':<12} {'Difference':<14} {'% Diff':<10}")
    print(f"  {'-'*60}")

    for wl, (label, _) in KEY_BANDS.items():
        near = refl[refl["wavelength_nm"].between(wl - 1, wl + 1)]
        fresh_mean  = near[near["apple_type"]=="Fresh"]["intensity"].mean()
        rotten_mean = near[near["apple_type"]=="Rotten"]["intensity"].mean()
        diff = fresh_mean - rotten_mean
        pct = (diff / rotten_mean * 100) if rotten_mean else 0

        print(f"  {wl}nm ({label[:10]:10s}) "
              f"{fresh_mean:8.2f}    {rotten_mean:8.2f}    "
              f"{diff:+8.2f}      {pct:+6.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGenerating extended analysis...\n")

    plot_clean_full_range()
    plot_clean_mean()
    plot_skin_influence()
    plot_deep_nir_zoom()

    print_clean_findings()

    print("\n" + "="*60)
    print("EXTENDED ANALYSIS COMPLETE")
    print("="*60)
    print(f"  Output directory: {OUTPUT_DIR}\n")