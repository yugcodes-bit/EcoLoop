# ============================================================
# ECO-LOOP — Master Dataset Builder
# build_master_dataset.py
#
# Combines all 30 cleaned IISER CSV files into one master file
# Adds proper labels: apple_type, position, physical_apple_id
# Output: master_iiser_dataset.csv ready for analysis
# ============================================================

import os
import pandas as pd
import re

# ── Paths ──────────────────────────────────────────────────────────────────
CLEANED_DIR  = r"D:\ECOLOOP\ecoloop\iiser_processing\cleaned"
OUTPUT_FILE  = r"D:\ECOLOOP\ecoloop\iiser_processing\master_iiser_dataset.csv"
SUMMARY_FILE = os.path.join(CLEANED_DIR, "_summary.csv")

# ══════════════════════════════════════════════════════════════════════════
# IDENTIFY POSITION FROM SAMPLE NAME
# ══════════════════════════════════════════════════════════════════════════

def detect_position(sample_name):
    """
    Extracts scan position from sample name.
    Examples:
      "Top fresh apple with skin_1"     → "top"
      "Bottom fresh apple with skin_2"  → "bottom"
      "Bruised fresh apple with skin_1" → "bruised"
      "Side rot apple_3"                → "side"
    """
    name_lower = sample_name.lower()

    if "top" in name_lower:
        return "top"
    elif "bottom" in name_lower:
        return "bottom"
    elif "bruised" in name_lower or "bruise" in name_lower:
        return "bruised"
    elif "side" in name_lower:
        return "side"
    elif "skin" in name_lower:
        return "skin"
    else:
        return "unknown"


def detect_physical_apple(folder_name):
    """
    Maps folder number to physical apple ID.
    Since you scanned 4 physical apples (2 fresh, 2 rotten)
    multiple times — we need to figure out which scans 
    belong to which physical apple.
    
    For now, we group:
    - fresh_apple_1 to fresh_apple_4 → Physical Fresh Apple 1
    - fresh_apple_5 to fresh_apple_8 → Physical Fresh Apple 2
    - rotten_apple_1 to rotten_apple_4 → Physical Rotten Apple 1
    - rotten_apple_5 to rotten_apple_7 → Physical Rotten Apple 2
    
    YOU CAN ADJUST THIS based on actual scan order.
    """
    # Extract the number from folder name
    match = re.search(r'_(\d+)$', folder_name)
    if not match:
        return "Unknown"

    num = int(match.group(1))

    if "fresh" in folder_name.lower():
        if num <= 4:
            return "Fresh_Apple_A"
        else:
            return "Fresh_Apple_B"
    elif "rot" in folder_name.lower():
        if num <= 4:
            return "Rotten_Apple_A"
        else:
            return "Rotten_Apple_B"

    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════
# BUILD MASTER DATASET
# ══════════════════════════════════════════════════════════════════════════

def build_master():
    print("\n" + "="*60)
    print("ECO-LOOP — Building Master Dataset")
    print("="*60)

    # Read summary to get folder → apple type mapping
    summary = pd.read_csv(SUMMARY_FILE)

    print(f"\nLoaded summary: {len(summary)} entries")
    print(f"Cleaned directory: {CLEANED_DIR}\n")

    all_data = []

    for _, row in summary.iterrows():

        # Skip peaks files (different format)
        if "peaks" in row["data_type"]:
            continue

        # Read the cleaned CSV
        csv_path = os.path.join(CLEANED_DIR, row["output_file"])
        if not os.path.exists(csv_path):
            print(f"❌ Missing: {row['output_file']}")
            continue

        df = pd.read_csv(csv_path)

        # Add metadata columns
        df["apple_type"]      = row["apple_type"]
        df["folder"]          = row["folder"]
        df["sample_name"]     = row["sample_name"]
        df["position"]        = detect_position(row["sample_name"])
        df["physical_apple"]  = detect_physical_apple(row["folder"])
        df["measurement"]     = row["data_type"].replace("_full", "")  # "reflectance" or "absorbance"

        # Rename value column for clarity
        df.rename(columns={"value": "intensity"}, inplace=True)

        all_data.append(df)

        print(f"  ✓ {row['folder']:20s} | {row['data_type']:20s} | "
              f"{detect_position(row['sample_name']):8s} | "
              f"{detect_physical_apple(row['folder'])}")

    # Combine everything
    master_df = pd.concat(all_data, ignore_index=True)

    # Reorder columns for readability
    column_order = [
        "physical_apple", "apple_type", "folder", "position",
        "sample_name", "measurement", "wavelength_nm", "intensity"
    ]
    master_df = master_df[column_order]

    # Save
    master_df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary stats ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MASTER DATASET BUILT")
    print("="*60)
    print(f"  Output file       : {OUTPUT_FILE}")
    print(f"  Total rows        : {len(master_df):,}")
    print(f"  Unique scans      : {master_df['folder'].nunique()}")
    print(f"  Physical apples   : {master_df['physical_apple'].nunique()}")
    print(f"  Wavelength range  : {master_df['wavelength_nm'].min():.0f}-"
          f"{master_df['wavelength_nm'].max():.0f}nm")

    print("\n  Scans per physical apple:")
    counts = master_df.groupby(["physical_apple", "measurement"])["folder"].nunique().reset_index()
    counts.columns = ["physical_apple", "measurement", "scan_count"]
    print(counts.to_string(index=False))

    print("\n  Position distribution:")
    pos_counts = master_df.drop_duplicates(["folder", "measurement"])["position"].value_counts()
    for pos, cnt in pos_counts.items():
        print(f"     {pos:10s} : {cnt} scans")

    print("\nNext: open master_iiser_dataset.csv and verify the structure")
    print()


if __name__ == "__main__":
    build_master()