# ============================================================
# ECO-LOOP — IISER Data Cleaning Script
# clean_iiser_data.py
#
# Reads all 15 IISER scan folders
# Cleans the TXT files (removes headers, anomalies)
# Outputs structured CSV files for analysis
# ============================================================

import os
import pandas as pd
import re

# ── Paths ──────────────────────────────────────────────────────────────────
INPUT_DIR  = r"D:\ECOLOOP\dataset"
OUTPUT_DIR = r"D:\ECOLOOP\ecoloop\iiser_processing\cleaned"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# CORE PARSING FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def parse_full_spectrum_file(filepath):
    """
    Parses a Shimadzu UV-3600i full spectrum TXT file.
    
    File format:
        "Sample Name - RawData"            ← header line 1
        "Wavelength nm.","R%"              ← header line 2
        300.00,6.8330                      ← data starts
        301.00,6.8148
        ...
    
    Returns:
        sample_name : str  — extracted from header
        df          : DataFrame with wavelength + value columns
    """

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Extract sample name from first line
    # Format: "Bruised fresh apple with skin_1 - RawData"
    # Extract sample name from first line
    # Reflectance format: "Bruised fresh apple with skin_1 - RawData"
    # Absorbance format : "Bruised fresh apple with skin_1 abs data - Bruised fresh apple with skin_1"
    sample_name = "unknown"
    if lines:
        first_line = lines[0].strip().strip('"')

        # Try reflectance pattern first
        match = re.search(r'^(.+?)\s*-\s*RawData', first_line)
        if match:
            sample_name = match.group(1).strip()
        else:
            # Try absorbance pattern: "X abs data - X"
            match = re.search(r'^(.+?)\s+abs\s+data', first_line, re.IGNORECASE)
            if match:
                sample_name = match.group(1).strip()
            else:
                # Fallback — just take whatever is before first dash
                if " - " in first_line:
                    sample_name = first_line.split(" - ")[0].strip()
                else:
                    sample_name = first_line.strip()
    # Find where actual data starts
    # Skip header lines until we find a line starting with a number
    data_start_idx = 0
    for i, line in enumerate(lines):
        line_clean = line.strip()
        # First character should be a digit (wavelength like 300.00)
        if line_clean and line_clean[0].isdigit():
            data_start_idx = i
            break

    # Parse data lines
    wavelengths = []
    values      = []

    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line:
            continue

        # Split by comma — format: "300.00,6.8330"
        parts = line.split(",")
        if len(parts) != 2:
            continue

        try:
            wl  = float(parts[0].strip())
            val = float(parts[1].strip())

            # Sanity check — wavelength should be 280-1600nm range
            if 280 <= wl <= 1600:
                wavelengths.append(wl)
                values.append(val)
        except ValueError:
            continue  # skip malformed lines

    df = pd.DataFrame({
        "wavelength_nm": wavelengths,
        "value":         values
    })

    return sample_name, df


# ══════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════

def process_all_folders():
    """
    Walks through every folder in dataset/
    Cleans each TXT file
    Saves to organized CSV format
    """

    print("\n" + "="*60)
    print("ECO-LOOP — IISER Data Cleaning")
    print("="*60)
    print(f"Input  : {INPUT_DIR}")
    print(f"Output : {OUTPUT_DIR}\n")

    # Track everything for the summary
    all_records = []
    success_count = 0
    error_count   = 0

    # Iterate through each apple folder
    for folder_name in sorted(os.listdir(INPUT_DIR)):
        folder_path = os.path.join(INPUT_DIR, folder_name)

        if not os.path.isdir(folder_path):
            continue

        print(f"\n📁 Processing: {folder_name}")

        # Determine apple type from folder name
        if "fresh" in folder_name.lower():
            apple_type = "Fresh"
        elif "rot" in folder_name.lower():
            apple_type = "Rotten"
        else:
            apple_type = "Unknown"

        # Process the 4 expected files in each folder
        files_to_process = {
            "reflectance_full":   ["reflactance_full.txt", "reflectance_full.txt"],
            "reflectance_peaks":  ["relactance_peaks.txt", "reflactance_peaks.txt", "reflectance_peaks.txt"],
            "absorbance_full":    ["absorbance_full.txt"],
            "absorbance_peaks":   ["absorbance_peaks.txt"],
        }

        for data_type, possible_filenames in files_to_process.items():

            # Find the actual file (handles spelling variations)
            target_file = None
            for filename in possible_filenames:
                potential = os.path.join(folder_path, filename)
                if os.path.exists(potential):
                    target_file = potential
                    break

            if target_file is None:
                print(f"   ❌ Missing: {data_type}")
                error_count += 1
                continue

            # Skip peak files for now (different format) — we focus on full spectrum
            if "peaks" in data_type:
                continue

            try:
                sample_name, df = parse_full_spectrum_file(target_file)

                # Save cleaned CSV
                output_filename = f"{folder_name}__{data_type}.csv"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                df.to_csv(output_path, index=False)

                print(f"   ✅ {data_type:20s} → {len(df):4d} points "
                      f"({df['wavelength_nm'].min():.0f}-"
                      f"{df['wavelength_nm'].max():.0f}nm)")

                all_records.append({
                    "folder":         folder_name,
                    "apple_type":     apple_type,
                    "data_type":      data_type,
                    "sample_name":    sample_name,
                    "wavelength_min": df["wavelength_nm"].min(),
                    "wavelength_max": df["wavelength_nm"].max(),
                    "data_points":    len(df),
                    "value_min":      df["value"].min(),
                    "value_max":      df["value"].max(),
                    "value_mean":     df["value"].mean(),
                    "output_file":    output_filename
                })

                success_count += 1

            except Exception as e:
                print(f"   ❌ Error processing {data_type}: {e}")
                error_count += 1

    # ── Save master summary ────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_records)
    summary_path = os.path.join(OUTPUT_DIR, "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # ── Final report ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CLEANING COMPLETE")
    print("="*60)
    print(f"  ✅ Successfully cleaned : {success_count} files")
    print(f"  ❌ Errors               : {error_count} files")
    print(f"\n  Apple type distribution:")
    if not summary_df.empty:
        type_counts = summary_df.drop_duplicates("folder")["apple_type"].value_counts()
        for apple_type, count in type_counts.items():
            print(f"     {apple_type:10s} : {count} folders")

    print(f"\n  Master summary saved   : {summary_path}")
    print(f"  Individual cleaned CSVs: {OUTPUT_DIR}")
    print("\nNext step: review _summary.csv to verify everything looks correct\n")


# ══════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    process_all_folders()