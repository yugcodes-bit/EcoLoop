# ============================================================
# ECO-LOOP — Sample Diagnosis
# diagnose_samples.py
#
# Identifies skin vs no-skin scans
# Identifies the outlier
# Tells us exactly what we have before extended analysis
# ============================================================

import pandas as pd

INPUT_FILE = r"D:\ECOLOOP\ecoloop\iiser_processing\master_iiser_dataset.csv"

print("\n" + "="*60)
print("ECO-LOOP — Sample Diagnosis")
print("="*60)

df = pd.read_csv(INPUT_FILE)

# Filter to reflectance only and 300-1500nm range
refl = df[(df["measurement"] == "reflectance") &
          (df["wavelength_nm"] >= 300) & 
          (df["wavelength_nm"] <= 1500)].copy()

# ── List every unique scan with its sample name ───────────────────────────
print("\n--- ALL SCANS WITH SAMPLE NAMES ---\n")

scans = refl[["folder", "apple_type", "position", "sample_name", 
              "physical_apple"]].drop_duplicates()

# Detect skin vs no-skin from sample name
def detect_skin(name):
    name_lower = str(name).lower()
    if "without skin" in name_lower or "no skin" in name_lower or "flesh" in name_lower:
        return "WITHOUT_SKIN"
    elif "with skin" in name_lower or "skin" in name_lower:
        return "WITH_SKIN"
    else:
        return "UNCLEAR"

scans["skin_status"] = scans["sample_name"].apply(detect_skin)

print(scans.to_string(index=False))

# ── Distribution ───────────────────────────────────────────────────────────
print("\n--- SKIN STATUS DISTRIBUTION ---\n")
print(scans.groupby(["apple_type", "skin_status"]).size().to_string())

# ── Find the outlier — scans where reflectance flatlines near zero ───────
print("\n--- OUTLIER DETECTION ---")
print("Looking for scans where 700-1000nm reflectance < 5% (flatlined)...")

outliers = []
for folder in refl["folder"].unique():
    scan = refl[refl["folder"] == folder]
    nir_region = scan[(scan["wavelength_nm"] >= 700) & 
                      (scan["wavelength_nm"] <= 1000)]
    mean_nir = nir_region["intensity"].mean()
    
    if mean_nir < 5:
        sample_name = scan["sample_name"].iloc[0]
        apple_type  = scan["apple_type"].iloc[0]
        outliers.append({
            "folder": folder,
            "apple_type": apple_type,
            "sample_name": sample_name,
            "mean_700_1000nm": round(mean_nir, 3)
        })

if outliers:
    print("\n⚠️  Outlier scans found:")
    for o in outliers:
        print(f"   {o['folder']:20s} | {o['apple_type']:8s} | "
              f"NIR mean: {o['mean_700_1000nm']:.2f}% | {o['sample_name']}")
else:
    print("   No flatlined outliers detected")

# ── Save sample mapping to CSV ────────────────────────────────────────────
output = r"D:\ECOLOOP\ecoloop\iiser_processing\sample_mapping.csv"
scans.to_csv(output, index=False)
print(f"\n  Sample mapping saved: {output}")
print()