import pandas as pd

INPUT_FILE = r"D:\ECOLOOP\ecoloop\iiser_processing\master_iiser_dataset.csv"

df = pd.read_csv(INPUT_FILE)

refl = df[(df["measurement"] == "reflectance") &
          (df["apple_type"] == "Fresh") &
          (df["wavelength_nm"].between(700, 1000))]

# Average reflectance in 700-1000nm for each Fresh scan
print("\nFresh scan averages in 700-1000nm region:")
print("="*65)

for folder in sorted(refl["folder"].unique()):
    scan = refl[refl["folder"] == folder]
    sample_name = scan["sample_name"].iloc[0]
    position    = scan["position"].iloc[0]
    skin_status = "WITHOUT" if "without skin" in sample_name.lower() else "WITH"
    mean_val    = scan["intensity"].mean()

    print(f"  {folder:18s} | {position:8s} | {skin_status:7s} skin | "
          f"mean: {mean_val:.2f}% | {sample_name}")

# Calculate group average to spot the outlier
mean_all = refl.groupby("folder")["intensity"].mean()
group_mean = mean_all.mean()
print(f"\n  Group mean (all Fresh): {group_mean:.2f}%")
print(f"  Outliers (more than 8% below group mean):")
for folder, val in mean_all.items():
    if val < group_mean - 8:
        print(f"     ⚠️  {folder} : {val:.2f}% (gap: {group_mean - val:.2f}%)")