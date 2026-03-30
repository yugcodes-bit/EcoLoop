import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/synthetic_apple_data.csv")

CHANNEL_COLS = [c for c in df.columns if c.startswith("ch_")]

# ── CHART 1: Correlation Heatmap ───────────────────────────────────────────
# This shows how strongly each of the 18 channels correlates with pH
# Correlation ranges from -1 to +1
# +1 = as channel goes up, pH goes up (positive relationship)
# -1 = as channel goes up, pH goes down (negative relationship)
#  0 = no relationship at all

# Build a dataframe with just channels + pH
corr_df = df[CHANNEL_COLS + ["pH"]].copy()

# Calculate correlation of every column with every other column
corr_matrix = corr_df.corr()

# We specifically want the last row — correlation of each channel WITH pH
ph_correlation = corr_matrix["pH"].drop("pH").sort_values()

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Eco-Loop — Channel Correlation Analysis", 
             fontsize=14, fontweight="bold")

# LEFT: Full correlation heatmap (all channels vs all channels + pH)
ax1 = axes[0]
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True  # hide upper triangle (duplicate)

sns.heatmap(
    corr_matrix,
    ax=ax1,
    cmap="RdYlGn",
    center=0,
    vmin=-1, vmax=1,
    annot=False,
    linewidths=0.3,
    linecolor="white",
    cbar_kws={"label": "Correlation Coefficient"}
)
ax1.set_title("Full Correlation Matrix\n(All 18 Channels + pH)", 
              fontweight="bold")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=8)

# RIGHT: Bar chart — correlation of each channel specifically with pH
ax2 = axes[1]

colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in ph_correlation.values]

bars = ax2.barh(ph_correlation.index, ph_correlation.values, color=colors, 
                edgecolor="white", linewidth=0.5)

# Add value labels on bars
for bar, val in zip(bars, ph_correlation.values):
    ax2.text(val + (0.01 if val >= 0 else -0.01),
             bar.get_y() + bar.get_height()/2,
             f"{val:.3f}",
             va="center",
             ha="left" if val >= 0 else "right",
             fontsize=8)

ax2.axvline(x=0, color="black", linewidth=0.8)
ax2.set_title("Correlation of Each Channel with pH\n(sorted weakest → strongest)", 
              fontweight="bold")
ax2.set_xlabel("Pearson Correlation Coefficient")
ax2.set_xlim(-1.1, 1.1)

# Mark the 3 key diagnostic bands
key_bands = ["ch_680nm", "ch_810nm", "ch_940nm"]
for label in ax2.get_yticklabels():
    if label.get_text() in key_bands:
        label.set_fontweight("bold")
        label.set_color("#8e44ad")

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ecc71", label="Positive correlation with pH"),
    Patch(facecolor="#e74c3c", label="Negative correlation with pH")
]
ax2.legend(handles=legend_elements, loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("data/chart1_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart 1 saved to data/chart1_correlation_heatmap.png")












