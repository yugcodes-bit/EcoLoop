import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib.patches import Patch

# ── Load data and model ────────────────────────────────────────────────────
df = pd.read_csv("data/synthetic_apple_data.csv")
CHANNEL_COLS = [c for c in df.columns if c.startswith("ch_")]

clf = joblib.load("models/best_classifier.pkl")

# ── Get feature importances ────────────────────────────────────────────────
importances = clf.feature_importances_

importance_df = pd.DataFrame({
    "channel": CHANNEL_COLS,
    "importance": importances
}).sort_values("importance", ascending=True)

# ── Color key diagnostic bands differently ─────────────────────────────────
wavelengths = [int(c.replace("ch_", "").replace("nm", ""))
               for c in importance_df["channel"]]

bar_colors = []
for wl in wavelengths:
    if wl == 680:
        bar_colors.append("#8e44ad")   # purple — chlorophyll
    elif wl == 810:
        bar_colors.append("#2980b9")   # blue — cell structure
    elif wl == 940:
        bar_colors.append("#27ae60")   # green — water band
    else:
        bar_colors.append("#95a5a6")   # gray — others

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Eco-Loop — Feature Importance Analysis (Random Forest Classifier)",
             fontsize=14, fontweight="bold")

# LEFT: Feature importance bar chart
ax1 = axes[0]
bars = ax1.barh(importance_df["channel"], importance_df["importance"],
                color=bar_colors, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, importance_df["importance"]):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=8)

ax1.set_title("Which Channels Did the AI Use Most?\n(Higher = More Important for Classification)",
              fontweight="bold")
ax1.set_xlabel("Feature Importance Score")
ax1.set_xlim(0, importance_df["importance"].max() * 1.25)

legend_elements = [
    Patch(facecolor="#8e44ad", label="680nm — Chlorophyll band"),
    Patch(facecolor="#2980b9", label="810nm — Cell structure band"),
    Patch(facecolor="#27ae60", label="940nm — Water band"),
    Patch(facecolor="#95a5a6", label="Other channels"),
]
ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

# RIGHT: Cumulative importance curve
ax2 = axes[1]
importance_sorted_desc = importance_df.sort_values("importance", ascending=False)
cumulative = importance_sorted_desc["importance"].cumsum().values
channels_count = list(range(1, len(cumulative) + 1))

ax2.plot(channels_count, cumulative, color="#2c3e50", linewidth=2.5,
         marker="o", markersize=5)

for threshold, color, label in [(0.80, "#e74c3c", "80%"),
                                  (0.90, "#f39c12", "90%"),
                                  (0.95, "#2ecc71", "95%")]:
    ax2.axhline(y=threshold, color=color, linestyle="--", alpha=0.7)
    n_needed = next((i+1 for i, v in enumerate(cumulative) if v >= threshold), 18)
    ax2.axvline(x=n_needed, color=color, linestyle="--", alpha=0.7)
    ax2.text(n_needed + 0.2, threshold - 0.02,
             f"{n_needed} channels\nfor {label}", fontsize=8, color=color)

ax2.set_title("Cumulative Feature Importance\n(How many channels explain X% of AI decisions?)",
              fontweight="bold")
ax2.set_xlabel("Number of Channels (ranked by importance)")
ax2.set_ylabel("Cumulative Importance")
ax2.set_ylim(0, 1.05)
ax2.set_xlim(1, 18)
ax2.set_xticks(range(1, 19))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/chart2_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart 2 saved to data/chart2_feature_importance.png")