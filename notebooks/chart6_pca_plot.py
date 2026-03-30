import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/synthetic_apple_data.csv")

CHANNEL_COLS = [c for c in df.columns if c.startswith("ch_")]
CLASS_NAMES  = {0: "Fresh", 1: "Early-Rot", 2: "Severe-Rot"}
COLORS       = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

X      = df[CHANNEL_COLS].values
y      = df["class_label"].values
y_ph   = df["pH"].values

# ── Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Apply PCA ──────────────────────────────────────────────────────────────
# PCA compresses 18 dimensions down to 2 so we can visualize it
# PC1 = direction of maximum variance in the data
# PC2 = direction of second most variance, perpendicular to PC1
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# How much of the total information each component captures
var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
total_var = var_pc1 + var_pc2

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Eco-Loop — PCA Analysis (Principal Component Analysis)\n"
             f"18 spectral channels compressed to 2 dimensions "
             f"(capturing {total_var:.1f}% of total data variance)",
             fontsize=13, fontweight="bold")

# LEFT: PCA scatter plot colored by rot class
ax1 = axes[0]

for class_label, class_name in CLASS_NAMES.items():
    mask = y == class_label
    ax1.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        c=COLORS[class_label],
        label=f"{class_name} (n={mask.sum()})",
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
        s=80
    )

    # Draw an ellipse around each cluster to show the boundary
    from matplotlib.patches import Ellipse
    cluster_points = X_pca[mask]
    mean_x = cluster_points[:, 0].mean()
    mean_y = cluster_points[:, 1].mean()
    std_x  = cluster_points[:, 0].std() * 2
    std_y  = cluster_points[:, 1].std() * 2

    ellipse = Ellipse(
        (mean_x, mean_y), width=std_x*2, height=std_y*2,
        edgecolor=COLORS[class_label],
        facecolor=COLORS[class_label],
        alpha=0.1, linewidth=2, linestyle="--"
    )
    ax1.add_patch(ellipse)

    # Label cluster center
    ax1.annotate(class_name,
                 xy=(mean_x, mean_y),
                 fontsize=11, fontweight="bold",
                 color=COLORS[class_label],
                 ha="center", va="center")

ax1.set_title(f"3 Rot Classes in 2D PCA Space\n"
              f"PC1={var_pc1:.1f}% variance | PC2={var_pc2:.1f}% variance",
              fontweight="bold")
ax1.set_xlabel(f"Principal Component 1 ({var_pc1:.1f}% variance explained)")
ax1.set_ylabel(f"Principal Component 2 ({var_pc2:.1f}% variance explained)")
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
ax1.axvline(x=0, color="black", linewidth=0.5, alpha=0.3)

# RIGHT: PCA scatter plot colored by pH value (continuous)
# This shows that the PCA separation also corresponds to pH gradient
ax2 = axes[1]

scatter = ax2.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_ph,
    cmap="RdYlGn",
    alpha=0.85,
    edgecolors="white",
    linewidth=0.5,
    s=80,
    vmin=y_ph.min(),
    vmax=y_ph.max()
)

cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label("Internal pH Value", fontsize=10)

ax2.set_title("Same PCA Space — Colored by pH Value\n"
              "(Green = High pH Fresh | Red = Low pH Severe-Rot)",
              fontweight="bold")
ax2.set_xlabel(f"Principal Component 1 ({var_pc1:.1f}% variance explained)")
ax2.set_ylabel(f"Principal Component 2 ({var_pc2:.1f}% variance explained)")
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
ax2.axvline(x=0, color="black", linewidth=0.5, alpha=0.3)

# ── Print PCA summary to terminal ──────────────────────────────────────────
print("\nPCA Summary:")
print("="*50)
print(f"PC1 explains : {var_pc1:.2f}% of total variance")
print(f"PC2 explains : {var_pc2:.2f}% of total variance")
print(f"Total (2D)   : {total_var:.2f}% of total variance")
print(f"\nThis means {total_var:.1f}% of all information in")
print(f"18 spectral channels is captured in just 2 dimensions.")

print("\nPCA Component Loadings (which channels drive PC1 and PC2):")
loadings = pd.DataFrame(
    pca.components_.T,
    index=CHANNEL_COLS,
    columns=["PC1_loading", "PC2_loading"]
).round(3)
print(loadings.sort_values("PC1_loading", ascending=False).to_string())

plt.tight_layout()
plt.savefig("data/chart5_pca_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart 5 saved to data/chart5_pca_plot.png")