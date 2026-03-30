import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.patches import Patch
import seaborn as sns

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/synthetic_apple_data.csv")

CHANNEL_COLS = [c for c in df.columns if c.startswith("ch_")]
CLASS_NAMES  = ["Fresh", "Early-Rot", "Severe-Rot"]

X       = df[CHANNEL_COLS].values
y_class = df["class_label"].values

# ── Same split as training — MUST match exactly ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# ── Load saved scaler and classifier ──────────────────────────────────────
scaler = joblib.load("models/scaler.pkl")
clf    = joblib.load("models/best_classifier.pkl")

X_test_sc = scaler.transform(X_test)
y_pred    = clf.predict(X_test_sc)

# ── Compute confusion matrix ───────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

# Normalize to percentages (row-wise)
# Each row = actual class, each column = predicted class
# Diagonal = correct predictions
cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Eco-Loop — Confusion Matrix Analysis (Random Forest Classifier)",
             fontsize=14, fontweight="bold")

# LEFT: Raw count confusion matrix
ax1 = axes[0]
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=ax1,
    linewidths=1,
    linecolor="white",
    cbar_kws={"label": "Number of Apples"},
    annot_kws={"size": 18, "weight": "bold"}
)
ax1.set_title("Raw Count — How Many Apples\nPredicted Correctly vs Incorrectly",
              fontweight="bold")
ax1.set_xlabel("Predicted Class", fontsize=12)
ax1.set_ylabel("Actual Class", fontsize=12)

# Diagonal explanation annotation
ax1.text(1.5, -0.3,
         "Diagonal = Correct Predictions\nOff-diagonal = Mistakes",
         ha="center", fontsize=9, color="#7f8c8d",
         transform=ax1.transData)

# RIGHT: Percentage confusion matrix
ax2 = axes[1]
sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=ax2,
    linewidths=1,
    linecolor="white",
    vmin=0, vmax=100,
    cbar_kws={"label": "Percentage (%)"},
    annot_kws={"size": 18, "weight": "bold"}
)
ax2.set_title("Percentage — What % of Each Class\nDid the Model Get Right?",
              fontweight="bold")
ax2.set_xlabel("Predicted Class", fontsize=12)
ax2.set_ylabel("Actual Class", fontsize=12)

# ── Print classification report to terminal ────────────────────────────────
print("\nDetailed Classification Report:")
print("="*55)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

print("\nConfusion Matrix (raw counts):")
print(f"Rows = Actual class | Columns = Predicted class")
print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

print("\nConfusion Matrix (percentages):")
print(pd.DataFrame(cm_percent.round(1),
                   index=CLASS_NAMES,
                   columns=CLASS_NAMES))

plt.tight_layout()
plt.savefig("data/chart3_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart 3 saved to data/chart3_confusion_matrix.png")