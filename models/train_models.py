import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/synthetic_apple_data.csv")

CHANNEL_COLS = [c for c in df.columns if c.startswith("ch_")]

X = df[CHANNEL_COLS].values       # 18 spectral values → input features
y_class = df["class_label"].values # 0/1/2 → classification target
y_ph    = df["pH"].values          # pH float → regression target

# ── Train/Test split ───────────────────────────────────────────────────────
# 80% training, 20% testing — stratified so each class is evenly split
X_train, X_test, yc_train, yc_test, yp_train, yp_test = train_test_split(
    X, y_class, y_ph,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Features         : {X_train.shape[1]} spectral channels")

# ── Scale features ─────────────────────────────────────────────────────────
# SVM and Ridge need scaling — RF does not but we scale all for consistency
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Save scaler — needed later in FastAPI to scale new apple readings
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════
# PART A — CLASSIFIERS (predict rot class 0/1/2)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("CLASSIFICATION RESULTS")
print("="*55)

CLASS_NAMES = ["Fresh", "Early-Rot", "Severe-Rot"]

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":           SVC(kernel="rbf", C=1.0, random_state=42),
    "kNN":           KNeighborsClassifier(n_neighbors=5),
}

best_clf       = None
best_clf_score = 0
best_clf_name  = ""

for name, clf in classifiers.items():
    # Train
    clf.fit(X_train_sc, yc_train)

    # Predict on test set
    y_pred = clf.predict(X_test_sc)

    # Cross validation score (5-fold) — more reliable than single test
    cv_scores = cross_val_score(clf, X_train_sc, yc_train, cv=5, scoring="f1_weighted")

    print(f"\n── {name} ──")
    print(f"  CV F1 Score (5-fold) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Test Accuracy        : {(y_pred == yc_test).mean():.3f}")
    print(classification_report(yc_test, y_pred, target_names=CLASS_NAMES))

    # Track best classifier
    if cv_scores.mean() > best_clf_score:
        best_clf_score = cv_scores.mean()
        best_clf       = clf
        best_clf_name  = name

# Save best classifier
joblib.dump(best_clf, "models/best_classifier.pkl")
print(f"\n✅ Best Classifier → {best_clf_name} (F1: {best_clf_score:.3f})")
print(f"   Saved to models/best_classifier.pkl")

# ══════════════════════════════════════════════════════════════════════════
# PART B — REGRESSORS (predict pH value)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("REGRESSION RESULTS (pH Prediction)")
print("="*55)

regressors = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "PLS Regression": PLSRegression(n_components=3),
    "Ridge":          Ridge(alpha=1.0),
}

best_reg      = None
best_reg_mae  = 999
best_reg_name = ""

for name, reg in regressors.items():
    # Train
    reg.fit(X_train_sc, yp_train)

    # Predict
    y_pred_ph = reg.predict(X_test_sc)

    # PLS returns 2D array — flatten it
    if hasattr(y_pred_ph, "ravel"):
        y_pred_ph = y_pred_ph.ravel()

    mae = mean_absolute_error(yp_train, reg.predict(X_train_sc).ravel()
                              if hasattr(reg.predict(X_train_sc), "ravel")
                              else reg.predict(X_train_sc))
    test_mae = mean_absolute_error(yp_test, y_pred_ph)
    r2       = r2_score(yp_test, y_pred_ph)

    print(f"\n── {name} ──")
    print(f"  Test MAE : {test_mae:.3f} pH")
    print(f"  Test R²  : {r2:.3f}")

    # Track best regressor (lowest MAE)
    if test_mae < best_reg_mae:
        best_reg_mae  = test_mae
        best_reg      = reg
        best_reg_name = name

# Save best regressor
joblib.dump(best_reg, "models/best_regressor.pkl")
print(f"\n✅ Best Regressor → {best_reg_name} (MAE: {best_reg_mae:.3f})")
print(f"   Saved to models/best_regressor.pkl")

print("\n✅ All models trained and saved to models/")