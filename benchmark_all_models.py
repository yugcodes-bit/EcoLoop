# benchmark_all_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              r2_score, mean_squared_error, confusion_matrix,
                              classification_report)

df = pd.read_csv("data/synthetic_apple_data.csv")
CHANNEL_COLS = [c for c in df.columns if c.startswith("ch_")]

X = df[CHANNEL_COLS].values
y_class = df["class_label"].values
y_ph = df["pH"].values

X_train, X_test, yc_train, yc_test, yp_train, yp_test = train_test_split(
    X, y_class, y_ph, test_size=0.2, random_state=42, stratify=y_class
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Test set size: {len(yc_test)}\n")

print("=== CLASSIFICATION MODELS ===")
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
}
for name, clf in classifiers.items():
    clf.fit(X_train_s, yc_train)
    pred = clf.predict(X_test_s)
    acc = accuracy_score(yc_test, pred)
    prec = precision_score(yc_test, pred, average="macro", zero_division=0)
    rec = recall_score(yc_test, pred, average="macro", zero_division=0)
    print(f"\n{name}")
    print(f"  Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(yc_test, pred)}")
    print(f"  Per-class report:\n{classification_report(yc_test, pred, target_names=['Fresh','Early-Rot','Severe-Rot'], zero_division=0)}")

print("\n=== REGRESSION MODELS (pH) ===")

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_s, yp_train)
pred = rf_reg.predict(X_test_s)
rmse = np.sqrt(mean_squared_error(yp_test, pred))
print(f"Random Forest Reg | R2: {r2_score(yp_test, pred):.3f} | RMSE: {rmse:.3f}")

pls = PLSRegression(n_components=5)
pls.fit(X_train_s, yp_train)
pred = pls.predict(X_test_s).ravel()
rmse = np.sqrt(mean_squared_error(yp_test, pred))
print(f"PLSR              | R2: {r2_score(yp_test, pred):.3f} | RMSE: {rmse:.3f}")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, yp_train)
pred = ridge.predict(X_test_s)
rmse = np.sqrt(mean_squared_error(yp_test, pred))
print(f"Ridge             | R2: {r2_score(yp_test, pred):.3f} | RMSE: {rmse:.3f}")