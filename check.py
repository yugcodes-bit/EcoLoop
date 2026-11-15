import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/synthetic_apple_data.csv")
CHANNEL_COLS = [c for c in df.columns if c.startswith("ch_")]
X = df[CHANNEL_COLS].values
y = df["class_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = joblib.load("models/scaler.pkl")
clf = joblib.load("models/best_classifier.pkl")

X_test_s = scaler.transform(X_test)
pred = clf.predict(X_test_s)

from sklearn.metrics import confusion_matrix, accuracy_score
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))