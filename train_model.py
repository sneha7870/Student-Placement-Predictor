"""
train_model.py
----------------
Generates a realistic synthetic dataset of student academic/extracurricular
records and trains a RandomForestClassifier to predict placement (Yes/No).

Run this once to produce model.pkl, which app.py loads at runtime.

    python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

np.random.seed(42)

N = 3000  # number of synthetic student records

# ---- Feature generation ---------------------------------------------------
cgpa = np.clip(np.random.normal(7.2, 1.0, N), 4.0, 10.0)
tenth_pct = np.clip(np.random.normal(78, 10, N), 40, 100)
twelfth_pct = np.clip(np.random.normal(75, 10, N), 40, 100)
attendance = np.clip(np.random.normal(80, 10, N), 40, 100)
backlogs = np.random.poisson(0.6, N)
internships = np.random.poisson(1.0, N)
projects = np.random.poisson(2.0, N)
communication = np.clip(np.random.normal(6.5, 1.8, N), 1, 10)
coding_score = np.clip(np.random.normal(65, 15, N), 0, 100)  # e.g. HackerRank/LeetCode style score

df = pd.DataFrame({
    "cgpa": cgpa,
    "tenth_pct": tenth_pct,
    "twelfth_pct": twelfth_pct,
    "attendance": attendance,
    "backlogs": backlogs,
    "internships": internships,
    "projects": projects,
    "communication": communication,
    "coding_score": coding_score,
})

# ---- Target generation (weighted "placement score" + noise) ---------------
score = (
    0.35 * (df.cgpa / 10) +
    0.10 * (df.tenth_pct / 100) +
    0.10 * (df.twelfth_pct / 100) +
    0.10 * (df.attendance / 100) +
    0.15 * (df.internships / (df.internships.max() + 1)) +
    0.10 * (df.projects / (df.projects.max() + 1)) +
    0.15 * (df.communication / 10) +
    0.15 * (df.coding_score / 100) -
    0.20 * (df.backlogs / (df.backlogs.max() + 1))
)

score += np.random.normal(0, 0.05, N)  # noise
threshold = np.percentile(score, 45)   # ~55% placed, 45% not placed
df["placed"] = (score > threshold).astype(int)

print("Class balance:\n", df["placed"].value_counts(normalize=True))

# ---- Train/test split & model ----------------------------------------------
FEATURES = ["cgpa", "tenth_pct", "twelfth_pct", "attendance",
            "backlogs", "internships", "projects", "communication", "coding_score"]

X = df[FEATURES]
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

# ---- Save model + feature order --------------------------------------------
joblib.dump({"model": model, "features": FEATURES}, "model.pkl")
print("\nSaved trained model to model.pkl")
